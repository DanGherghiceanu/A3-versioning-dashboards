"""
A3 Step 1 - Dataset versioning with MLflow.

Creates two dataset versions and records each one in MLflow:

  v1  the exact A1 data: same images, same 85/15 split (seed 42), same test set.
      Read from A1's data/processed/split_manifest.json, so nothing is re-split.

  v2  a *drifted* copy of v1: every image passed through a simulated
      "new scanner" (lower contrast, brighter, slightly blurred). Same files,
      same labels, same split - only the pixels change. That isolates drift
      as the single difference between the versions.

For each version the script logs one MLflow run in the experiment
"xray-datasets" containing:
  - the train/val/test splits as MLflow Datasets (name + content digest)
  - params  : version, parent version, seed, drift settings
  - metrics : image counts, class balance, image statistics, drift scores
  - files   : manifest.csv, dataset card, histograms, sample images

Re-running is safe: if a version with the same digest is already logged,
it is skipped instead of creating a duplicate.

Usage (from the a3-versioning folder, venv active, docker compose up):
    python scripts/01_version_datasets.py --a1-root "E:\\...\\AI_ML_Vanier_A1"
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write PNGs without opening windows
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

# Two harmless MLflow notices about file:// sources and integer columns.
warnings.filterwarnings("ignore", message=".*can be interpreted in multiple ways.*")
warnings.filterwarnings("ignore", message=".*integer column.*")

SEED = 42
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
EXPERIMENT = "xray-datasets"
A3_ROOT = Path(__file__).resolve().parent.parent
VERSIONS_DIR = A3_ROOT / "data" / "versions"

# The drift applied to make v2. Changing any value = a new dataset version.
DRIFT = {
    "contrast_factor": 0.6,    # 1.0 = unchanged, <1 flattens the image
    "brightness_factor": 1.15, # 1.0 = unchanged, >1 brightens
    "blur_rel_radius": 0.003,  # blur radius as a fraction of the longer side
}


# ----------------------------------------------------------------------
# Image helpers (top-level functions so Windows multiprocessing can use them)
# ----------------------------------------------------------------------
def image_stats(img: Image.Image) -> dict:
    """Size and pixel statistics of a grayscale image (0-255 scale)."""
    w, h = img.size
    small = img.copy()
    small.thumbnail((512, 512))  # stats on a thumbnail: same answer, much faster
    arr = np.asarray(small, dtype=np.float32)
    return {
        "width": w,
        "height": h,
        "mean_intensity": float(arr.mean()),
        "std_intensity": float(arr.std()),  # a simple contrast measure
    }


def stats_for_file(path: str) -> dict:
    with Image.open(path) as im:
        return image_stats(im.convert("L"))


def apply_drift(img: Image.Image, drift: dict) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(drift["contrast_factor"])
    img = ImageEnhance.Brightness(img).enhance(drift["brightness_factor"])
    radius = drift["blur_rel_radius"] * max(img.size)
    return img.filter(ImageFilter.GaussianBlur(radius))


def drift_one(job: tuple[str, str, dict]) -> dict:
    """Create one drifted image (unless it already exists) and return its stats."""
    src, dst, drift = job
    dst_path = Path(dst)
    if not dst_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            apply_drift(im.convert("L"), drift).save(dst_path, quality=95)
    return stats_for_file(dst)


# ----------------------------------------------------------------------
# Building the version tables
# ----------------------------------------------------------------------
def load_v1(a1_root: Path) -> tuple[pd.DataFrame, Path]:
    """Turn A1's split manifest into one row per image."""
    manifest_path = a1_root / "data" / "processed" / "split_manifest.json"
    raw_dir = a1_root / "data" / "raw" / "chest_xray_pneumonia"
    with open(manifest_path) as f:
        m = json.load(f)
    if m.get("seed") != SEED:
        raise SystemExit(f"Manifest seed is {m.get('seed')}, expected {SEED}")

    rows = []
    for split in ("train", "val", "test"):
        for p, label in zip(m[split], m[f"{split}_labels"]):
            p = Path(p)
            if not p.is_absolute():
                p = a1_root / p
            try:
                rel = p.relative_to(raw_dir)
            except ValueError:
                raise SystemExit(
                    f"Manifest path is not under {raw_dir}:\n  {p}\n"
                    "Was the A1 folder moved after the manifest was written? "
                    "Check --a1-root."
                )
            rows.append({
                # path relative to the version's root, with / so it is OS-neutral
                "relpath": rel.as_posix(),
                "split": split,
                "label": int(label),
                "label_name": CLASS_NAMES[int(label)],
            })
    df = pd.DataFrame(rows).sort_values(["split", "relpath"]).reset_index(drop=True)

    missing = [r for r in df.relpath if not (raw_dir / r).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images in the manifest are missing, e.g. {missing[0]}")
    return df, raw_dir


def add_stats(df: pd.DataFrame, root: Path, workers: int) -> pd.DataFrame:
    paths = [str(root / r) for r in df.relpath]
    with ProcessPoolExecutor(workers) as ex:
        stats = list(ex.map(stats_for_file, paths, chunksize=32))
    return pd.concat([df, pd.DataFrame(stats)], axis=1)


def build_v2(v1: pd.DataFrame, v1_root: Path, workers: int) -> tuple[pd.DataFrame, Path]:
    v2_root = VERSIONS_DIR / "v2" / "images"
    jobs = [(str(v1_root / r), str(v2_root / r), DRIFT) for r in v1.relpath]
    with ProcessPoolExecutor(workers) as ex:
        stats = list(ex.map(drift_one, jobs, chunksize=16))
    base = v1[["relpath", "split", "label", "label_name"]]
    return pd.concat([base, pd.DataFrame(stats)], axis=1), v2_root


# ----------------------------------------------------------------------
# Drift measurement
# ----------------------------------------------------------------------
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.25 moderate, >0.25 major shift.
    Bins come from the reference (v1) distribution's deciles."""
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


# ----------------------------------------------------------------------
# Logging one version to MLflow
# ----------------------------------------------------------------------
def summary_metrics(df: pd.DataFrame) -> dict:
    out = {"n_images": len(df)}
    for split, g in df.groupby("split"):
        out[f"n_{split}"] = len(g)
        out[f"pct_pneumonia_{split}"] = round(100 * g.label.mean(), 2)
    for col in ("mean_intensity", "std_intensity", "width", "height"):
        out[f"{col}_avg"] = round(float(df[col].mean()), 3)
    return out


def histogram_png(df: pd.DataFrame, ref: pd.DataFrame | None, path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, label in zip(axes, ("mean_intensity", "std_intensity"),
                              ("Mean intensity (brightness)", "Std of intensity (contrast)")):
        if ref is not None:
            ax.hist(ref[col], bins=50, alpha=0.5, label="v1 (reference)")
        ax.hist(df[col], bins=50, alpha=0.6, label=title)
        ax.set_xlabel(label)
        ax.set_ylabel("images")
        ax.legend()
    fig.suptitle(f"Image statistics - {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def samples_png(v1_root: Path, v2_root: Path, df: pd.DataFrame, path: Path):
    picks = pd.concat([df[(df.split == "test") & (df.label == k)].head(2) for k in (0, 1)])
    fig, axes = plt.subplots(2, len(picks), figsize=(3 * len(picks), 6.5))
    for i, r in enumerate(picks.itertuples()):
        for row, (root, name) in enumerate(((v1_root, "v1"), (v2_root, "v2"))):
            with Image.open(root / r.relpath) as im:
                axes[row, i].imshow(im.convert("L"), cmap="gray", vmin=0, vmax=255)
            axes[row, i].set_title(f"{name} - {r.label_name}", fontsize=9)
            axes[row, i].axis("off")
    fig.suptitle("Same test images before (v1) and after (v2) the simulated scanner drift")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def dataset_card(version: str, root: Path, metrics: dict, parent: str | None) -> str:
    lines = [
        f"# Chest X-ray dataset - {version}",
        "",
        "Source: Kaggle paultimothymooney/chest-xray-pneumonia (CC BY 4.0).",
        f"Image root on the machine that logged it: `{root}`",
        f"Split: A1 re-split, stratified 85/15 train/val, seed {SEED}; original Kaggle test set.",
    ]
    if parent:
        lines += ["", f"Derived from **{parent}** by a simulated scanner drift:",
                  *[f"- `{k}` = {v}" for k, v in DRIFT.items()],
                  "", "Labels and split membership are identical to the parent; only pixels differ."]
    lines += ["", "## Summary", "", "| metric | value |", "|---|---|",
              *[f"| {k} | {v} |" for k, v in metrics.items()]]
    return "\n".join(lines) + "\n"


def already_logged(version: str, digest: str) -> bool:
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        filter_string=f"tags.dataset_version = '{version}'",
    )
    if runs.empty:
        return False
    if (runs["tags.dataset_digest"] == digest).any():
        return True
    raise SystemExit(
        f"{version} is already logged with different contents (digest "
        f"{runs['tags.dataset_digest'].iloc[0]} vs {digest}).\n"
        "Data changed under an existing version name - give it a new version instead."
    )


def log_version(version: str, df: pd.DataFrame, root: Path, parent: str | None,
                ref: pd.DataFrame | None, extra_metrics: dict, v1_root: Path):
    out_dir = VERSIONS_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    # One MLflow Dataset per split. The digest is a hash of the table contents,
    # so any change to files, labels or pixels (via the stats) changes it.
    datasets = {
        split: mlflow.data.from_pandas(
            df[df.split == split].reset_index(drop=True),
            source=root.resolve().as_uri(),
            name=f"chest-xray-{version}-{split}",
            targets="label",
        )
        for split in ("train", "val", "test")
    }
    full_digest = mlflow.data.from_pandas(df, source=root.resolve().as_uri()).digest

    if already_logged(version, full_digest):
        print(f"[{version}] already in MLflow with digest {full_digest} - skipping")
        return

    metrics = {**summary_metrics(df), **extra_metrics}
    manifest = out_dir / "manifest.csv"
    df.to_csv(manifest, index=False)
    (out_dir / "dataset_card.md").write_text(dataset_card(version, root, metrics, parent))
    histogram_png(df, ref, out_dir / "intensity_hist.png", version)

    with mlflow.start_run(run_name=f"dataset-{version}"):
        mlflow.set_tags({
            "dataset_version": version,
            "dataset_digest": full_digest,
            "parent_version": parent or "none",
            "mlflow.note.content": (
                f"Dataset {version}. " + ("Original A1 data." if not parent
                                          else f"Drifted copy of {parent}.")
            ),
        })
        mlflow.log_params({"dataset_version": version, "seed": SEED,
                           "parent_version": parent or "none",
                           "image_root": str(root)})
        if parent:
            mlflow.log_params({f"drift_{k}": v for k, v in DRIFT.items()})
        for split, ds in datasets.items():
            mlflow.log_input(ds, context=split)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(manifest))
        mlflow.log_artifact(str(out_dir / "dataset_card.md"))
        mlflow.log_artifact(str(out_dir / "intensity_hist.png"))
        if parent:
            samples_png(v1_root, root, df, out_dir / "samples_v1_vs_v2.png")
            mlflow.log_artifact(str(out_dir / "samples_v1_vs_v2.png"))

    print(f"[{version}] logged: {len(df)} images, digest {full_digest}")
    for k, v in metrics.items():
        print(f"    {k:28s} {v}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a1-root", required=True, type=Path,
                    help="A1 project folder (contains data/processed/split_manifest.json)")
    ap.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    print(f"MLflow: {args.tracking_uri}  experiment: {EXPERIMENT}")

    print("Reading A1 split manifest ...")
    v1, v1_root = load_v1(args.a1_root)
    print(f"  {len(v1)} images. Computing image statistics ({args.workers} workers) ...")
    v1 = add_stats(v1, v1_root, args.workers)

    print("Building v2 (drifted copy) - first run writes ~5,900 images, a few minutes ...")
    v2, v2_root = build_v2(v1, v1_root, args.workers)

    t1, t2 = v1[v1.split == "test"], v2[v2.split == "test"]
    drift_metrics = {
        "psi_mean_intensity": round(psi(v1.mean_intensity.values, v2.mean_intensity.values), 4),
        "psi_std_intensity": round(psi(v1.std_intensity.values, v2.std_intensity.values), 4),
        "psi_mean_intensity_test": round(psi(t1.mean_intensity.values, t2.mean_intensity.values), 4),
    }

    log_version("v1", v1, v1_root, parent=None, ref=None, extra_metrics={}, v1_root=v1_root)
    log_version("v2", v2, v2_root, parent="v1", ref=v1, extra_metrics=drift_metrics, v1_root=v1_root)
    print(f"\nDone. Open {args.tracking_uri} -> Experiments -> {EXPERIMENT}")


if __name__ == "__main__":  # required on Windows for the worker processes
    main()

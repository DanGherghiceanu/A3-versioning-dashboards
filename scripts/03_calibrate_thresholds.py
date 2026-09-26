"""
A3 Step 2b - choose each model's decision threshold honestly.

Picking the threshold by sliding it on the TEST set and keeping the best one
is a form of test-set leakage: the reported score is then optimistic,
because the test set was used to make a decision. The honest procedure is:

  1. predict on the VALIDATION split of a dataset version
  2. choose the threshold that maximises macro F1 on validation
  3. score the untouched TEST split at that threshold

For every registered model version and every dataset version this script
does exactly that, and logs one MLflow run per model version
(experiment "xray-models", tag model_role = calibration) containing:
  - metrics : <dv>_threshold, val_<dv>_f1_macro_at_threshold,
              test_<dv>_<metric>_at_threshold    (the honest result)
              test_<dv>_oracle_threshold / _oracle_f1_macro (best possible on
              test - reported only to show how optimistic peeking would be)
  - files   : predictions/val_<dv>.csv (per-image validation predictions)
It also tags the model version in the registry with recommended_threshold_<dv>,
so the threshold travels with the model.

Test predictions are not recomputed - they were stored by Step 2.

Usage (venv active, docker compose up, Steps 1-2 done):
    python scripts/03_calibrate_thresholds.py
Takes ~1 minute per model per dataset on CPU. Re-running skips calibrated
versions; add --force to redo them.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
step2 = importlib.import_module("02_register_models")  # reuse Step 2's data + model code

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

# Candidate thresholds: even steps in log-odds, so the grid is as fine near
# 0.99 and 0.999 as it is near 0.5. Covers 0.0025 ... 0.999994.
GRID = 1 / (1 + np.exp(-np.arange(-6, 12.0001, 0.05)))
DEFAULT = step2.THRESHOLD


def scores_at(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = (p >= t).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tp = int(((pred == 1) & (y == 1)).sum())
    f1_pos = 2 * tp / max(2 * tp + fp + fn, 1)
    f1_neg = 2 * tn / max(2 * tn + fn + fp, 1)
    return {"accuracy": (tp + tn) / len(y), "f1_macro": (f1_pos + f1_neg) / 2,
            "recall_normal": tn / max(tn + fp, 1), "recall_pneumonia": tp / max(tp + fn, 1),
            "pct_predicted_pneumonia": 100 * pred.mean()}


def f1_curve(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Macro F1 at every threshold in GRID (vectorised)."""
    pred = p[None, :] >= GRID[:, None]
    pos, neg = (y == 1)[None, :], (y == 0)[None, :]
    tp = (pred & pos).sum(1); fp = (pred & neg).sum(1)
    fn = (~pred & pos).sum(1); tn = (~pred & neg).sum(1)
    f1_pos = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    f1_neg = 2 * tn / np.maximum(2 * tn + fn + fp, 1)
    return (f1_pos + f1_neg) / 2


def best_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Threshold with the highest macro F1. On a plateau of equal scores,
    take its middle rather than its edge - less sensitive to noise."""
    f1 = f1_curve(y, p)
    top = np.flatnonzero(f1 >= f1.max() - 1e-12)
    i = top[len(top) // 2]
    return float(GRID[i]), float(f1[i])


def test_predictions(run_id: str, dv: str) -> pd.DataFrame:
    for f in mlflow.MlflowClient().list_artifacts(run_id, "predictions"):
        if f.path.endswith(f"_test_{dv}.csv"):
            return pd.read_csv(mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=f.path, dst_path=tempfile.mkdtemp()))
    raise SystemExit(f"No stored test predictions for data {dv} in run {run_id} - rerun Step 2.")


def already_calibrated(version: int) -> bool:
    runs = mlflow.search_runs(experiment_names=[step2.MODEL_EXPERIMENT],
                              filter_string=f"tags.model_role = 'calibration' and "
                                            f"tags.calibrated_model_version = '{version}'")
    return not runs.empty


def calibrate(role: str, version: int, datasets: dict, device, workers: int) -> list[dict]:
    client = mlflow.MlflowClient()
    source_run = client.get_model_version(step2.MODEL_NAME, str(version)).run_id
    label = step2.LABELS.get(role, role)
    print(f"[v{version} {label}] loading from the registry ...")
    model = mlflow.pytorch.load_model(f"models:/{step2.MODEL_NAME}/{version}",
                                      map_location="cpu").to(device)
    rows = []
    with mlflow.start_run(run_name=f"calibrate-v{version} ({label})"):
        mlflow.set_tags({"model_role": "calibration", "calibrated_model_version": str(version),
                         "calibrated_role": role, "source_run_id": source_run,
                         "mlflow.note.content": "Decision threshold chosen on the validation split "
                                                "(max macro F1), then scored on the untouched test split."})
        mlflow.log_params({"model_version": version, "selection_split": "val",
                           "selection_metric": "f1_macro", "grid": "sigmoid(-6..12 step 0.05)",
                           "default_threshold": DEFAULT})
        out_dir = Path(tempfile.mkdtemp())
        for dv, ds in datasets.items():
            t0 = time.time()
            val = ds.split("val")
            p_val = step2.predict(model, step2.loader(val, ds.root, step2.eval_transform, False, workers), device)
            ds.log_as_input("val", context=f"calibration_{dv}")
            f = out_dir / f"val_{dv}.csv"
            val[["relpath", "label"]].assign(prob_pneumonia=p_val.round(7)).to_csv(f, index=False)
            mlflow.log_artifact(str(f), "predictions")

            t_star, f1_val = best_threshold(val.label.values, p_val)
            test = test_predictions(source_run, dv)
            y, p = test.label.values, test.prob_pneumonia.values
            at_default, at_star = scores_at(y, p, DEFAULT), scores_at(y, p, t_star)
            t_oracle, f1_oracle = best_threshold(y, p)

            mlflow.log_metrics({
                f"{dv}_threshold": t_star,
                f"val_{dv}_f1_macro_at_threshold": f1_val,
                **{f"test_{dv}_{k}_at_threshold": v for k, v in at_star.items()},
                f"test_{dv}_oracle_threshold": t_oracle,
                f"test_{dv}_oracle_f1_macro": f1_oracle,
            })
            client.set_model_version_tag(step2.MODEL_NAME, str(version),
                                         f"recommended_threshold_{dv}", f"{t_star:.6f}")
            rows.append({"model": f"v{version} {label}", "data": dv, "threshold (val)": t_star,
                         "val F1": f1_val, "test F1 @0.50": at_default["f1_macro"],
                         "test F1 @threshold": at_star["f1_macro"],
                         "NORMAL recall @threshold": at_star["recall_normal"],
                         "PNEUMONIA recall @threshold": at_star["recall_pneumonia"],
                         "test F1 oracle": f1_oracle, "oracle threshold": t_oracle})
            print(f"    data {dv}: threshold {t_star:.4f} (val F1 {f1_val:.3f}) -> test F1 "
                  f"{at_default['f1_macro']:.3f} @0.50, {at_star['f1_macro']:.3f} @threshold, "
                  f"{f1_oracle:.3f} oracle ({time.time() - t0:.0f}s)")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="recalibrate versions already calibrated")
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(step2.MODEL_EXPERIMENT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {"v1": step2.DatasetVersion("v1"), "v2": step2.DatasetVersion("v2")}

    rows = []
    for role in ("baseline", "retrained_for_drift", "control"):
        version = step2.registered_version_for(role)
        if version is None:
            continue
        if already_calibrated(version) and not args.force:
            print(f"[v{version}] already calibrated - skipping (use --force to redo)")
            continue
        rows += calibrate(role, version, datasets, device, args.workers)

    if rows:
        table = pd.DataFrame(rows)
        step2.REPORTS.mkdir(exist_ok=True)
        table.to_csv(step2.REPORTS / "threshold_calibration.csv", index=False)
        print("\nThresholds chosen on VALIDATION, scored on TEST:")
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\n'oracle' = best threshold found by peeking at the test set. It is shown only to\n"
              "measure how optimistic that would be - it is not a fair result.")
    print(f"\nDone. Model versions now carry recommended_threshold_v1 / _v2 tags: "
          f"{args.tracking_uri}/#/models/{step2.MODEL_NAME}")


if __name__ == "__main__":
    main()

"""
A3 Step 2 - Model versioning with the MLflow Model Registry.

Registered model: "xray-pneumonia-classifier"

  version 1  the A1 ResNet18, exactly as trained in A1 on dataset v1.
             Nothing is retrained - the checkpoint is imported, evaluated and
             registered, with a link back to the dataset it was trained on.

  version 2  version 1 fine-tuned on dataset v2 (the drifted "new scanner"
             images) - what a team would do after the input data shifts.

  version 3  CONTROL: version 1 fine-tuned on the ORIGINAL dataset v1 with the
             exact same recipe (epochs, lr, seed). Version 2 differs from
             version 1 in two ways - more training AND drifted data - so on its
             own the comparison cannot say which one helped. The control
             differs from version 2 only in the data, which splits the effect:
               extra training effect = control - v1
               drifted-data effect   = v2 - control

Both versions are evaluated on BOTH test sets (v1 and v2), so the comparison
answers two questions:
  - how much does the old model suffer on drifted data?      (v1 model, v2 test)
  - does retraining fix it, and what does it cost on old data? (v2 model, both)

Every run records which dataset versions it used (as MLflow Dataset inputs,
looked up from the Step 1 runs), so each model version traces back to the
exact data digest it was trained and tested on.

Usage (venv active, docker compose up, Step 1 done):
    python scripts/02_register_models.py --checkpoint "..\\a2-deployment\\models\\resnet18_finetuned.pt"

Re-running skips any model version that is already registered.
Fine-tuning on CPU takes ~8-12 minutes per epoch (default 2 epochs per model).
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import random
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models

warnings.filterwarnings("ignore", message=".*can be interpreted in multiple ways.*")
warnings.filterwarnings("ignore", message=".*integer column.*")
# MLflow warns that pickled models run code when loaded. True in general; here we
# only ever load models we saved ourselves, so the warning is noise.
logging.getLogger("mlflow.pytorch").setLevel(logging.ERROR)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

MODEL_NAME = "xray-pneumonia-classifier"
DATA_EXPERIMENT = "xray-datasets"
MODEL_EXPERIMENT = "xray-models"
SEED = 42
THRESHOLD = 0.5
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
A3_ROOT = Path(__file__).resolve().parent.parent
REPORTS = A3_ROOT / "reports"

# Identical to A1 Step 3 - serving and evaluation must preprocess the same way.
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
eval_transform = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
train_transform = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# A1's training settings, recorded on model version 1.
A1_PARAMS = {
    "architecture": "resnet18", "init_weights": "imagenet1k_v1", "head": "Linear(512,1)",
    "epochs": 5, "optimizer": "adam", "lr": 1e-4, "batch_size": 32,
    "loss": "BCEWithLogitsLoss(pos_weight=neg/pos)", "threshold": THRESHOLD,
    "seed": SEED, "trained_in": "A1 notebook",
}


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
class XrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, transform):
        self.paths = [str(root / r) for r in df.relpath]
        self.labels = df.label.astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.transform(im.convert("L"))
        return x, self.labels[i]


class DatasetVersion:
    """A dataset version as recorded in Step 1: its manifest, image root and MLflow Datasets."""

    def __init__(self, version: str):
        runs = mlflow.search_runs(experiment_names=[DATA_EXPERIMENT],
                                  filter_string=f"tags.dataset_version = '{version}'")
        if runs.empty:
            raise SystemExit(f"Dataset {version} not found in MLflow - run Step 1 first.")
        run = mlflow.get_run(runs.iloc[0].run_id)
        self.version = version
        self.run_id = run.info.run_id
        self.digest = run.data.tags["dataset_digest"]
        self.root = Path(run.data.params["image_root"])
        # The manifest comes from MLflow, not the local disk: the registry is the source of truth.
        local = mlflow.artifacts.download_artifacts(run_id=self.run_id, artifact_path="manifest.csv",
                                                    dst_path=tempfile.mkdtemp())
        self.df = pd.read_csv(local)
        self.inputs = {i.dataset.name.rsplit("-", 1)[-1]: i.dataset for i in run.inputs.dataset_inputs}
        missing = [r for r in self.df.relpath.head(20) if not (self.root / r).exists()]
        if missing:
            raise SystemExit(f"Images for {version} not found under {self.root}")

    def split(self, name: str) -> pd.DataFrame:
        return self.df[self.df.split == name].reset_index(drop=True)

    def log_as_input(self, split: str, context: str):
        """Attach this dataset split to the active run (shows under 'Datasets used')."""
        mlflow.log_input(mlflow.data.from_pandas(
            self.split(split), source=self.root.resolve().as_uri(),
            name=f"chest-xray-{self.version}-{split}", targets="label"), context=context)


def loader(df, root, transform, shuffle, workers):
    return DataLoader(XrayDataset(df, root, transform), batch_size=32, shuffle=shuffle,
                      num_workers=workers, persistent_workers=workers > 0)


# ----------------------------------------------------------------------
# Model + evaluation
# ----------------------------------------------------------------------
def build_model() -> nn.Module:
    m = models.resnet18(weights=None)  # all weights come from the checkpoint
    m.fc = nn.Linear(512, 1)
    return m


@torch.no_grad()
def predict(model, dl, device) -> np.ndarray:
    model.eval()
    probs = []
    for x, _ in dl:
        probs.append(torch.sigmoid(model(x.to(device))).squeeze(1).cpu().numpy())
    return np.concatenate(probs)


def metrics_for(y: np.ndarray, p: np.ndarray, prefix: str) -> tuple[dict, np.ndarray]:
    pred = (p >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "roc_auc": roc_auc_score(y, p),
        "recall_normal": tn / (tn + fp),
        "recall_pneumonia": tp / (tp + fn),
        "pct_predicted_pneumonia": 100 * pred.mean(),
        "mean_prob_pneumonia": float(p.mean()),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    return {f"{prefix}_{k}": round(float(v), 4) for k, v in m.items()}, np.array([[tn, fp], [fn, tp]])


def confusion_png(cms: dict, title: str, path: Path):
    fig, axes = plt.subplots(1, len(cms), figsize=(4.6 * len(cms), 4))
    for ax, (name, cm) in zip(np.atleast_1d(axes), cms.items()):
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=13)
        ax.set_xticks([0, 1], CLASS_NAMES)
        ax.set_yticks([0, 1], CLASS_NAMES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(name)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def evaluate_and_log(model, device, datasets: list[DatasetVersion], workers: int, tag: str) -> dict:
    """Evaluate on each dataset's test split; log metrics, predictions and confusion matrices."""
    all_metrics, cms = {}, {}
    for ds in datasets:
        test = ds.split("test")
        t0 = time.time()
        p = predict(model, loader(test, ds.root, eval_transform, False, workers), device)
        m, cm = metrics_for(test.label.values, p, f"test_{ds.version}")
        all_metrics.update(m)
        cms[f"test set {ds.version}"] = cm
        ds.log_as_input("test", context=f"evaluation_{ds.version}")
        # Per-image predictions: the raw material for drift charts later.
        out = test[["relpath", "label", "mean_intensity", "std_intensity"]].assign(
            prob_pneumonia=p.round(5), predicted=(p >= THRESHOLD).astype(int))
        f = REPORTS / f"predictions_{tag}_test_{ds.version}.csv"
        out.to_csv(f, index=False)
        mlflow.log_artifact(str(f), "predictions")
        print(f"    test {ds.version}: acc={m[f'test_{ds.version}_accuracy']:.3f} "
              f"f1={m[f'test_{ds.version}_f1_macro']:.3f} auc={m[f'test_{ds.version}_roc_auc']:.3f} "
              f"normal_recall={m[f'test_{ds.version}_recall_normal']:.3f} ({time.time() - t0:.0f}s)")
    mlflow.log_metrics(all_metrics)
    f = REPORTS / f"confusion_{tag}.png"
    confusion_png(cms, f"Model {tag}", f)
    mlflow.log_artifact(str(f))
    return all_metrics


def log_and_register(model, run_id: str, role: str, train_version: str, description: str) -> int:
    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, 3, IMG_SIZE, IMG_SIZE), "image")]),
        outputs=Schema([TensorSpec(np.dtype(np.float32), (-1, 1), "logit")]),
    )
    info = mlflow.pytorch.log_model(
        model.cpu(), name="model", registered_model_name=MODEL_NAME,
        signature=signature, serialization_format="pickle",
        pip_requirements=[f"torch=={torch.__version__.split('+')[0]}",
                          f"torchvision=={__import__('torchvision').__version__.split('+')[0]}"],
    )
    version = int(info.registered_model_version)
    client = mlflow.MlflowClient()
    client.update_model_version(MODEL_NAME, str(version), description=description)
    for k, v in {"model_role": role, "train_dataset_version": train_version,
                 "threshold": str(THRESHOLD), "source_run_id": run_id}.items():
        client.set_model_version_tag(MODEL_NAME, str(version), k, v)
    print(f"    registered {MODEL_NAME} version {version} (role: {role})")
    return version


def registered_version_for(role: str) -> int | None:
    """Find the registered version that plays a role (baseline / retrained_for_drift / control).

    Versions registered before roles were tagged on them get the tag copied
    from their source run, so older registries keep working."""
    client = mlflow.MlflowClient()
    try:
        versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
    except mlflow.exceptions.MlflowException:
        return None
    for mv in versions:
        mv_role = mv.tags.get("model_role")
        if mv_role is None:
            mv_role = client.get_run(mv.run_id).data.tags.get("model_role")
            if mv_role:
                client.set_model_version_tag(MODEL_NAME, mv.version, "model_role", mv_role)
        if mv_role == role:
            return int(mv.version)
    return None


# ----------------------------------------------------------------------
# Model version 1 - import the A1 model
# ----------------------------------------------------------------------
def register_v1(checkpoint: Path, d1, d2, device, workers) -> int:
    existing = registered_version_for("baseline")
    if existing:
        print(f"[baseline] already registered as version {existing} - skipping")
        return existing
    print("[baseline] importing A1 checkpoint and evaluating on both test sets ...")
    model = build_model()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device)
    with mlflow.start_run(run_name="resnet18-v1 (A1, trained on data v1)") as run:
        mlflow.set_tags({"model_role": "baseline", "train_dataset_version": "v1",
                         "mlflow.note.content": "A1 ResNet18 imported unchanged and evaluated on "
                                                "test sets v1 and v2."})
        mlflow.log_params({**A1_PARAMS, "train_dataset_version": "v1",
                           "train_dataset_digest": d1.digest, "checkpoint": checkpoint.name})
        d1.log_as_input("train", context="training")
        evaluate_and_log(model, device, [d1, d2], workers, tag="baseline")
        return log_and_register(model, run.info.run_id, "baseline", "v1",
                                "A1 ResNet18 fine-tuned from ImageNet on dataset v1 (5 epochs).")


# ----------------------------------------------------------------------
# Fine-tuning from the baseline - used for both the drift model and the control
# ----------------------------------------------------------------------
FINETUNES = {
    # role: (training dataset version, run name, what it answers)
    "retrained_for_drift": ("v2", "resnet18-v2 (v1 fine-tuned on data v2)",
                            "Baseline fine-tuned on drifted dataset v2."),
    "control": ("v1", "resnet18-control (v1 fine-tuned on data v1)",
                "CONTROL: baseline fine-tuned on the original dataset v1 with the same recipe "
                "as the drift model - isolates the effect of extra training from the effect of the data."),
}


def finetune(role: str, datasets: dict, device, workers, epochs, lr) -> int:
    existing = registered_version_for(role)
    if existing:
        print(f"[{role}] already registered as version {existing} - skipping")
        return existing
    train_version, run_name, note = FINETUNES[role]
    ds = datasets[train_version]

    # Same seed for every fine-tune: the control must differ only in its data.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    base = registered_version_for("baseline")
    print(f"[{role}] loading model version {base} from the registry, fine-tuning on data {train_version} ...")
    model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{base}").to(device)

    train, val = ds.split("train"), ds.split("val")
    pos = train.label.sum()
    pos_weight = torch.tensor([(len(train) - pos) / pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_dl = loader(train, ds.root, train_transform, True, workers)
    val_dl = loader(val, ds.root, eval_transform, False, workers)

    params = {**A1_PARAMS, "init_weights": f"models:/{MODEL_NAME}/{base}",
              "epochs": epochs, "lr": lr, "pos_weight": round(pos_weight.item(), 4),
              "train_dataset_version": train_version, "train_dataset_digest": ds.digest,
              "trained_in": "A3 scripts/02_register_models.py", "device": str(device)}

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"model_role": role, "train_dataset_version": train_version,
                         "parent_model_version": str(base), "mlflow.note.content": note})
        mlflow.log_params(params)
        ds.log_as_input("train", context="training")
        ds.log_as_input("val", context="validation")

        # Epoch 0 = the starting point (the baseline) on this run's validation data.
        best_f1, best_state = -1.0, None
        for epoch in range(0, epochs + 1):
            t0 = time.time()
            train_loss = None
            if epoch > 0:
                model.train()
                total, n = 0.0, 0
                for x, y in train_dl:
                    x, y = x.to(device), y.float().unsqueeze(1).to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(x), y)
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * len(x)
                    n += len(x)
                train_loss = total / n
            p = predict(model, val_dl, device)
            val_f1 = f1_score(val.label.values, (p >= THRESHOLD).astype(int), average="macro")
            val_loss = criterion(torch.logit(torch.tensor(p, device=device).clamp(1e-6, 1 - 1e-6)).unsqueeze(1),
                                 torch.tensor(val.label.values, dtype=torch.float32, device=device).unsqueeze(1)).item()
            step_metrics = {"val_f1_macro": val_f1, "val_loss": val_loss}
            if train_loss is not None:
                step_metrics["train_loss"] = train_loss
            mlflow.log_metrics(step_metrics, step=epoch)
            print(f"    epoch {epoch}/{epochs} ({time.time() - t0:.0f}s) "
                  + " ".join(f"{k}={v:.4f}" for k, v in step_metrics.items()))
            if epoch > 0 and val_f1 > best_f1:
                best_f1, best_state = val_f1, copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        mlflow.log_metric("best_val_f1_macro", best_f1)
        print("    evaluating best epoch on both test sets ...")
        evaluate_and_log(model, device, [datasets["v1"], datasets["v2"]], workers, tag=role)
        return log_and_register(model, run.info.run_id, role, train_version,
                                f"{note} Fine-tuned from version {base} for {epochs} epochs.")


# ----------------------------------------------------------------------
# Comparison + aliases
# ----------------------------------------------------------------------
LABELS = {"baseline": "A1 baseline", "retrained_for_drift": "drift-retrained",
          "control": "control"}
KEYS = ["accuracy", "f1_macro", "roc_auc", "recall_normal", "recall_pneumonia"]


def compare_and_alias(versions: dict, changed: bool):
    """versions: role -> registered version number."""
    client = mlflow.MlflowClient()
    rows = []
    for role, v in versions.items():
        m = client.get_run(client.get_model_version(MODEL_NAME, str(v)).run_id).data.metrics
        for test in ("v1", "v2"):
            rows.append({"model": f"v{v} {LABELS[role]}", "role": role, "model_version": v,
                         "test_set": test, **{k: m[f"test_{test}_{k}"] for k in KEYS}})
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "model_comparison.csv", index=False)
    print("\nModel comparison (test sets):")
    print(table.drop(columns=["role", "model_version"]).to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    # Grouped bar chart: each metric, per model, per test set.
    n = len(versions)
    width = 0.8 / n
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
    for ax, test in zip(axes, ("v1", "v2")):
        sub = table[table.test_set == test]
        x = np.arange(len(KEYS))
        for i, (_, r) in enumerate(sub.iterrows()):
            ax.bar(x + (i - (n - 1) / 2) * width, [r[k] for k in KEYS], width, label=r.model)
        ax.set_xticks(x, KEYS, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Test set {test}" + (" (drifted)" if test == "v2" else " (original)"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n, frameon=False)
    fig.suptitle("Registered model versions on original and drifted test data")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(REPORTS / "model_comparison.png", dpi=110)
    plt.close(fig)

    # Split the drift model's gain into "more training" vs "different data".
    effects = {}
    if {"baseline", "retrained_for_drift", "control"} <= versions.keys():
        f1 = table.set_index(["role", "test_set"]).f1_macro
        print("\nWhere did the drift model's macro-F1 change come from?")
        print(f"{'test set':10s}{'total':>10s}{'extra training':>17s}{'drifted data':>15s}")
        for test in ("v1", "v2"):
            total = f1["retrained_for_drift", test] - f1["baseline", test]
            training = f1["control", test] - f1["baseline", test]
            data = f1["retrained_for_drift", test] - f1["control", test]
            effects |= {f"f1_gain_total_test_{test}": total,
                        f"f1_gain_extra_training_test_{test}": training,
                        f"f1_gain_drifted_data_test_{test}": data}
            print(f"{test:10s}{total:>+10.3f}{training:>+17.3f}{data:>+15.3f}")
        print("  total = drift model - baseline;  extra training = control - baseline;"
              "  drifted data = drift model - control")

    # Aliases name a version's role. The app will load "champion":
    # the best macro F1 on the data the system sees now (the drifted v2 data).
    f1_now = table[table.test_set == "v2"].set_index("model_version").f1_macro
    champion = int(f1_now.idxmax())
    client.set_registered_model_alias(MODEL_NAME, "champion", str(champion))
    for role, alias in (("baseline", "baseline"), ("control", "control")):
        if role in versions:
            client.set_registered_model_alias(MODEL_NAME, alias, str(versions[role]))
    client.update_registered_model(
        MODEL_NAME, description="Chest X-ray NORMAL vs PNEUMONIA classifier (ResNet18). "
                                "Alias 'champion' = best macro F1 on the current (v2) data; "
                                "'baseline' = the A1 model; 'control' = baseline fine-tuned on v1 data.")
    print("\nAliases: champion -> version " + str(champion) + ", "
          + ", ".join(f"{a} -> version {versions[r]}" for r, a in
                      (("baseline", "baseline"), ("control", "control")) if r in versions))

    # Attach the comparison to its own run so it appears in the experiment
    # (only when a model version was added, so re-runs don't duplicate it).
    if not changed and not mlflow.search_runs(filter_string="tags.model_role = 'comparison'").empty:
        return
    with mlflow.start_run(run_name="model-comparison"):
        mlflow.set_tags({"model_role": "comparison",
                         "compared_versions": ",".join(str(v) for v in versions.values())})
        if effects:
            mlflow.log_metrics({k: round(v, 4) for k, v in effects.items()})
        mlflow.log_artifact(str(REPORTS / "model_comparison.csv"))
        mlflow.log_artifact(str(REPORTS / "model_comparison.png"))


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path, help="A1/A2 resnet18_finetuned.pt")
    ap.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2, help="DataLoader worker processes")
    ap.add_argument("--no-control", action="store_true", help="skip training the control model")
    args = ap.parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    REPORTS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(args.tracking_uri)
    exp = mlflow.set_experiment(MODEL_EXPERIMENT)
    mlflow.MlflowClient().set_experiment_tag(exp.experiment_id, "mlflow.experimentKind",
                                             "custom_model_development")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MLflow: {args.tracking_uri}  experiment: {MODEL_EXPERIMENT}  device: {device}")

    datasets = {"v1": DatasetVersion("v1"), "v2": DatasetVersion("v2")}
    print(f"Datasets: v1 digest {datasets['v1'].digest}, v2 digest {datasets['v2'].digest}")

    roles = ["baseline", "retrained_for_drift"] + ([] if args.no_control else ["control"])
    before = {r: registered_version_for(r) for r in roles}

    versions = {"baseline": register_v1(args.checkpoint, datasets["v1"], datasets["v2"],
                                        device, args.workers)}
    for role in roles[1:]:
        versions[role] = finetune(role, datasets, device, args.workers, args.epochs, args.lr)

    compare_and_alias(versions, changed=versions != before)
    print(f"\nDone. Open {args.tracking_uri}/#/models/{MODEL_NAME}")


if __name__ == "__main__":
    main()

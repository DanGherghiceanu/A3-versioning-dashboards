"""Read-only access to what Steps 1 and 2 put in MLflow.

Everything the app shows comes from here: dataset versions, registered model
versions, their metrics and their artifacts. The app never trains or logs to
MLflow - the registry is the single source of truth.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

MODEL_NAME = "xray-pneumonia-classifier"
DATA_EXPERIMENT = "xray-datasets"
MODEL_EXPERIMENT = "xray-models"

ROLE_LABELS = {
    "baseline": "A1 baseline",
    "retrained_for_drift": "drift-retrained",
    "control": "control",
}
ROLE_ORDER = ["baseline", "retrained_for_drift", "control"]

# One fixed colour per model role, on every chart (palette slots 1-3,
# validated colour-blind-safe as a set). Colour follows the model, never its rank.
ROLE_COLORS = {
    "baseline": "#2a78d6",
    "retrained_for_drift": "#eb6834",
    "control": "#1baf7a",
}
DATASET_COLORS = {"v1": "#2a78d6", "v2": "#eb6834"}


@dataclass
class DatasetVersion:
    version: str
    run_id: str
    digest: str
    parent: str
    metrics: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    @property
    def image_root(self) -> Path:
        """Where the images are *inside this container*.

        The path logged in Step 1 is a Windows path on the host. Docker mounts
        each version's folder at DATA_ROOT_<VERSION>; outside Docker we fall
        back to the logged path."""
        env = os.environ.get(f"DATA_ROOT_{self.version.upper()}")
        return Path(env) if env else Path(self.params.get("image_root", "."))


@dataclass
class ModelVersion:
    version: int
    role: str
    train_dataset: str
    run_id: str
    description: str
    aliases: list[str]

    @property
    def label(self) -> str:
        tail = "".join(f" @{a}" for a in sorted(self.aliases))
        return f"v{self.version} · {ROLE_LABELS.get(self.role, self.role)}{tail}"

    @property
    def color(self) -> str:
        return ROLE_COLORS.get(self.role, "#52514e")


def client() -> MlflowClient:
    return MlflowClient()


def _prefixed(row: pd.Series, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in row.items()
            if k.startswith(prefix) and pd.notna(v)}


def dataset_versions() -> list[DatasetVersion]:
    runs = mlflow.search_runs(experiment_names=[DATA_EXPERIMENT],
                              order_by=["attributes.start_time ASC"])
    found: dict[str, DatasetVersion] = {}
    for _, r in runs.iterrows():
        tags = _prefixed(r, "tags.")
        if "dataset_version" not in tags:
            continue
        found[tags["dataset_version"]] = DatasetVersion(
            version=tags["dataset_version"], run_id=r.run_id,
            digest=tags.get("dataset_digest", "?"), parent=tags.get("parent_version", "none"),
            metrics=_prefixed(r, "metrics."), params=_prefixed(r, "params."))
    return [found[k] for k in sorted(found)]


def model_versions() -> list[ModelVersion]:
    # Aliases live on the registered model (alias -> version), not in version search results.
    alias_map: dict[str, list[str]] = {}
    for alias, v in client().get_registered_model(MODEL_NAME).aliases.items():
        alias_map.setdefault(str(v), []).append(alias)
    out = []
    for mv in client().search_model_versions(f"name = '{MODEL_NAME}'"):
        role = mv.tags.get("model_role") or client().get_run(mv.run_id).data.tags.get("model_role", "?")
        out.append(ModelVersion(
            version=int(mv.version), role=role,
            train_dataset=mv.tags.get("train_dataset_version", "?"),
            run_id=mv.run_id, description=mv.description or "",
            aliases=alias_map.get(str(mv.version), [])))
    return sorted(out, key=lambda m: m.version)


def run_metrics(run_id: str) -> dict:
    return client().get_run(run_id).data.metrics


def run_params(run_id: str) -> dict:
    return client().get_run(run_id).data.params


def metric_history(run_id: str, key: str) -> pd.DataFrame:
    hist = client().get_metric_history(run_id, key)
    return pd.DataFrame({"epoch": [m.step for m in hist], key: [m.value for m in hist]}
                        ).sort_values("epoch")


def download(run_id: str, path: str) -> Path:
    return Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path,
                                                    dst_path=tempfile.mkdtemp()))


def manifest(run_id: str) -> pd.DataFrame:
    return pd.read_csv(download(run_id, "manifest.csv"))


def test_predictions(run_id: str, dataset_version: str) -> pd.DataFrame | None:
    """Per-image test predictions saved by Step 2 (predictions/*_test_<version>.csv)."""
    for f in client().list_artifacts(run_id, "predictions"):
        if f.path.endswith(f"_test_{dataset_version}.csv"):
            return pd.read_csv(download(run_id, f.path))
    return None


def comparison_metrics() -> dict:
    runs = mlflow.search_runs(experiment_names=[MODEL_EXPERIMENT],
                              filter_string="tags.model_role = 'comparison'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    return {} if runs.empty else _prefixed(runs.iloc[0], "metrics.")


def calibration(version: int) -> dict | None:
    """Latest threshold-calibration run for a model version (Step 2b), or None."""
    runs = mlflow.search_runs(
        experiment_names=[MODEL_EXPERIMENT],
        filter_string=f"tags.model_role = 'calibration' and tags.calibrated_model_version = '{version}'",
        order_by=["attributes.start_time DESC"], max_results=1)
    if runs.empty:
        return None
    r = runs.iloc[0]
    return {"run_id": r.run_id, "metrics": _prefixed(r, "metrics.")}


def val_predictions(run_id: str, dataset_version: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(download(run_id, f"predictions/val_{dataset_version}.csv"))
    except Exception:  # noqa: BLE001 - missing artifact
        return None

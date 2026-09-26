"""
A3 Step 3 - the MLflow app.

A Streamlit front end over the MLflow registry built in Steps 1 and 2:
  - Dataset version selector   (sidebar)  - every version logged in Step 1
  - Model version selector     (sidebar)  - every registered version + aliases
  - Predict                    (tab 1)    - test image or upload, one or all models
  - Dataset                    (tab 2)    - statistics and drift of the chosen version
  - Model metrics              (tab 3)    - metrics, confusion matrix, threshold what-if,
                                            training curves
  - Compare models             (tab 4)    - all versions side by side, ROC curves,
                                            where the drift model's gain came from

Every prediction is logged to Postgres (monitoring.py) for the Grafana dashboard.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

import inference
import monitoring
import registry as reg

st.set_page_config(page_title="X-ray Model Registry", page_icon="🩻", layout="wide")

MLFLOW_UI = os.environ.get("MLFLOW_UI_URL", "http://localhost:5000")
THRESHOLD = inference.THRESHOLD
METRICS = {"accuracy": "Accuracy", "f1_macro": "Macro F1", "roc_auc": "ROC-AUC",
           "recall_normal": "NORMAL recall", "recall_pneumonia": "PNEUMONIA recall"}
CLASS_COLORS = {"NORMAL": "#4a3aa7", "PNEUMONIA": "#eda100"}
GRID = "rgba(128,128,128,0.18)"


# ----------------------------------------------------------------------
# Cached reads - the registry is read once a minute, models loaded once
# ----------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def datasets():
    return reg.dataset_versions()


@st.cache_data(ttl=60, show_spinner=False)
def models():
    return reg.model_versions()


@st.cache_data(ttl=300, show_spinner=False)
def metrics(run_id):
    return reg.run_metrics(run_id)


@st.cache_data(ttl=300, show_spinner=False)
def params(run_id):
    return reg.run_params(run_id)


@st.cache_data(show_spinner=False)
def manifest(run_id):
    return reg.manifest(run_id)


@st.cache_data(show_spinner=False)
def predictions(run_id, dataset_version):
    return reg.test_predictions(run_id, dataset_version)


@st.cache_data(ttl=300, show_spinner=False)
def history(run_id, key):
    return reg.metric_history(run_id, key)


@st.cache_data(ttl=60, show_spinner=False)
def comparison():
    return reg.comparison_metrics()


@st.cache_data(ttl=60, show_spinner=False)
def calibration(version):
    return reg.calibration(version)


@st.cache_data(show_spinner=False)
def val_predictions(run_id, dataset_version):
    return reg.val_predictions(run_id, dataset_version)


@st.cache_resource(show_spinner="Loading model from the registry ...")
def model(version: int):
    return inference.load_model(version)


def style(fig: go.Figure, title: str | None = None, height: int = 340) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)) if title else None,
        height=height, margin=dict(l=10, r=10, t=45 if title else 10, b=10),
        # Legend under the plot, clear of the title and the x-axis title.
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        barcornerradius=4, bargap=0.25, bargroupgap=0.08,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def psi_status(v: float) -> str:
    return "🟢 stable" if v < 0.1 else ("🟠 moderate shift" if v < 0.25 else "🔴 major shift")


# Thresholds on a log-odds scale: probabilities pile up near 1.0 under drift, and
# on a plain 0-1 axis 0.95, 0.99 and 0.999 would sit on top of each other.
T_GRID = 1 / (1 + np.exp(-np.arange(-6, 12.0001, 0.05)))    # thresholds - same grid as scripts/03
PROB_TICKS = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999]
SLIDER_STEPS = sorted({*np.round(np.arange(0.05, 0.951, 0.05), 2), *np.round(np.arange(0.96, 0.991, 0.01), 2),
                       *np.round(np.arange(0.991, 0.9991, 0.001), 3), 0.9995, 0.9999, 0.99995, 0.99999})


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fmt_t(t: float) -> str:
    return f"{t:.5f}".rstrip("0").rstrip(".")


def logit_axis(fig: go.Figure, title: str):
    fig.update_xaxes(title=title, tickvals=logit(PROB_TICKS), ticktext=[fmt_t(t) for t in PROB_TICKS],
                     range=[logit(0.0005)[()], logit(0.999995)[()]])


def f1_curve(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Macro F1 at every threshold in T_GRID (vectorised)."""
    pred = p[None, :] >= T_GRID[:, None]
    pos, neg = (y == 1)[None, :], (y == 0)[None, :]
    tp = (pred & pos).sum(1); fp = (pred & neg).sum(1)
    fn = (~pred & pos).sum(1); tn = (~pred & neg).sum(1)
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1) + 2 * tn / np.maximum(2 * tn + fn + fp, 1)) / 2


def threshold_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tp = int(((pred == 1) & (y == 1)).sum())
    return {"accuracy": accuracy_score(y, pred), "f1_macro": f1_score(y, pred, average="macro"),
            "recall_normal": tn / max(tn + fp, 1), "recall_pneumonia": tp / max(tp + fn, 1),
            "cm": np.array([[tn, fp], [fn, tp]])}


# ----------------------------------------------------------------------
# Connect
# ----------------------------------------------------------------------
try:
    ds_list, mv_list = datasets(), models()
except Exception as e:  # noqa: BLE001
    st.error(f"Cannot reach MLflow at `{os.environ.get('MLFLOW_TRACKING_URI')}`.\n\n{e}")
    st.stop()
if not ds_list or not mv_list:
    st.warning("MLflow is reachable but empty - run Steps 1 and 2 first.")
    st.stop()

by_ds = {d.version: d for d in ds_list}
by_mv = {m.version: m for m in mv_list}
baseline = next((m for m in mv_list if m.role == "baseline"), mv_list[0])

# ----------------------------------------------------------------------
# Sidebar - the two version selectors
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🩻 X-ray model registry")
    st.caption("Chest X-ray NORMAL vs PNEUMONIA · ResNet18 · MLflow-backed")

    dsv = st.selectbox(
        "Dataset version", list(by_ds), index=len(by_ds) - 1,
        format_func=lambda v: f"{v} — " + ("original A1 data" if by_ds[v].parent == "none"
                                           else f"drifted copy of {by_ds[v].parent}"))
    ds = by_ds[dsv]
    st.caption(f"digest `{ds.digest}` · {int(ds.metrics.get('n_images', 0)):,} images")

    champion = next((m.version for m in mv_list if "champion" in m.aliases), mv_list[-1].version)
    mvv = st.selectbox("Model version", list(by_mv), index=list(by_mv).index(champion),
                       format_func=lambda v: by_mv[v].label)
    mv = by_mv[mvv]
    st.caption(f"Trained on dataset **{mv.train_dataset}** · role `{mv.role}`")
    st.caption(mv.description)

    st.divider()
    st.markdown(f"[Open model v{mv.version} in MLflow]({MLFLOW_UI}/#/models/{reg.MODEL_NAME}/versions/{mv.version})  \n"
                f"[Open dataset {dsv} run in MLflow]({MLFLOW_UI}/#/experiments/"
                f"{reg.client().get_run(ds.run_id).info.experiment_id}/runs/{ds.run_id})")
    problem = monitoring.status()
    if problem:
        st.warning(f"Prediction logging off: {problem}")
    else:
        st.caption(f"📝 Prediction log: {monitoring.count():,} rows in Postgres")
    if st.button("↻ Refresh from MLflow"):
        st.cache_data.clear()
        st.rerun()

t_pred, t_data, t_model, t_cmp = st.tabs(
    ["🔍 Predict", "📊 Dataset", "📈 Model metrics", "⚖️ Compare models"])

# ======================================================================
# Tab 1 - Predict
# ======================================================================
with t_pred:
    st.subheader(f"Predict with model {mv.label}")
    src = st.segmented_control("Image source", ["Test image from dataset", "Upload an X-ray"],
                               default="Test image from dataset", key="src")
    img, ref, true_label, source, log_ds = None, None, None, None, None

    if src == "Upload an X-ray":
        up = st.file_uploader("Chest X-ray (JPEG or PNG)", type=["jpg", "jpeg", "png"])
        if up is not None:
            img, ref, source = Image.open(up), up.name, "upload"
    else:
        test = manifest(ds.run_id)
        test = test[test.split == "test"]
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        cls = c1.segmented_control("Class", ["Any", "NORMAL", "PNEUMONIA"], default="Any", key="cls")
        pool = test if cls in (None, "Any") else test[test.label_name == cls]
        # The same file exists in every dataset version, so switching the dataset
        # version keeps the image and shows how drift changes its prediction.
        if (c2.button("🎲 Another image", width="stretch") or st.session_state.get("pick") not in set(pool.relpath)):
            st.session_state.pick = pool.sample(1).iloc[0].relpath
        row = test[test.relpath == st.session_state.pick].iloc[0]
        path = ds.image_root / row.relpath
        if path.exists():
            img, ref, true_label, source, log_ds = Image.open(path), row.relpath, int(row.label), "dataset", dsv
        else:
            st.error(f"Image not found at `{path}`. Is the dataset folder mounted into the app "
                     f"container? (DATA_ROOT_{dsv.upper()} in docker-compose.yml)")

    if img is not None:
        prob, ms = inference.predict(model(mv.version), img)
        stats = inference.image_stats(img)
        pred = int(prob >= THRESHOLD)

        # Log once per (model, image) - Streamlit reruns this script on every click.
        key = (mv.version, ref, source, log_ds)
        if st.session_state.get("last_logged") != key:
            monitoring.log(source=source, model_version=mv.version, model_role=mv.role,
                           dataset_version=log_ds, image_ref=str(ref)[:300], true_label=true_label,
                           prob_pneumonia=prob, predicted=pred, threshold=THRESHOLD,
                           latency_ms=ms, **stats)
            st.session_state.last_logged = key

        left, right = st.columns([1, 1.25], gap="large")
        left.image(img, caption=str(ref), width="stretch")
        with right:
            st.markdown(f"## {'🫁 PNEUMONIA' if pred else '✅ NORMAL'}")
            gauge = go.Figure(go.Indicator(
                mode="number+gauge", value=prob, number=dict(valueformat=".3f"),
                title=dict(text="P(pneumonia)", font=dict(size=14)),
                gauge=dict(shape="bullet", axis=dict(range=[0, 1]),
                           bar=dict(color=CLASS_COLORS["PNEUMONIA" if pred else "NORMAL"], thickness=0.6),
                           threshold=dict(line=dict(color="#52514e", width=3), thickness=0.9,
                                          value=THRESHOLD))))
            st.plotly_chart(style(gauge, height=110), width="stretch")
            st.caption(f"The grey mark is the decision threshold ({THRESHOLD}).")
            if abs(prob - THRESHOLD) < 0.1:
                st.warning("Borderline: within 0.10 of the threshold.")
            if true_label is not None:
                truth = ["NORMAL", "PNEUMONIA"][true_label]
                (st.success if pred == true_label else st.error)(
                    f"Ground truth: **{truth}** — {'correct' if pred == true_label else 'wrong'}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Inference", f"{ms:.0f} ms")
            v1_avg = by_ds.get("v1", ds).metrics.get("mean_intensity_avg")
            m2.metric("Brightness", f"{stats['mean_intensity']:.1f}",
                      help=f"Mean pixel value. Dataset averages: " + ", ".join(
                          f"{d.version} {d.metrics.get('mean_intensity_avg', 0):.1f}" for d in ds_list))
            m3.metric("Contrast", f"{stats['std_intensity']:.1f}",
                      help=f"Std of pixel values. Dataset averages: " + ", ".join(
                          f"{d.version} {d.metrics.get('std_intensity_avg', 0):.1f}" for d in ds_list))

        if st.toggle("Run every model version on this image"):
            rows = []
            for m in mv_list:
                p, _ = inference.predict(model(m.version), img)
                rows.append((m, p))
            fig = go.Figure()
            for m, p in rows:
                fig.add_bar(x=[p], y=[m.label], orientation="h", marker_color=m.color, name=m.label,
                            text=[f"{p:.3f}"], textposition="outside", showlegend=False,
                            hovertemplate=f"{m.label}<br>P(pneumonia) %{{x:.3f}}<extra></extra>")
            fig.add_vline(x=THRESHOLD, line_dash="dash", line_color="#52514e",
                          annotation_text="threshold", annotation_position="top")
            fig.update_xaxes(range=[0, 1.12], title="P(pneumonia)")
            st.plotly_chart(style(fig, "Every model version on this image", 120 + 45 * len(rows)),
                            width="stretch")
            st.caption("Comparison predictions are not written to the prediction log.")

    st.divider()
    with st.expander("🚦 Simulate traffic — feeds the Grafana dashboard"):
        st.caption("Sends a batch of test images through the selected model and logs every "
                   "prediction, as if a scanner were sending them. Run a batch from v1, then one "
                   "from v2, and the drift panels in Grafana will show the shift.")
        c1, c2, c3 = st.columns([1, 2, 1], vertical_alignment="bottom")
        sim_ds = c1.selectbox("Images from dataset", list(by_ds), index=list(by_ds).index(dsv), key="sim_ds")
        n = c2.slider("Number of images", 10, 300, 60, 10)
        if c3.button("Run batch", type="primary", width="stretch"):
            d = by_ds[sim_ds]
            t = manifest(d.run_id)
            t = t[t.split == "test"]
            n = min(n, len(t))
            t = t.sample(n)
            m = model(mv.version)
            bar = st.progress(0.0)
            ys, ps = [], []
            for i, r in enumerate(t.itertuples()):
                with Image.open(d.image_root / r.relpath) as im:
                    p, ms = inference.predict(m, im)
                    s = inference.image_stats(im)
                monitoring.log(source="simulation", model_version=mv.version, model_role=mv.role,
                               dataset_version=sim_ds, image_ref=r.relpath, true_label=int(r.label),
                               prob_pneumonia=p, predicted=int(p >= THRESHOLD), threshold=THRESHOLD,
                               latency_ms=ms, **s)
                ys.append(int(r.label)); ps.append(p)
                bar.progress((i + 1) / n, text=f"{i + 1}/{n} images")
            ys, pr = np.array(ys), (np.array(ps) >= THRESHOLD).astype(int)
            k1, k2, k3 = st.columns(3)
            k1.metric("Batch accuracy", f"{(ys == pr).mean():.1%}")
            k2.metric("Predicted PNEUMONIA", f"{pr.mean():.1%}", help="No labels needed - a drift signal.")
            k3.metric("Actually PNEUMONIA", f"{ys.mean():.1%}")

# ======================================================================
# Tab 2 - Dataset
# ======================================================================
with t_data:
    st.subheader(f"Dataset {dsv}")
    parent_txt = "original A1 data" if ds.parent == "none" else f"derived from **{ds.parent}**"
    st.caption(f"digest `{ds.digest}` · {parent_txt} · seed {ds.params.get('seed', '?')} · "
               f"MLflow run `{ds.run_id[:8]}`")
    m = ds.metrics
    ref = by_ds.get("v1") if dsv != "v1" else None

    def delta(key):
        if ref is None or key not in ref.metrics:
            return None
        return f"{m[key] - ref.metrics[key]:+.1f} vs v1"

    c = st.columns(5)
    c[0].metric("Images", f"{int(m.get('n_images', 0)):,}")
    c[1].metric("Train / val / test", f"{int(m.get('n_train', 0))} / {int(m.get('n_val', 0))} / {int(m.get('n_test', 0))}")
    c[2].metric("% PNEUMONIA (test)", f"{m.get('pct_pneumonia_test', 0):.1f}%")
    c[3].metric("Brightness (mean)", f"{m.get('mean_intensity_avg', 0):.1f}", delta("mean_intensity_avg"),
                delta_color="off")
    c[4].metric("Contrast (std)", f"{m.get('std_intensity_avg', 0):.1f}", delta("std_intensity_avg"),
                delta_color="off")

    if ds.parent != "none":
        st.markdown(f"#### Drift from {ds.parent}")
        c = st.columns(3)
        for col, (k, label) in zip(c, [("psi_mean_intensity", "PSI · brightness"),
                                       ("psi_std_intensity", "PSI · contrast"),
                                       ("psi_mean_intensity_test", "PSI · brightness (test only)")]):
            if k in m:
                col.metric(label, f"{m[k]:.3f}",
                           help="Population Stability Index: <0.1 stable, 0.1–0.25 moderate, >0.25 major.")
                col.caption(psi_status(m[k]))
        drift = {k.removeprefix("drift_"): v for k, v in ds.params.items() if k.startswith("drift_")}
        if drift:
            st.caption("How it was made: " + " · ".join(f"`{k}` = {v}" for k, v in drift.items()))

    st.markdown("#### Image statistics — all dataset versions")
    c1, c2 = st.columns(2)
    for col, key, title in ((c1, "mean_intensity", "Brightness (mean pixel value)"),
                            (c2, "std_intensity", "Contrast (std of pixel values)")):
        fig = go.Figure()
        for d in ds_list:
            man = manifest(d.run_id)
            fig.add_histogram(x=man[key], name=d.version, nbinsx=60,
                              marker_color=reg.DATASET_COLORS.get(d.version, "#52514e"),
                              opacity=0.8 if d.version == dsv else 0.35,
                              hovertemplate=f"{d.version}<br>%{{x}}: %{{y}} images<extra></extra>")
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title=title)
        fig.update_yaxes(title="images")
        col.plotly_chart(style(fig), width="stretch")

    c1, c2 = st.columns([1, 1.4])
    man = manifest(ds.run_id)
    bal = man.groupby("split").label.agg(["mean", "size"]).reindex(["train", "val", "test"])
    fig = go.Figure(go.Bar(x=bal.index, y=100 * bal["mean"], marker_color="#2a78d6",
                           text=[f"{v:.1f}%" for v in 100 * bal["mean"]], textposition="outside",
                           customdata=bal["size"],
                           hovertemplate="%{x}: %{y:.1f}% PNEUMONIA of %{customdata} images<extra></extra>"))
    fig.update_yaxes(range=[0, 100], title="% PNEUMONIA")
    c1.plotly_chart(style(fig, "Class balance per split"), width="stretch")

    with c2:
        st.markdown("**Sample test images** (the same files in every version)")
        picks = pd.concat([man[(man.split == "test") & (man.label == k)].head(2) for k in (0, 1)])
        cols = st.columns(len(picks))
        for col, r in zip(cols, picks.itertuples()):
            p = ds.image_root / r.relpath
            if p.exists():
                col.image(str(p), caption=r.label_name, width="stretch")

# ======================================================================
# Tab 3 - Model metrics
# ======================================================================
with t_model:
    st.subheader(f"Model {mv.label} on test set {dsv}")
    same = mv.train_dataset == dsv
    st.caption(f"Trained on dataset **{mv.train_dataset}**; evaluated on the test split of dataset "
               f"**{dsv}** ({'same data distribution as training' if same else 'a different distribution from training'}). "
               f"Deltas compare with the {reg.ROLE_LABELS['baseline']} (v{baseline.version}).")
    met, bmet = metrics(mv.run_id), metrics(baseline.run_id)
    c = st.columns(5)
    for col, (k, label) in zip(c, METRICS.items()):
        v = met.get(f"test_{dsv}_{k}")
        b = bmet.get(f"test_{dsv}_{k}")
        col.metric(label, "—" if v is None else f"{v:.3f}",
                   None if (v is None or b is None or mv.version == baseline.version) else f"{v - b:+.3f}")

    p = predictions(mv.run_id, dsv)
    cal = calibration(mv.version)
    cm_ = cal["metrics"] if cal else {}
    t_star = cm_.get(f"{dsv}_threshold")
    pv = val_predictions(cal["run_id"], dsv) if cal else None

    st.markdown("#### Choosing the decision threshold")
    if p is None:
        st.info("No per-image predictions stored for this model and test set.")
    else:
        y, prob = p.label.values, p.prob_pneumonia.values
        at05 = threshold_metrics(y, prob, THRESHOLD)
        if t_star is None:
            st.info("No validation-chosen threshold for this model yet — run "
                    "`python scripts/03_calibrate_thresholds.py`.")
        else:
            st.caption("The honest procedure: pick the threshold on the **validation** split, then score the "
                       "untouched **test** split at that threshold. Choosing it by looking at test results "
                       "would make the score optimistic.")
            f1_star = cm_.get(f"test_{dsv}_f1_macro_at_threshold")
            c = st.columns(4)
            c[0].metric("Threshold chosen on validation", fmt_t(t_star),
                        help=f"Max macro F1 on the {dsv} validation split "
                             f"({cm_.get(f'val_{dsv}_f1_macro_at_threshold', 0):.3f}).")
            c[1].metric("Test macro F1 @ 0.50", f"{at05['f1_macro']:.3f}")
            c[2].metric("Test macro F1 @ chosen threshold", f"{f1_star:.3f}",
                        f"{f1_star - at05['f1_macro']:+.3f} vs 0.50")
            c[3].metric("Best possible on test", f"{cm_.get(f'test_{dsv}_oracle_f1_macro', 0):.3f}",
                        help=f"At {fmt_t(cm_.get(f'test_{dsv}_oracle_threshold', 0))}, found by peeking at the "
                             "test set. Shown only to measure how optimistic peeking is — not a fair result.")

        # F1 vs threshold - validation (where the choice is made) and test (the report)
        fig = go.Figure()
        if pv is not None:
            fig.add_scatter(x=logit(T_GRID), y=f1_curve(pv.label.values, pv.prob_pneumonia.values),
                            name=f"validation {dsv} (used to choose)", mode="lines",
                            line=dict(color=mv.color, width=2, dash="dot"), customdata=T_GRID,
                            hovertemplate="threshold %{customdata:.5f}<br>val F1 %{y:.3f}<extra></extra>")
        fig.add_scatter(x=logit(T_GRID), y=f1_curve(y, prob), name=f"test {dsv} (reported)", mode="lines",
                        line=dict(color=mv.color, width=2), customdata=T_GRID,
                        hovertemplate="threshold %{customdata:.5f}<br>test F1 %{y:.3f}<extra></extra>")
        fig.add_vline(x=logit(THRESHOLD)[()], line_dash="dot", line_color="#a3a29d",
                      annotation_text="default 0.5", annotation_position="bottom right")
        if t_star is not None:
            fig.add_vline(x=logit(t_star)[()], line_dash="dash", line_color="#52514e",
                          annotation_text=f"chosen on validation {fmt_t(t_star)}",
                          annotation_position="top left")
        logit_axis(fig, "decision threshold (log-odds scale)")
        fig.update_yaxes(title="macro F1")
        st.plotly_chart(style(fig, "Macro F1 as the threshold moves", 360), width="stretch")

        # Free exploration
        # The slider starts on the validation-chosen threshold (added to the steps so it lands exactly).
        steps = sorted({*SLIDER_STEPS, *([round(t_star, 5)] if t_star is not None else [])})
        start_at = round(t_star, 5) if t_star is not None else THRESHOLD
        thr = st.select_slider("Try any threshold (test set)", options=steps, value=start_at,
                               format_func=fmt_t)
        at = threshold_metrics(y, prob, thr)
        c = st.columns(4)
        for col, k in zip(c, ["accuracy", "f1_macro", "recall_normal", "recall_pneumonia"]):
            col.metric(f"{METRICS[k]} @ {fmt_t(thr)}", f"{at[k]:.3f}",
                       None if thr == THRESHOLD else f"{at[k] - at05[k]:+.3f} vs 0.50")

        c1, c2 = st.columns([1, 1.6])
        cm = at["cm"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=["pred NORMAL", "pred PNEUMONIA"], y=["NORMAL", "PNEUMONIA"],
            colorscale=[[0, "#cde2fb"], [1, "#104281"]], showscale=False,
            text=cm, texttemplate="%{text}", textfont=dict(size=18),
            hovertemplate="actual %{y}<br>%{x}: %{z}<extra></extra>", xgap=2, ygap=2))
        fig.update_yaxes(autorange="reversed", title="actual")
        c1.plotly_chart(style(fig, f"Confusion matrix @ {fmt_t(thr)}"), width="stretch")

        # Probability histogram on the same log-odds axis: the 0.95-0.999 region is readable.
        edges = np.arange(-14, 14.01, 0.5)
        lo, hi = 1 / (1 + np.exp(-edges[:-1])), 1 / (1 + np.exp(-edges[1:]))
        fig = go.Figure()
        for lab, name in ((0, "NORMAL"), (1, "PNEUMONIA")):
            counts, _ = np.histogram(logit(prob[y == lab]), edges)
            fig.add_bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=0.5, name=f"actual {name}",
                        marker_color=CLASS_COLORS[name], opacity=0.75, customdata=np.c_[lo, hi],
                        hovertemplate=f"actual {name}<br>P %{{customdata[0]:.5f}}–%{{customdata[1]:.5f}}"
                                      f"<br>%{{y}} images<extra></extra>")
        fig.add_vline(x=logit(thr)[()], line_dash="dash", line_color="#52514e",
                      annotation_text=f"threshold {fmt_t(thr)}", annotation_position="top left")
        fig.update_layout(barmode="overlay", bargap=0)
        logit_axis(fig, "P(pneumonia) — log-odds scale")
        fig.update_yaxes(title="images")
        c2.plotly_chart(style(fig, "Predicted probability by actual class"), width="stretch")

    st.markdown("#### Training curves")
    f1h, vl, tl = (history(mv.run_id, k) for k in ("val_f1_macro", "val_loss", "train_loss"))
    if f1h.empty:
        st.info("This version was imported from A1 — its training happened in the A1 notebook, "
                "so there are no per-epoch curves in MLflow.")
    else:
        c1, c2 = st.columns(2)
        fig = go.Figure(go.Scatter(x=f1h.epoch, y=f1h.val_f1_macro, mode="lines+markers",
                                   line=dict(color=mv.color, width=2), marker=dict(size=9),
                                   hovertemplate="epoch %{x}<br>val F1 %{y:.4f}<extra></extra>"))
        fig.update_xaxes(title="epoch (0 = starting model)", dtick=1)
        c1.plotly_chart(style(fig, "Validation macro F1"), width="stretch")
        fig = go.Figure()
        fig.add_scatter(x=vl.epoch, y=vl.val_loss, name="validation", mode="lines+markers",
                        line=dict(color=mv.color, width=2), marker=dict(size=9))
        if not tl.empty:
            fig.add_scatter(x=tl.epoch, y=tl.train_loss, name="training", mode="lines+markers",
                            line=dict(color=mv.color, width=2, dash="dot"), marker=dict(size=9, symbol="diamond"))
        fig.update_xaxes(title="epoch", dtick=1)
        fig.update_yaxes(title="loss")
        c2.plotly_chart(style(fig, "Loss"), width="stretch")

    with st.expander("Training parameters (lineage)"):
        st.dataframe(pd.DataFrame(sorted(params(mv.run_id).items()), columns=["parameter", "value"]),
                     hide_index=True)

# ======================================================================
# Tab 4 - Compare models
# ======================================================================
with t_cmp:
    st.subheader("Compare model versions")
    metric = st.segmented_control("Metric", list(METRICS), format_func=METRICS.get,
                                  default="f1_macro", key="cmp_metric") or "f1_macro"

    tests = [d.version for d in ds_list]
    tlabel = {t: f"test {t} " + ("(original)" if by_ds[t].parent == "none" else "(drifted)") for t in tests}
    fig = go.Figure()
    for m in mv_list:
        mm = metrics(m.run_id)
        vals = [mm.get(f"test_{t}_{metric}") for t in tests]
        fig.add_bar(x=[tlabel[t] for t in tests], y=vals, name=m.label, marker_color=m.color,
                    text=[f"{v:.3f}" if v is not None else "" for v in vals], textposition="outside",
                    hovertemplate=f"{m.label}<br>%{{x}}: %{{y:.3f}}<extra></extra>")
    fig.update_yaxes(range=[0, 1.08], title=METRICS[metric])
    st.plotly_chart(style(fig, f"{METRICS[metric]} by model version and test set", 380), width="stretch")

    rows = []
    for m in mv_list:
        mm = metrics(m.run_id)
        cal_m = (calibration(m.version) or {}).get("metrics", {})
        for t in tests:
            rows.append({"model": m.label, "trained on": m.train_dataset, "test set": t,
                         **{METRICS[k]: mm.get(f"test_{t}_{k}") for k in METRICS},
                         "val threshold": cal_m.get(f"{t}_threshold"),
                         "F1 @ val threshold": cal_m.get(f"test_{t}_f1_macro_at_threshold")})
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 column_config={**{METRICS[k]: st.column_config.NumberColumn(format="%.3f") for k in METRICS},
                                "val threshold": st.column_config.NumberColumn(format="%.4f"),
                                "F1 @ val threshold": st.column_config.NumberColumn(format="%.3f")})
    st.caption("Metrics use the default threshold 0.50, except the last two columns: the threshold chosen on "
               "the validation split (scripts/03) and the test macro F1 it gives.")

    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#a3a29d", dash="dot", width=1),
                    showlegend=False, hoverinfo="skip")
    for m in mv_list:
        p = predictions(m.run_id, dsv)
        if p is None:
            continue
        fpr, tpr, _ = roc_curve(p.label, p.prob_pneumonia)
        auc = roc_auc_score(p.label, p.prob_pneumonia)
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{m.label} (AUC {auc:.3f})",
                        line=dict(color=m.color, width=2),
                        hovertemplate=f"{m.label}<br>FPR %{{x:.3f}} · TPR %{{y:.3f}}<extra></extra>")
    fig.update_xaxes(title="false positive rate (healthy flagged)", range=[0, 1])
    fig.update_yaxes(title="true positive rate (pneumonia caught)", range=[0, 1.02])
    c1.plotly_chart(style(fig, f"ROC curves on test {dsv}", 420), width="stretch")

    with c2:
        cmpm = comparison()
        roles = {m.role: m for m in mv_list}
        if {"baseline", "retrained_for_drift", "control"} <= roles.keys() and f"f1_gain_total_test_{dsv}" in cmpm:
            base_f1 = metrics(roles["baseline"].run_id)[f"test_{dsv}_f1_macro"]
            tr = cmpm[f"f1_gain_extra_training_test_{dsv}"]
            da = cmpm[f"f1_gain_drifted_data_test_{dsv}"]
            fig = go.Figure(go.Waterfall(
                measure=["absolute", "relative", "relative", "total"],
                x=["A1 baseline", "extra training", "drifted data", "drift-retrained"],
                y=[base_f1, tr, da, 0], text=[f"{base_f1:.3f}", f"{tr:+.3f}", f"{da:+.3f}",
                                              f"{base_f1 + tr + da:.3f}"],
                textposition="outside",
                increasing=dict(marker=dict(color="#2a78d6")),
                decreasing=dict(marker=dict(color="#e34948")),
                totals=dict(marker=dict(color="#52514e")),
                connector=dict(line=dict(color="#a3a29d", width=1))))
            lo = min(base_f1, base_f1 + tr, base_f1 + tr + da)
            fig.update_yaxes(range=[max(0, lo - 0.1), 1.0], title="macro F1")
            st.plotly_chart(style(fig, f"Where the drift model's F1 on test {dsv} came from", 420),
                            width="stretch")
            st.caption("**extra training** = control − baseline (same recipe, original data); "
                       "**drifted data** = drift-retrained − control (same recipe, only the data differs). "
                       "Blue = gain, red = loss.")
        else:
            st.info("The F1 breakdown needs the baseline, drift-retrained and control versions "
                    "(Step 2 with the control model).")

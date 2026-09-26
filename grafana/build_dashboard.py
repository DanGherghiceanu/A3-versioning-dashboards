"""Builds grafana/dashboards/xray-monitoring.json (run: python grafana/build_dashboard.py grafana/dashboards/xray-monitoring.json). Every panel's SQL lives in QUERIES
so it can be tested against Postgres before Grafana ever sees it."""
import json
import sys

DS = {"type": "grafana-postgresql-datasource", "uid": "mlflow-postgres"}
MODEL = "xray-pneumonia-classifier"

# Colours (validated palette): model roles follow the app; dataset v1 blue / v2 orange.
BLUE, ORANGE, AQUA, GREY, INK, RED = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29d", "#52514e", "#e34948"
MODEL_COLORS = {"model v1": BLUE, "model v2": ORANGE, "model v3": AQUA}

# ---------------------------------------------------------------- reusable SQL
DATASETS = f"""ds AS (
  SELECT DISTINCT ON (t.value) t.value AS version, r.run_uuid
  FROM runs r
  JOIN experiments e ON e.experiment_id = r.experiment_id AND e.name = 'xray-datasets'
  JOIN tags t ON t.run_uuid = r.run_uuid AND t.key = 'dataset_version'
  WHERE r.lifecycle_stage = 'active'
  ORDER BY t.value, r.start_time DESC)"""

VERSIONS = f"""mv AS (
  SELECT v.version, v.run_id,
         max(CASE WHEN g.key = 'model_role' THEN g.value END) AS role,
         max(CASE WHEN g.key = 'train_dataset_version' THEN g.value END) AS trained_on,
         max(CASE WHEN g.key = 'recommended_threshold_v1' THEN g.value END)::float AS thr_v1,
         max(CASE WHEN g.key = 'recommended_threshold_v2' THEN g.value END)::float AS thr_v2
  FROM model_versions v
  LEFT JOIN model_version_tags g ON g.name = v.name AND g.version = v.version
  WHERE v.name = '{MODEL}'
  GROUP BY v.version, v.run_id),
label AS (
  SELECT version, run_id, role, trained_on, thr_v1, thr_v2,
         'v' || version || ' ' || CASE role WHEN 'baseline' THEN 'A1 baseline'
                                            WHEN 'retrained_for_drift' THEN 'drift-retrained'
                                            WHEN 'control' THEN 'control' ELSE role END AS model
  FROM mv)"""

CALIB = """cal AS (
  SELECT DISTINCT ON (tv.value) tv.value::int AS version, r.run_uuid
  FROM runs r
  JOIN tags tr ON tr.run_uuid = r.run_uuid AND tr.key = 'model_role' AND tr.value = 'calibration'
  JOIN tags tv ON tv.run_uuid = r.run_uuid AND tv.key = 'calibrated_model_version'
  WHERE r.lifecycle_stage = 'active'
  ORDER BY tv.value, r.start_time DESC)"""


SRC = "('${source}' = 'all' OR source = '${source}')"

# Split simulated traffic into batches: a new batch starts when the model or the
# image dataset changes, or after a pause of more than 60 s (gaps-and-islands).
BATCHES = """s AS (
  SELECT *, CASE WHEN lag(ts) OVER w IS NULL
                   OR lag(model_version) OVER w IS DISTINCT FROM model_version
                   OR lag(dataset_version) OVER w IS DISTINCT FROM dataset_version
                   OR ts - lag(ts) OVER w > interval '60 seconds' THEN 1 ELSE 0 END AS new_batch
  FROM app_predictions WHERE $__timeFilter(ts) AND source = 'simulation'
  WINDOW w AS (ORDER BY ts)),
b AS (SELECT *, sum(new_batch) OVER (ORDER BY ts) AS batch FROM s)"""


def dsm(version, key):
    return (f"(SELECT m.value FROM ds JOIN latest_metrics m ON m.run_uuid = ds.run_uuid "
            f"WHERE ds.version = '{version}' AND m.key = '{key}')")


QUERIES = {
    # ---- 1. data
    "dataset_table": f"""WITH {DATASETS}
SELECT ds.version AS "Version",
       (SELECT value FROM tags WHERE run_uuid = ds.run_uuid AND key = 'dataset_digest') AS "Digest",
       (SELECT value FROM tags WHERE run_uuid = ds.run_uuid AND key = 'parent_version') AS "Derived from",
       max(CASE WHEN m.key = 'n_images' THEN m.value END) AS "Images",
       max(CASE WHEN m.key = 'n_train' THEN m.value END) AS "Train",
       max(CASE WHEN m.key = 'n_val' THEN m.value END) AS "Val",
       max(CASE WHEN m.key = 'n_test' THEN m.value END) AS "Test",
       max(CASE WHEN m.key = 'pct_pneumonia_test' THEN m.value END) AS "% PNEUMONIA (test)",
       coalesce((SELECT 'contrast ×' || max(CASE WHEN key = 'drift_contrast_factor' THEN value END)
                  || ', brightness ×' || max(CASE WHEN key = 'drift_brightness_factor' THEN value END)
                  FROM params WHERE run_uuid = ds.run_uuid), '—') AS "Drift applied"
FROM ds JOIN latest_metrics m ON m.run_uuid = ds.run_uuid
GROUP BY ds.version, ds.run_uuid ORDER BY ds.version""",

    "dataset_pixels": f"""WITH {DATASETS}
SELECT 'Brightness (mean pixel)' AS "measure",
       {dsm('v1', 'mean_intensity_avg')} AS "dataset v1", {dsm('v2', 'mean_intensity_avg')} AS "dataset v2"
UNION ALL
SELECT 'Contrast (std of pixels)',
       {dsm('v1', 'std_intensity_avg')}, {dsm('v2', 'std_intensity_avg')}""",

    "psi_brightness": f"WITH {DATASETS}\nSELECT {dsm('v2', 'psi_mean_intensity')} AS \"PSI brightness\"",
    "psi_contrast": f"WITH {DATASETS}\nSELECT {dsm('v2', 'psi_std_intensity')} AS \"PSI contrast\"",

    # ---- 2. models
    "registry_table": f"""WITH {VERSIONS}, {CALIB}
SELECT l.model AS "Model version",
       coalesce((SELECT string_agg('@' || alias, ' ' ORDER BY alias) FROM registered_model_aliases a
                 WHERE a.name = '{MODEL}' AND a.version = l.version), '') AS "Aliases",
       l.trained_on AS "Trained on",
       max(CASE WHEN m.key = 'test_v1_f1_macro' THEN m.value END) AS "F1 test v1",
       max(CASE WHEN m.key = 'test_v2_f1_macro' THEN m.value END) AS "F1 test v2",
       max(CASE WHEN m.key = 'test_v2_recall_normal' THEN m.value END) AS "NORMAL recall v2",
       l.thr_v2 AS "Threshold v2 (val)",
       (SELECT cm.value FROM cal JOIN latest_metrics cm ON cm.run_uuid = cal.run_uuid
        WHERE cal.version = l.version AND cm.key = 'test_v2_f1_macro_at_threshold') AS "F1 v2 @ val threshold"
FROM label l JOIN latest_metrics m ON m.run_uuid = l.run_id
GROUP BY l.version, l.model, l.trained_on, l.thr_v2 ORDER BY l.version""",

    "model_metric": f"""WITH {VERSIONS}
SELECT l.model AS "model",
       max(CASE WHEN m.key = 'test_v1_$metric' THEN m.value END) AS "test v1 (original)",
       max(CASE WHEN m.key = 'test_v2_$metric' THEN m.value END) AS "test v2 (drifted)"
FROM label l JOIN latest_metrics m ON m.run_uuid = l.run_id
GROUP BY l.version, l.model ORDER BY l.version""",

    "threshold_fairness": f"""WITH {VERSIONS}, {CALIB}
SELECT l.model AS "model",
       max(CASE WHEN m.key = 'test_v2_f1_macro' THEN m.value END) AS "default 0.50",
       max(CASE WHEN c.key = 'test_v2_f1_macro_at_threshold' THEN c.value END) AS "chosen on validation (honest)",
       max(CASE WHEN c.key = 'test_v2_oracle_f1_macro' THEN c.value END) AS "best on test (peeking)"
FROM label l
JOIN latest_metrics m ON m.run_uuid = l.run_id
LEFT JOIN cal ON cal.version = l.version
LEFT JOIN latest_metrics c ON c.run_uuid = cal.run_uuid
GROUP BY l.version, l.model ORDER BY l.version""",

    "training_curves": f"""WITH {VERSIONS}
SELECT 'epoch ' || x.step AS "epoch",
       max(CASE WHEN l.role = 'retrained_for_drift' THEN x.value END) AS "drift-retrained (data v2)",
       max(CASE WHEN l.role = 'control' THEN x.value END) AS "control (data v1)"
FROM label l JOIN metrics x ON x.run_uuid = l.run_id AND x.key = 'val_f1_macro'
WHERE l.role IN ('retrained_for_drift', 'control')
GROUP BY x.step ORDER BY x.step""",

    # ---- 3. live traffic. SRC is the "Traffic" selector: simulated batches only (default) or all.
    "kpi_count": f"SELECT count(*) AS \"Predictions\" FROM app_predictions WHERE $__timeFilter(ts) AND {SRC}",
    "kpi_pred_share": f"""SELECT (100 * avg(predicted))::float AS "Predicted PNEUMONIA"
FROM (SELECT predicted FROM app_predictions WHERE {SRC} ORDER BY ts DESC LIMIT 100) last100""",
    "kpi_true_share": f"""SELECT (100 * avg(true_label))::float AS "Actually PNEUMONIA"
FROM (SELECT true_label FROM app_predictions WHERE true_label IS NOT NULL AND {SRC}
      ORDER BY ts DESC LIMIT 100) last100""",
    "kpi_brightness_shift": f"""WITH {DATASETS}
SELECT abs(avg(p.mean_intensity) - {dsm('v1', 'mean_intensity_avg')}) AS "Brightness shift"
FROM (SELECT mean_intensity FROM app_predictions WHERE {SRC} ORDER BY ts DESC LIMIT 100) p""",
    "kpi_accuracy": f"""SELECT (100 * avg((predicted = true_label)::int))::float AS "Accuracy"
FROM (SELECT predicted, true_label FROM app_predictions WHERE true_label IS NOT NULL AND {SRC}
      ORDER BY ts DESC LIMIT 100) last100""",

    # One row per simulated batch, in order - the story regardless of timing.
    "batch_summary": f"""WITH {BATCHES}
SELECT '#' || lpad(batch::text, 2, '0') || '  model v' || min(model_version) || ' · '
         || min(dataset_version) || ' images' AS "batch",
       (100 * avg(predicted))::float AS "predicted PNEUMONIA %",
       (100 * avg(true_label))::float AS "actually PNEUMONIA %",
       (100 * avg((predicted = true_label)::int))::float AS "accuracy %"
FROM b GROUP BY b.batch ORDER BY min(ts)""",
    "batch_brightness": f"""WITH {BATCHES}
SELECT '#' || lpad(batch::text, 2, '0') || '  model v' || min(model_version) || ' · '
         || min(dataset_version) || ' images' AS "batch",
       avg(mean_intensity) FILTER (WHERE dataset_version = 'v1') AS "images from v1",
       avg(mean_intensity) FILTER (WHERE dataset_version = 'v2') AS "images from v2"
FROM b GROUP BY b.batch ORDER BY min(ts)""",
    "batch_table": f"""WITH {BATCHES}
SELECT batch AS "#", min(ts) AS "Started", 'v' || min(model_version) AS "Model",
       min(dataset_version) AS "Images from", count(*) AS "Images",
       (100 * avg(predicted))::float AS "Predicted PNEUMONIA %",
       (100 * avg(true_label))::float AS "Actually PNEUMONIA %",
       (100 * avg((predicted = true_label)::int))::float AS "Accuracy %",
       avg(mean_intensity) AS "Brightness", avg(std_intensity) AS "Contrast"
FROM b GROUP BY batch ORDER BY min(ts)""",

    "ts_pred_share": f"""SELECT $__timeGroupAlias(ts, '1m'),
       (100 * avg(predicted) FILTER (WHERE model_version = 1))::float AS "model v1",
       (100 * avg(predicted) FILTER (WHERE model_version = 2))::float AS "model v2",
       (100 * avg(predicted) FILTER (WHERE model_version = 3))::float AS "model v3"
FROM app_predictions WHERE $__timeFilter(ts) AND {SRC}
GROUP BY 1 ORDER BY 1""",
    "ts_true_share": f"""SELECT $__timeGroupAlias(ts, '1m'), (100 * avg(true_label))::float AS "actually PNEUMONIA (labels)"
FROM app_predictions WHERE $__timeFilter(ts) AND true_label IS NOT NULL AND {SRC}
GROUP BY 1 ORDER BY 1""",

    "ts_brightness": f"""SELECT $__timeGroupAlias(ts, '1m'), avg(mean_intensity) AS "incoming images"
FROM app_predictions WHERE $__timeFilter(ts) AND {SRC}
GROUP BY 1 ORDER BY 1""",
    "ts_contrast": f"""SELECT $__timeGroupAlias(ts, '1m'), avg(std_intensity) AS "incoming images"
FROM app_predictions WHERE $__timeFilter(ts) AND {SRC}
GROUP BY 1 ORDER BY 1""",
    # Reference lines drawn across the whole time range, not only where traffic exists.
    "ref_brightness": f"""WITH {DATASETS}
SELECT g.t AS "time", {dsm('v1', 'mean_intensity_avg')} AS "dataset v1 average",
       {dsm('v2', 'mean_intensity_avg')} AS "dataset v2 average"
FROM generate_series($__timeFrom()::timestamptz, $__timeTo()::timestamptz, interval '1 minute') AS g(t)
ORDER BY 1""",
    "ref_contrast": f"""WITH {DATASETS}
SELECT g.t AS "time", {dsm('v1', 'std_intensity_avg')} AS "dataset v1 average",
       {dsm('v2', 'std_intensity_avg')} AS "dataset v2 average"
FROM generate_series($__timeFrom()::timestamptz, $__timeTo()::timestamptz, interval '1 minute') AS g(t)
ORDER BY 1""",

    "ts_accuracy": f"""SELECT $__timeGroupAlias(ts, '1m'),
       (100 * avg((predicted = true_label)::int) FILTER (WHERE model_version = 1))::float AS "model v1",
       (100 * avg((predicted = true_label)::int) FILTER (WHERE model_version = 2))::float AS "model v2",
       (100 * avg((predicted = true_label)::int) FILTER (WHERE model_version = 3))::float AS "model v3"
FROM app_predictions WHERE $__timeFilter(ts) AND true_label IS NOT NULL AND {SRC}
GROUP BY 1 ORDER BY 1""",

    "log_table": """SELECT ts AS "Time", source AS "Source", 'v' || model_version AS "Model",
       dataset_version AS "Images from", image_ref AS "Image",
       CASE true_label WHEN 1 THEN 'PNEUMONIA' WHEN 0 THEN 'NORMAL' END AS "Truth",
       CASE predicted WHEN 1 THEN 'PNEUMONIA' ELSE 'NORMAL' END AS "Predicted",
       prob_pneumonia AS "P(pneumonia)", mean_intensity AS "Brightness", std_intensity AS "Contrast",
       latency_ms AS "Latency (ms)"
FROM app_predictions WHERE $__timeFilter(ts) ORDER BY ts DESC LIMIT 200""",

    # ---- annotations: one marker where each simulated batch starts
    "annotation_batches": f"""WITH {BATCHES}
SELECT min(ts) AS time,
       'Batch #' || batch || ': model v' || min(model_version) || ' on ' || min(dataset_version) || ' images'
         AS text,
       'batch' AS tags
FROM b GROUP BY batch ORDER BY batch""",
}

# ---------------------------------------------------------------- panel builders
_id = [0]


def nid():
    _id[0] += 1
    return _id[0]


def target(key, fmt="table", ref="A"):
    return {"refId": ref, "datasource": DS, "editorMode": "code", "rawQuery": True,
            "format": fmt, "rawSql": QUERIES[key]}


def row(title, y):
    return {"type": "row", "id": nid(), "title": title, "collapsed": False,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "panels": []}


def text(content, pos, title=""):
    return {"type": "text", "id": nid(), "title": title, "gridPos": pos,
            "options": {"mode": "markdown", "content": content}}


def color_overrides(mapping, dashed=(), continuous=("dataset v1 average", "dataset v2 average")):
    out = []
    for name, color in mapping.items():
        props = [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}}]
        if name in dashed:
            props += [{"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [8, 6]}},
                      {"id": "custom.showPoints", "value": "never"},
                      {"id": "custom.lineWidth", "value": 2}]
        if name in continuous:
            props += [{"id": "custom.insertNulls", "value": False}]
        out.append({"matcher": {"id": "byName", "options": name}, "properties": props})
    return out


def stat(title, key, pos, unit="none", decimals=2, steps=None, desc="", color_mode="background"):
    steps = steps or [{"color": BLUE, "value": None}]
    return {"type": "stat", "id": nid(), "title": title, "description": desc, "gridPos": pos,
            "datasource": DS, "targets": [target(key)],
            "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals,
                                         "thresholds": {"mode": "absolute", "steps": steps},
                                         "color": {"mode": "thresholds"}}, "overrides": []},
            "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                        "colorMode": color_mode, "graphMode": "none", "textMode": "value",
                        "justifyMode": "center", "orientation": "auto"}}


def barchart(title, key, pos, colors, unit="none", decimals=3, desc="", ymin=None, ymax=None):
    defaults = {"unit": unit, "decimals": decimals, "min": ymin, "max": ymax,
                "color": {"mode": "palette-classic"},
                "custom": {"fillOpacity": 90, "lineWidth": 0, "gradientMode": "none",
                           "axisSoftMin": 0}}
    return {"type": "barchart", "id": nid(), "title": title, "description": desc, "gridPos": pos,
            "datasource": DS, "targets": [target(key)],
            "fieldConfig": {"defaults": {k: v for k, v in defaults.items() if v is not None},
                            "overrides": color_overrides(colors)},
            "options": {"orientation": "vertical", "groupWidth": 0.75, "barWidth": 0.95,
                        "showValue": "always", "stacking": "none", "xTickLabelRotation": 0,
                        "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"},
                        "tooltip": {"mode": "multi", "sort": "none"}}}


def timeseries(title, targets, pos, colors, unit="none", decimals=1, desc="", dashed=(),
               ymin=None, ymax=None):
    defaults = {"unit": unit, "decimals": decimals, "color": {"mode": "palette-classic"},
                "custom": {"drawStyle": "line", "lineWidth": 2, "showPoints": "always",
                           "pointSize": 7, "spanNulls": False, "insertNulls": 90000,
                           "fillOpacity": 0, "axisSoftMin": ymin, "axisSoftMax": ymax}}
    return {"type": "timeseries", "id": nid(), "title": title, "description": desc, "gridPos": pos,
            "datasource": DS, "targets": targets,
            "fieldConfig": {"defaults": defaults, "overrides": color_overrides(colors, dashed)},
            "options": {"legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"},
                        "tooltip": {"mode": "multi", "sort": "none"}}}


def table(title, key, pos, overrides=(), desc=""):
    return {"type": "table", "id": nid(), "title": title, "description": desc, "gridPos": pos,
            "datasource": DS, "targets": [target(key)],
            "fieldConfig": {"defaults": {"custom": {"align": "auto", "cellOptions": {"type": "auto"}}},
                            "overrides": list(overrides)},
            "options": {"showHeader": True, "cellHeight": "sm"}}


def decimals_override(names, d, unit=None):
    props = [{"id": "decimals", "value": d}] + ([{"id": "unit", "value": unit}] if unit else [])
    return {"matcher": {"id": "byRegexp", "options": "^(" + "|".join(names) + ")$"}, "properties": props}


# ---------------------------------------------------------------- layout
panels = []
y = 0
panels.append(text(
    "### Chest X-ray classifier — versioning & drift monitoring\n"
    "A story in three acts, read top to bottom. **① The data:** dataset **v1** is the original A1 data; "
    "**v2** is the same images through a simulated *new scanner* (brighter, lower contrast). "
    "**② The models:** the MLflow registry holds the A1 model (**v1**), a model retrained on drifted data "
    "(**v2**, `@champion`) and a **control** (**v3**) trained the same way on the old data. "
    "**③ Live traffic:** predictions logged by the app. Vertical markers show where each traffic batch "
    "starts — watch image brightness and the PNEUMONIA share jump when drifted images arrive, "
    "with no labels needed, and settle when the retrained model takes over.\n\n"
    "Sources: MLflow's Postgres tables (registry, runs, metrics) and the app's `app_predictions` log.",
    {"h": 6, "w": 24, "x": 0, "y": y}))
y += 6

# ① data
panels.append(row("① The data — what changed between dataset versions", y)); y += 1
panels.append(table("Dataset versions (MLflow dataset runs)", "dataset_table",
                    {"h": 5, "w": 24, "x": 0, "y": y},
                    [decimals_override(["Images", "Train", "Val", "Test"], 0),
                     decimals_override(["% PNEUMONIA \\(test\\)"], 1)],
                    desc="One row per dataset version logged in Step 1. The digest is a hash of the "
                         "dataset's contents: same digest = provably the same data."))
psi_steps = [{"color": "green", "value": None}, {"color": "orange", "value": 0.1},
             {"color": "red", "value": 0.25}]
psi_desc = "Population Stability Index, v2 vs v1: <0.1 stable, 0.1–0.25 moderate, >0.25 major shift."
y += 5
panels.append(stat("Drift score · brightness (PSI)", "psi_brightness", {"h": 8, "w": 6, "x": 0, "y": y},
                   decimals=2, steps=psi_steps, desc=psi_desc))
panels.append(stat("Drift score · contrast (PSI)", "psi_contrast", {"h": 8, "w": 6, "x": 6, "y": y},
                   decimals=2, steps=psi_steps, desc=psi_desc))
panels.append(barchart("Image statistics by dataset version", "dataset_pixels",
                       {"h": 8, "w": 12, "x": 12, "y": y},
                       {"dataset v1": BLUE, "dataset v2": ORANGE},
                       unit="none", decimals=1,
                       desc="Average brightness and contrast (0–255 pixel scale). The drift made images "
                            "brighter and flatter; everything else (files, labels, split) is identical."))
y += 8

# ② models
panels.append(row("② The models — what the registry knows", y)); y += 1
panels.append(table("Registered model versions", "registry_table", {"h": 6, "w": 24, "x": 0, "y": y},
                    [decimals_override(["F1 test v1", "F1 test v2", "NORMAL recall v2",
                                        "F1 v2 @ val threshold"], 3),
                     decimals_override(["Threshold v2 \\(val\\)"], 4)],
                    desc="From the MLflow model registry: aliases, the dataset each version was trained "
                         "on, test metrics at 0.50, and the threshold chosen on validation (Step 2b)."))
y += 6
panels.append(barchart("$metric_label by model version and test set (threshold 0.50)", "model_metric",
                       {"h": 9, "w": 12, "x": 0, "y": y},
                       {"test v1 (original)": BLUE, "test v2 (drifted)": ORANGE}, ymin=0, ymax=1,
                       desc="Pick the metric with the selector at the top of the dashboard."))
panels.append(barchart("Drifted test set: how the threshold is chosen changes the story",
                       "threshold_fairness", {"h": 9, "w": 12, "x": 12, "y": y},
                       {"default 0.50": GREY, "chosen on validation (honest)": BLUE,
                        "best on test (peeking)": AQUA}, ymin=0, ymax=1,
                       desc="Macro F1 on the drifted test set. 'Peeking' picks the threshold on the test "
                            "set itself — shown only to measure how optimistic that would be."))
y += 9
panels.append(barchart("Fine-tuning curves — validation macro F1 per epoch", "training_curves",
                       {"h": 7, "w": 24, "x": 0, "y": y},
                       {"drift-retrained (data v2)": ORANGE, "control (data v1)": AQUA},
                       ymin=0.9, ymax=1,
                       desc="Epoch 0 is the A1 model before fine-tuning, on each run's own validation "
                            "data. The A1 model was never trained in A3, so it has no curve."))
y += 7

# ③ live traffic
panels.append(row("③ Live traffic — drift appearing and being fixed", y)); y += 1
panels.append(stat("Predictions in time range", "kpi_count", {"h": 4, "w": 4, "x": 0, "y": y},
                   decimals=0, color_mode="none"))
panels.append(stat("Predicted PNEUMONIA · last 100", "kpi_pred_share", {"h": 4, "w": 5, "x": 4, "y": y},
                   unit="percent", decimals=1,
                   steps=[{"color": "green", "value": None}, {"color": "orange", "value": 80},
                          {"color": "red", "value": 83}],
                   desc="Share of the last 100 predictions that said PNEUMONIA. Needs no labels — a "
                        "jump here is an early drift alarm. On undrifted data the A1 model sits near 77% (it over-flags); "
                        "with 100 images, ±4 points is normal noise."))
panels.append(stat("Actually PNEUMONIA · last 100 labelled", "kpi_true_share",
                   {"h": 4, "w": 5, "x": 9, "y": y}, unit="percent", decimals=1, color_mode="none",
                   desc="True share in the same traffic (known only because simulated traffic has labels)."))
panels.append(stat("Brightness shift vs dataset v1 · last 100", "kpi_brightness_shift",
                   {"h": 4, "w": 5, "x": 14, "y": y}, decimals=1,
                   steps=[{"color": "green", "value": None}, {"color": "orange", "value": 5},
                          {"color": "red", "value": 10}],
                   desc="|average brightness of the last 100 images − dataset v1 average|. "
                        "Input drift, visible before any label arrives."))
panels.append(stat("Accuracy · last 100 labelled", "kpi_accuracy", {"h": 4, "w": 5, "x": 19, "y": y},
                   unit="percent", decimals=1,
                   steps=[{"color": "red", "value": None}, {"color": "orange", "value": 80},
                          {"color": "green", "value": 84}]))
y += 4
panels.append(barchart("The story, batch by batch — what the model predicted vs the truth", "batch_summary",
                       {"h": 10, "w": 14, "x": 0, "y": y},
                       {"predicted PNEUMONIA %": ORANGE, "actually PNEUMONIA %": GREY, "accuracy %": BLUE},
                       unit="percent", decimals=0, ymin=0, ymax=100,
                       desc="One group per simulated traffic batch, in the order they ran. When drifted images "
                            "reach the old model, predicted PNEUMONIA rises above the true share and accuracy "
                            "falls; the retrained model brings them back together."))
panels.append(barchart("Batch by batch — incoming image brightness", "batch_brightness",
                       {"h": 10, "w": 10, "x": 14, "y": y},
                       {"images from v1": BLUE, "images from v2": ORANGE},
                       decimals=1, ymin=0,
                       desc="The label-free drift signal: the average brightness of each batch. Bars are "
                            "coloured by where the images really came from; the heights alone reveal it "
                            "(dataset averages: v1 ≈ 123, v2 ≈ 140)."))
panels[-1]["options"]["stacking"] = "normal"   # one bar per batch (the other series is empty)
y += 10
panels.append(table("Batch summary", "batch_table", {"h": 6, "w": 24, "x": 0, "y": y},
                    [decimals_override(["Predicted PNEUMONIA %", "Actually PNEUMONIA %", "Accuracy %"], 1),
                     decimals_override(["Brightness", "Contrast"], 1)],
                    desc="A new batch starts when the model or the image dataset changes, or after a pause "
                         "longer than 60 seconds."))
y += 6
panels.append(timeseries(
    "Predicted PNEUMONIA share per minute, by model version",
    [target("ts_pred_share", "time_series", "A"), target("ts_true_share", "time_series", "B")],
    {"h": 9, "w": 12, "x": 0, "y": y},
    {**MODEL_COLORS, "actually PNEUMONIA (labels)": GREY}, unit="percent",
    dashed=("actually PNEUMONIA (labels)",), ymin=40, ymax=100,
    desc="Solid: what the model predicts. Dashed: the true share. The gap widens when drift arrives."))
panels.append(timeseries(
    "Accuracy per minute (labelled traffic), by model version",
    [target("ts_accuracy", "time_series")], {"h": 9, "w": 12, "x": 12, "y": y},
    MODEL_COLORS, unit="percent", ymin=60, ymax=100,
    desc="Only possible because simulated traffic carries labels — real traffic usually doesn't, "
         "which is why the label-free signals on the left and below matter."))
y += 9
panels.append(timeseries(
    "Incoming image brightness vs dataset references",
    [target("ts_brightness", "time_series", "A"), target("ref_brightness", "time_series", "B")],
    {"h": 8, "w": 12, "x": 0, "y": y},
    {"incoming images": INK, "dataset v1 average": BLUE, "dataset v2 average": ORANGE},
    dashed=("dataset v1 average", "dataset v2 average"),
    desc="Average brightness of the images the app received each minute, against the Step 1 averages."))
panels.append(timeseries(
    "Incoming image contrast vs dataset references",
    [target("ts_contrast", "time_series", "A"), target("ref_contrast", "time_series", "B")],
    {"h": 8, "w": 12, "x": 12, "y": y},
    {"incoming images": INK, "dataset v1 average": BLUE, "dataset v2 average": ORANGE},
    dashed=("dataset v1 average", "dataset v2 average")))
y += 8
panels.append(table("Prediction log (latest 200)", "log_table", {"h": 9, "w": 24, "x": 0, "y": y},
                    [decimals_override(["P\\(pneumonia\\)"], 4),
                     decimals_override(["Brightness", "Contrast", "Latency \\(ms\\)"], 1)]))

dashboard = {
    "uid": "xray-monitoring",
    "title": "X-ray classifier — versions & drift",
    "tags": ["A3", "mlflow", "drift"],
    "timezone": "browser",
    "editable": True,
    "graphTooltip": 1,
    "refresh": "30s",
    "time": {"from": "now-2h", "to": "now"},
    "schemaVersion": 39,
    "version": 2,
    "panels": panels,
    "templating": {"list": [{
        "name": "source", "label": "Traffic", "type": "custom", "hide": 0,
        "query": "Simulated batches only : simulation, All traffic (incl. hand-picked) : all",
        "current": {"selected": True, "text": "Simulated batches only", "value": "simulation"},
        "options": [], "multi": False, "includeAll": False}, {
        "name": "metric", "label": "Metric", "type": "custom", "hide": 0,
        "query": "Macro F1 : f1_macro, Accuracy : accuracy, ROC-AUC : roc_auc, "
                 "NORMAL recall : recall_normal, PNEUMONIA recall : recall_pneumonia",
        "current": {"selected": True, "text": "Macro F1", "value": "f1_macro"},
        "options": [], "multi": False, "includeAll": False}]},
    "annotations": {"list": [
        {"builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"}, "enable": True,
         "hide": True, "iconColor": "rgba(0, 211, 255, 1)", "name": "Annotations & Alerts",
         "type": "dashboard"},
        {"name": "Traffic batches", "enable": True, "iconColor": INK, "datasource": DS,
         "target": {"refId": "Anno", "editorMode": "code", "rawQuery": True, "format": "table",
                    "rawSql": QUERIES["annotation_batches"]}}]},
}
# Batch charts can hold many batches: tilt their labels so they don't collide.
for p in panels:
    if p.get("type") == "barchart" and p["title"].startswith(("The story, batch", "Batch by batch")):
        p["options"]["xTickLabelRotation"] = -30
        p["options"]["xTickLabelMaxLength"] = 28
# The bar chart title uses the variable's display text.
for p in panels:
    if p.get("title", "").startswith("$metric_label"):
        p["title"] = p["title"].replace("$metric_label", "${metric:text}")

if __name__ == "__main__":
    out = sys.argv[1]
    with open(out, "w") as f:
        json.dump(dashboard, f, indent=2)
    print(f"wrote {out}: {len(panels)} panels, {len(QUERIES)} queries")

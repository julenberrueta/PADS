"""Read-only MLflow queries used by the frontend.

Everything here degrades to empty results when MLflow tracking is off, so the
UI keeps working (minus live metrics) even without a tracking server.
"""
from __future__ import annotations

import math
import os
from typing import Any

from pads_app.config import get_settings

# Tag stamped on every run of a job (via PADS_RUN_TAGS) so we can find them back.
JOB_TAG = "pads_job_id"


def _json_float(v: Any) -> float | None:
    """Coerce a metric value to a JSON-safe float.

    MLflow may return NaN/Inf (e.g. a diverging loss or a recall with no
    positives). Starlette's JSONResponse uses ``allow_nan=False``, so those
    must become ``null`` or the whole response 500s.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _client():
    from mlflow.tracking import MlflowClient

    return MlflowClient()


def _experiment_id() -> str | None:
    settings = get_settings()
    exp = _client().get_experiment_by_name(settings.experiment)
    return exp.experiment_id if exp else None


def job_runs(job_id: str) -> list[dict[str, Any]]:
    """All MLflow runs tagged with this job_id, newest first.

    Returns lightweight dicts (not Run objects) so they serialise straight to
    JSON for the frontend.
    """
    settings = get_settings()
    if not settings.mlflow_enabled:
        return []
    exp_id = _experiment_id()
    if exp_id is None:
        return []
    runs = _client().search_runs(
        [exp_id],
        filter_string=f"tags.{JOB_TAG} = '{job_id}'",
        order_by=["attributes.start_time ASC"],
    )
    out: list[dict[str, Any]] = []
    for r in runs:
        out.append(
            {
                "run_id": r.info.run_id,
                "run_name": r.data.tags.get("mlflow.runName", r.info.run_id[:8]),
                "status": r.info.status,
                "step": r.data.tags.get("step") or r.data.tags.get("model", ""),
                "retrain_type": r.data.tags.get("retrain_type", ""),
                "params": dict(r.data.params),
                "metrics": {k: _json_float(v) for k, v in r.data.metrics.items()},
            }
        )
    return out


def delete_job_runs(job_id: str) -> int:
    """Delete every MLflow run tagged with this job_id. Returns how many."""
    if not get_settings().mlflow_enabled:
        return 0
    client = _client()
    deleted = 0
    for run in job_runs(job_id):
        try:
            client.delete_run(run["run_id"])
            deleted += 1
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
    return deleted


def metric_history(run_id: str, key: str) -> list[dict[str, float]]:
    """Per-step history of one metric, e.g. 'mort/val_loss' → [{step, value}]."""
    if not get_settings().mlflow_enabled:
        return []
    hist = _client().get_metric_history(run_id, key)
    return [{"step": m.step, "value": _json_float(m.value)} for m in hist]


def list_artifacts(run_id: str, path: str = "") -> list[dict[str, Any]]:
    """Flat listing of artifacts under `path` for one run."""
    if not get_settings().mlflow_enabled:
        return []
    return [
        {"path": f.path, "is_dir": f.is_dir, "size": f.file_size}
        for f in _client().list_artifacts(run_id, path or None)
    ]


def download_artifact(run_id: str, path: str) -> str:
    """Materialise an artifact locally and return its filesystem path."""
    return _client().download_artifacts(run_id, path)


def _read_predictions(run_id: str, base: str) -> Any:
    """Read a prediction artifact, preferring Parquet over CSV.

    ``base`` is the artifact path without extension (e.g.
    ``"inference/results_inference"``). Parquet is smaller to download and
    faster to parse; CSV is the fallback for runs created before the Parquet
    mirrors existed. Raises if neither is present.
    """
    import pandas as pd

    try:
        return pd.read_parquet(download_artifact(run_id, base + ".parquet"))
    except Exception:  # noqa: BLE001 - no parquet (old run) → fall back to CSV
        return pd.read_csv(download_artifact(run_id, base + ".csv"))


def download_job_zip(job_id: str) -> str:
    """Bundle artifacts from *every* run of a job into one .zip; return its path.

    A job spans several MLflow runs (retrain mortality/discharge, calculate
    metrics, inference, ...), so the models live in different runs than the
    metrics. Each run's artifacts are downloaded into a folder named after the
    run so they don't collide, then the whole tree is zipped.
    """
    import re
    import shutil
    import tempfile

    runs = job_runs(job_id)
    if not runs:
        raise ValueError(f"No MLflow runs found for job {job_id}")

    staging = tempfile.mkdtemp()
    client = _client()
    # Group by retrain_type (original/full/lstm/dense/scratch). Each run's
    # artifacts sit under a distinct subfolder (metrics, inference,
    # mortality_model, discharge_model, dataset), so runs of the same type merge
    # into one folder without colliding.
    for run in runs:
        top = re.sub(r"[^A-Za-z0-9._-]+", "_", run["retrain_type"]) or "other"
        dest = os.path.join(staging, top)
        os.makedirs(dest, exist_ok=True)
        client.download_artifacts(run["run_id"], "", dest)

    base = os.path.join(tempfile.mkdtemp(), f"pads_job_{job_id}")
    return shutil.make_archive(base, "zip", staging)


def _roc_points(
    y_true: Any, y_score: Any, threshold: float | None = None, max_points: int = 250
) -> dict | None:
    """ROC curve (downsampled) + AUC for one model. None if AUC is undefined.

    When ``threshold`` is given, also returns the operating point ``op`` (the
    curve point closest to that decision threshold) so the UI can mark the dot.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve

    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:  # only one class present → ROC/AUC undefined
        return None
    fpr, tpr, ths = roc_curve(y_true, y_score)
    op = None
    if threshold is not None:
        i = int(np.argmin(np.abs(ths - threshold)))
        op = {"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4),
              "threshold": round(float(threshold), 4)}
    if len(fpr) > max_points:
        keep = np.unique(np.linspace(0, len(fpr) - 1, max_points).round().astype(int))
        fpr, tpr = fpr[keep], tpr[keep]
    return {
        "fpr": [round(float(x), 4) for x in fpr],
        "tpr": [round(float(y), 4) for y in tpr],
        "auc": round(auc, 4),
        "op": op,
    }


def job_comparison(job_id: str) -> dict[str, Any]:
    """Per-model ROC curves + mean error for a job, for the comparison view.

    One entry per inference run (original baseline + each retrained type), each
    with mortality and discharge ROC points from its predictions artifact.
    The mean error is read from the already-logged ``inf/<type>/mean_error``.
    """
    models: list[dict[str, Any]] = []
    for run in job_runs(job_id):
        if run.get("step") != "inference":
            continue
        try:
            df = _read_predictions(run["run_id"], "inference/results_inference")
        except Exception:  # noqa: BLE001 - skip a run whose predictions are missing
            continue
        mean_error = next(
            (v for k, v in run["metrics"].items() if k.endswith("/mean_error")), None
        )
        models.append({
            "retrain_type": run["retrain_type"] or "original",
            "mort": _roc_points(df["mortality_gt"], df["mortality_prob"]),
            "disch": _roc_points(df["disch_gt"], df["disch_prob"]),
            "mean_error": mean_error,
        })

    # Baseline first, then the retrained types alphabetically.
    models.sort(key=lambda m: (m["retrain_type"] != "original", m["retrain_type"]))
    return {"models": models}


# Clinical categories in a fixed order for the error-heatmap axes.
CATEGORIES = ["EXITUS <48h", "EXITUS >48h", "ALIVE <48h", "ALIVE >48h"]


def _error_bars(df: Any) -> dict[str, Any]:
    """Proportion of each error-severity group (0..3) + the mean error."""
    vc = df["error"].value_counts(normalize=True).sort_index()
    return {
        "groups": [int(g) for g in vc.index],
        "proportions": [round(float(p), 4) for p in vc.values],
        "mean": round(float(df["error"].mean()), 4),
    }


def _error_heatmap(df: Any) -> dict[str, Any]:
    """Counts per (real category, predicted category) cell + that cell's error."""
    g = (
        df.groupby(["real_color_group", "color_group"])
        .agg(count=("error", "size"), error=("error", "first"))
        .reset_index()
    )
    cells = [
        {"real": r["real_color_group"], "pred": r["color_group"],
         "count": int(r["count"]), "error": int(r["error"])}
        for _, r in g.iterrows()
    ]
    return {"categories": CATEGORIES, "cells": cells}


def _params_thresholds(run_id: str, path: str) -> tuple[float | None, float | None]:
    import json

    try:
        with open(download_artifact(run_id, path)) as fh:
            p = json.load(fh)
        return p.get("th_mort"), p.get("th_disch")
    except Exception:  # noqa: BLE001 - no params file → just skip the operating point
        return None, None


def run_charts(run_id: str) -> dict[str, Any]:
    """Chart data for one run, so the UI can draw ROC/error plots in JS.

    Inference runs return ROC (with operating point) + error bars + heatmap;
    metrics runs return ROC only. ``roc`` is None when no predictions exist.
    """
    # Inference run: wide table with both models sharing the same rows.
    try:
        df = _read_predictions(run_id, "inference/results_inference")
    except Exception:  # noqa: BLE001 - not an inference run / artifact missing
        df = None
    if df is not None:
        th_m, th_d = _params_thresholds(run_id, "inference/model_parameters_inference.json")
        return {
            "roc": {
                "mort": _roc_points(df["mortality_gt"], df["mortality_prob"], th_m),
                "disch": _roc_points(df["disch_gt"], df["disch_prob"], th_d),
            },
            "error_bars": _error_bars(df),
            "error_heatmap": _error_heatmap(df),
        }

    # Metrics run: long table (model, prob, gt) — ROC only, no error categories.
    try:
        df = _read_predictions(run_id, "metrics/results_metrics")
    except Exception:  # noqa: BLE001 - run has no predictions artifact
        return {"roc": None, "error_bars": None, "error_heatmap": None}
    th_m, th_d = _params_thresholds(run_id, "metrics/model_parameters_test.json")
    m = df[df["model"] == "mortality"]
    d = df[df["model"] == "discharge"]
    return {
        "roc": {
            "mort": _roc_points(m["gt"], m["prob"], th_m),
            "disch": _roc_points(d["gt"], d["prob"], th_d),
        },
        "error_bars": None,
        "error_heatmap": None,
    }

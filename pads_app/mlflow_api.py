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


def successful_inference_runs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Finished inference runs from succeeded jobs, ranked by mean error (asc).

    One entry per inference run (the original baseline + each retrained type)
    across every successful job, so the UI can show a global leaderboard.
    ``mean_error`` is the per-stay error severity averaged over the test cohort
    (lower is better). ``jobs`` is the manager's public job list (passed in to
    keep this module decoupled from the JobManager).
    """
    if not get_settings().mlflow_enabled:
        return []
    out: list[dict[str, Any]] = []
    for job in jobs:
        if job.get("status") != "succeeded":
            continue
        for run in job_runs(job["id"]):
            if run.get("step") != "inference" or run.get("status") != "FINISHED":
                continue
            metrics = run["metrics"]
            # test_type + mean error share the key inf/<test_type>/mean_error, so each
            # inference window (full / last_48h / …) is its own ranked entry.
            test_type, mean_error = None, None
            for k, v in metrics.items():
                if k.startswith("inf/") and k.endswith("/mean_error"):
                    test_type = k[len("inf/"):-len("/mean_error")]
                    mean_error = v
                    break
            if mean_error is None:
                continue
            crit = next((v for k, v in metrics.items() if k.endswith("/critical_error_rate")), None)
            out.append(
                {
                    "job_id": job["id"],
                    "run_id": run["run_id"],
                    "run_name": run["run_name"],
                    "retrain_type": run["retrain_type"] or "original",
                    "test_type": test_type or "full",
                    "data_filename": job.get("params", {}).get("data_filename"),
                    "created_at": job.get("created_at"),
                    "mean_error": mean_error,
                    "critical_error_rate": crit,
                    # The job's requested training params (the user's choices), so the
                    # window can show them next to the charts.
                    "params": job.get("params", {}),
                    "metrics": metrics,
                }
            )
    out.sort(key=lambda e: e["mean_error"])
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
    metrics. Each run's artifacts are downloaded into a folder named after its
    retrain_type, then the whole tree is zipped. Inference runs are the one case
    that needs care (see the loop below).
    """
    import re
    import shutil
    import tempfile

    runs = job_runs(job_id)
    if not runs:
        raise ValueError(f"No MLflow runs found for job {job_id}")

    staging = tempfile.mkdtemp()
    # Drop the job's run parameters at the root of the archive (next to the
    # per-type model folders) so whoever unzips it can see exactly which
    # settings produced these models without digging into the app.
    _write_job_parameters(job_id, staging)

    client = _client()
    # Group by retrain_type (original/full/lstm/dense/scratch). Most runs of a
    # type sit under distinct subfolders (metrics, mortality_model,
    # discharge_model, dataset) and merge cleanly. The exception is inference:
    # a type has one inference run per test_type (full/last_48h/last_96h) and
    # they ALL log under "inference/" with the same filenames, so merging them
    # into one folder would overwrite all but one. Rename each inference run's
    # folder to inference_<test_type> so every window survives side by side.
    for run in runs:
        top = re.sub(r"[^A-Za-z0-9._-]+", "_", run["retrain_type"]) or "other"
        dest = os.path.join(staging, top)
        os.makedirs(dest, exist_ok=True)
        client.download_artifacts(run["run_id"], "", dest)
        if run.get("step") == "inference":
            _split_inference_folder(dest, run)

    base = os.path.join(tempfile.mkdtemp(), f"pads_job_{job_id}")
    return shutil.make_archive(base, "zip", staging)


def _split_inference_folder(dest: str, run: dict[str, Any]) -> None:
    """Rename an inference run's ``inference/`` folder to ``inference_<test_type>``.

    All three inference runs of a retrain_type (full/last_48h/last_96h) log under
    the same ``inference/`` path with identical filenames, so leaving them merged
    in ``dest`` makes them overwrite each other. We tag the folder with the run's
    test_type (from the ``test_type_active`` param, falling back to the run name
    ``inference_<rt>_<test_type>``) so each window's results survive separately.
    """
    import re
    import shutil

    test_type = run.get("params", {}).get("test_type_active")
    if not test_type:
        # Fallback: run name is "inference_<retrain_type>_<test_type>".
        name = run.get("run_name", "")
        rt = run.get("retrain_type", "")
        prefix = f"inference_{rt}_"
        test_type = name[len(prefix):] if name.startswith(prefix) else (name or "unknown")
    test_type = re.sub(r"[^A-Za-z0-9._-]+", "_", test_type) or "unknown"

    src = os.path.join(dest, "inference")
    if not os.path.isdir(src):
        return
    target = os.path.join(dest, f"inference_{test_type}")
    if os.path.isdir(target):
        shutil.rmtree(target)  # re-download of the same window → replace
    shutil.move(src, target)


def _write_job_parameters(job_id: str, staging: str) -> None:
    """Write the job's stored metadata as ``parameters.json`` into ``staging``.

    The job record (status, steps and the ``params`` used to launch it) lives at
    ``<state_dir>/jobs/<job_id>.json``. Copying it to the archive root gives the
    downloaded bundle a self-describing parameter manifest. Best-effort: if the
    record is missing the archive is still built without it.
    """
    import json

    meta_path = get_settings().state_dir / "jobs" / f"{job_id}.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - no record (e.g. MLflow-only run) → skip
        return
    out = os.path.join(staging, "parameters.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


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
        # test_type + mean error come from the same logged key, inf/<type>/mean_error,
        # so the UI can filter the comparison by inference window.
        test_type, mean_error = None, None
        for k, v in run["metrics"].items():
            if k.startswith("inf/") and k.endswith("/mean_error"):
                test_type = k[len("inf/"):-len("/mean_error")]
                mean_error = v
                break
        models.append({
            "retrain_type": run["retrain_type"] or "original",
            "test_type": test_type or "full",
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

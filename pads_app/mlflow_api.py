"""Read-only MLflow queries used by the frontend.

Everything here degrades to empty results when MLflow tracking is off, so the
UI keeps working (minus live metrics) even without a tracking server.
"""
from __future__ import annotations

import math
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

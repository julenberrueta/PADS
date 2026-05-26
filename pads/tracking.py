"""MLflow tracking with graceful no-op fallback.

If `MLFLOW_TRACKING_URI` is not set in the environment, every function in this
module is a no-op. The pipeline runs identically with or without MLflow.

Optional env vars:
  MLFLOW_TRACKING_URI       backend store (e.g. http://server:5000)
  PADS_MLFLOW_EXPERIMENT    experiment name (default "pads")
  PADS_RUN_TAGS             "k1=v1,k2=v2" extra tags applied to every run
"""
from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Flipped by check_connection() when the tracking URI is set but the server is
# unreachable, so the rest of the process degrades cleanly to no-op tracking.
_RUNTIME_DISABLED = False


def enabled() -> bool:
    return bool(os.getenv("MLFLOW_TRACKING_URI")) and not _RUNTIME_DISABLED


def check_connection(*, timeout: float = 5.0) -> bool:
    """Probe the MLflow tracking server once, at pipeline start.

    Only http(s) tracking URIs are probed — file:// and sqlite:// stores are
    local, so there is nothing to reach. If the server cannot be reached,
    tracking is disabled for the rest of the process and a warning is printed;
    the pipeline keeps running and still writes models/artifacts to disk.

    Returns True if tracking is off by design or the server is reachable,
    False if the configured server is unreachable (tracking now disabled).
    """
    global _RUNTIME_DISABLED
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        return True  # tracking off by design — nothing to check

    from urllib.parse import urlparse

    if urlparse(uri).scheme not in ("http", "https"):
        return True  # local store — no server to probe

    import socket
    from urllib.error import HTTPError, URLError
    from urllib.request import urlopen

    try:
        urlopen(uri.rstrip("/") + "/health", timeout=timeout)
    except HTTPError:
        # Any HTTP status (even 401/403/5xx) means the server answered → reachable.
        pass
    except (URLError, socket.timeout, OSError) as exc:
        _RUNTIME_DISABLED = True
        print(
            f"\n[PADS] WARNING: MLflow tracking is ON (MLFLOW_TRACKING_URI={uri}) "
            f"but the server is unreachable ({exc}).\n"
            f"[PADS] Continuing WITHOUT tracking - models and artifacts are still "
            f"written to disk, but nothing is logged to MLflow.\n"
            f"[PADS] On Windows + Docker, try 127.0.0.1 instead of localhost.\n",
            file=sys.stderr,
        )
        return False
    print(f"[PADS] MLflow tracking ON: {uri}", file=sys.stderr)
    return True


def _parse_env_tags() -> dict[str, str]:
    raw = os.getenv("PADS_RUN_TAGS", "")
    out: dict[str, str] = {}
    for piece in (p.strip() for p in raw.split(",") if p.strip()):
        if "=" in piece:
            k, v = piece.split("=", 1)
            out[k.strip()] = v.strip()
    return out


PARENT_RUN_ID_FILE = ".mlflow_parent_run_id"


def save_parent_run_id(run_id: str, base_path: str | Path = ".") -> None:
    """Persist a run id so other processes can nest under it (cross-process linkage)."""
    if not enabled():
        return
    Path(base_path).joinpath(PARENT_RUN_ID_FILE).write_text(run_id)


def load_parent_run_id(base_path: str | Path = ".") -> str | None:
    """Read the parent run id written by a previous step in the same pipeline run."""
    if not enabled():
        return None
    p = Path(base_path) / PARENT_RUN_ID_FILE
    return p.read_text().strip() if p.exists() else None


def clear_parent_run_id(base_path: str | Path = ".") -> None:
    """Delete the parent run id file, ignoring missing files."""
    p = Path(base_path) / PARENT_RUN_ID_FILE
    if p.exists():
        p.unlink()


@contextmanager
def run(name: str, *, parent_run_id: str | None = None, **tags: Any) -> Iterator[Any]:
    """Context manager that opens an MLflow run.

    Nesting rules:
      - If another run is already active in this process, the new run is nested
        under it (in-process nesting, e.g. retrain_mortality inside retrain_models).
      - If `parent_run_id` is supplied (cross-process nesting), the new run is
        tagged with `mlflow.parentRunId=<id>` so the UI shows it under that parent.

    If MLflow is disabled, yields None and does nothing.
    """
    if not enabled():
        yield None
        return

    import mlflow

    experiment = os.getenv("PADS_MLFLOW_EXPERIMENT", "pads")
    mlflow.set_experiment(experiment)
    in_process_nested = mlflow.active_run() is not None

    extra_tags: dict[str, str] = {}
    if parent_run_id and not in_process_nested:
        extra_tags["mlflow.parentRunId"] = parent_run_id

    with mlflow.start_run(run_name=name, nested=in_process_nested) as r:
        all_tags = {
            **_parse_env_tags(),
            **extra_tags,
            **{k: str(v) for k, v in tags.items()},
        }
        for k, v in all_tags.items():
            mlflow.set_tag(k, v)
        yield r


def log_params(params: dict[str, Any]) -> None:
    if not enabled():
        return
    import mlflow
    # MLflow only accepts scalar params
    flat = {k: v for k, v in params.items() if v is None or isinstance(v, (str, int, float, bool))}
    mlflow.log_params(flat)


def log_metric(key: str, value: float, step: int | None = None) -> None:
    if not enabled():
        return
    import mlflow
    mlflow.log_metric(key, float(value), step=step)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    if not enabled():
        return
    import mlflow
    mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)


def log_artifact(path: str | Path, artifact_path: str | None = None) -> None:
    if not enabled():
        return
    p = Path(path)
    if not p.exists():
        return
    import mlflow
    mlflow.log_artifact(str(p), artifact_path=artifact_path)


def log_dict(obj: dict, artifact_filename: str) -> None:
    if not enabled():
        return
    import mlflow
    mlflow.log_dict(obj, artifact_filename)


def file_sha256(path: str | Path) -> str:
    """Hex SHA-256 digest of a file, suitable as a dataset fingerprint."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

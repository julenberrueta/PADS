"""App settings — paths and MLflow wiring, resolved once at import time.

The app drives the *existing* pipeline, so it must run against a real project
layout (``data/``, ``models/``, ``normalizers/``, ``results/``). ``base_path``
defaults to the current working directory, overridable via PADS_APP_BASE_PATH.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# Reuse the CLI's .env loader so MLflow settings are picked up identically.
from pads.cli import _load_dotenv


@dataclass(frozen=True)
class Settings:
    base_path: Path
    data_dir: Path
    state_dir: Path  # where the app persists its job history (metadata + logs)
    experiment: str
    tracking_uri: str | None
    # Upper bound on concurrently *running* jobs. The pipeline writes to shared
    # folders (data/processed, models/, results/, .mlflow_parent_run_id), so v1
    # runs one job at a time and queues the rest.
    max_concurrent_jobs: int = field(default=1)

    @property
    def mlflow_enabled(self) -> bool:
        return bool(self.tracking_uri)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    base_path = Path(os.getenv("PADS_APP_BASE_PATH", ".")).resolve()
    _load_dotenv(base_path / ".env")
    state_dir = Path(os.getenv("PADS_APP_STATE_DIR", base_path / ".pads_app")).resolve()
    return Settings(
        base_path=base_path,
        data_dir=base_path / "data",
        state_dir=state_dir,
        experiment=os.getenv("PADS_MLFLOW_EXPERIMENT", "pads"),
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI") or None,
    )

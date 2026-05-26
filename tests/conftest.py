"""Shared test fixtures."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_CSV = REPO_ROOT / "data" / "synthetic_dataset.csv"
MODELS_DIR = REPO_ROOT / "models"


@pytest.fixture(scope="session")
def synthetic_csv_path() -> Path:
    if not SYNTHETIC_CSV.is_file():
        pytest.skip(f"synthetic dataset not present at {SYNTHETIC_CSV}")
    return SYNTHETIC_CSV


@pytest.fixture
def project_dir(tmp_path: Path, synthetic_csv_path: Path) -> Path:
    """Build a clean project layout under tmp_path, copying the synthetic CSV."""
    (tmp_path / "data").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "normalizers").mkdir()
    (tmp_path / "results").mkdir()
    shutil.copy(synthetic_csv_path, tmp_path / "data" / "synthetic_dataset.csv")
    return tmp_path


@pytest.fixture
def project_dir_with_models(project_dir: Path) -> Path:
    """Like `project_dir` but also copies the base LSTM models from the repo."""
    for name in ("lstm_mortality_model.keras", "lstm_disch_model.keras"):
        src = MODELS_DIR / name
        if not src.is_file():
            pytest.skip(f"base model {name} not present in repo")
        shutil.copy(src, project_dir / "models" / name)
    return project_dir


@pytest.fixture
def tiny_stay_df() -> pd.DataFrame:
    """A tiny dataset: 2 stays of 50 hours each, with all mandatory columns numeric."""
    rng = np.random.default_rng(0)
    rows = []
    for sid in (1000, 2000):
        for hr in range(50):
            rows.append(
                {
                    "stay_id": sid,
                    "hr": hr,
                    "rate_epinephrine": rng.uniform(0, 5),
                    "rate_norepinephrine": rng.uniform(0, 2),
                    "rate_dopamine": rng.uniform(0, 10),
                    "rate_dobutamine": rng.uniform(0, 5),
                    "meanbp_min": rng.uniform(40, 120),
                    "pao2fio2ratio_novent": rng.uniform(100, 400),
                    "pao2fio2ratio_vent": rng.uniform(100, 400),
                    "gcs_min": rng.integers(3, 15),
                    "bilirubin_max": rng.uniform(0.5, 5),
                    "creatinine_max": rng.uniform(0.5, 3),
                    "platelet_min": rng.uniform(50, 400),
                    "admission_age": 65,
                    "icu_expire_flag": int(sid == 2000),
                    "admission_type_Medical": 1,
                    "admission_type_ScheduledSurgical": 0,
                    "admission_type_UnscheduledSurgical": 0,
                    "charlson_comorbidity_index": 3,
                }
            )
    return pd.DataFrame(rows)

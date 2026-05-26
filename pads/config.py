"""Pipeline configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

RetrainType = Literal["full", "dense", "lstm", "scratch"]
TestType = Literal["full", "last_48h", "last_96h", "first_48h"]


class PADSConfig(BaseModel):
    """Configuration for the PADS pipeline.

    Inference model filenames are derived from `retrain_type` unless overridden.
    """

    base_path: Path = Path("./")

    # base model filenames (used as input to retraining)
    retrain_mort_model: str = "lstm_mortality_model.keras"
    retrain_disch_model: str = "lstm_disch_model.keras"

    # normalizer filenames — fitted on YOUR training data by prepare_data.
    # The committed mimic_iv_*.pkl baselines are left untouched as a reference.
    mort_normalizer: str = "mortality_normalizer.pkl"
    disch_normalizer: str = "discharge_normalizer.pkl"

    # retraining / inference modes
    retrain_type: RetrainType = "scratch"
    test_type: TestType = "full"

    # explicit overrides (if None, derived from retrain_type)
    inference_mort_model: str | None = None
    inference_disch_model: str | None = None

    # training
    learning_rate_mort: float = 1e-5
    learning_rate_disch: float = 1e-5
    epochs: int = 1000
    batch_size: int = 100
    early_stopping_patience: int = 50

    # parallelism
    parallel: bool = True
    n_jobs: int = 20

    # reproducibility
    seed: int = 42

    @model_validator(mode="after")
    def _derive_inference_names(self) -> PADSConfig:
        if self.inference_mort_model is None:
            self.inference_mort_model = f"RETRAINED_{self.retrain_type}_{self.retrain_mort_model}"
        if self.inference_disch_model is None:
            self.inference_disch_model = f"RETRAINED_{self.retrain_type}_{self.retrain_disch_model}"
        return self

    # path helpers -----------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.base_path / "data"

    @property
    def processed_dir(self) -> Path:
        """Generated intermediate artifacts (medians, stay splits, windowed .pkl)."""
        return self.data_dir / "processed"

    @property
    def model_dir(self) -> Path:
        return self.base_path / "models"

    @property
    def norm_dir(self) -> Path:
        return self.base_path / "normalizers"

    @property
    def results_dir(self) -> Path:
        return self.base_path / "results"

    @property
    def run_results_dir(self) -> Path:
        """Per-model results subfolder (keyed by retrain_type) so a sweep over
        retrain types keeps each variant's outputs side by side instead of
        overwriting them."""
        return self.results_dir / self.retrain_type

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.processed_dir, self.model_dir,
                  self.norm_dir, self.results_dir, self.run_results_dir):
            d.mkdir(parents=True, exist_ok=True)

"""Pipeline configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

# "original" is evaluation-only: the shipped base model with no retraining. It is
# a valid label for calculate_metrics/inference (to baseline the off-the-shelf
# model) but never for retrain_models — see PADSPipeline.retrain_models.
RetrainType = Literal["full", "dense", "lstm", "scratch", "original"]
TestType = Literal["full", "last_48h", "last_96h", "first_48h"]
# Criterion to pick the decision threshold from the ROC/PR curve when no fixed
# threshold is set. Canonical definition lives here (config stays import-light);
# pads.eval.thresholds re-imports it. See optimal_threshold for what each does.
ThresholdMethod = Literal["youden", "min_distance", "precision_recall"]
# Validation metric that EarlyStopping/ModelCheckpoint watch during retraining.
# Must match a compiled metric name (see pads.models.*). "loss" is minimised; the
# rest are maximised (trainer.fit derives the mode).
MonitorMetric = Literal["loss", "AUC", "accuracy", "f1_score", "precision", "recall"]


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

    # Optional FIXED decision thresholds. When both are set, calculate_metrics and
    # inference use them as the operating point instead of picking the optimum from
    # the data — e.g. to evaluate the shipped "original" model at its published
    # thresholds on your own cohort. The ROC curve / AUC still reflect your data;
    # only the marked point and the threshold-dependent metrics (F1/precision/
    # recall/error) change. Default None: every model picks its own data-derived
    # optimum. The app only sets them for the "original" baseline run, so retrained
    # models are never forced onto the base model's operating point.
    fixed_th_mort: float | None = None
    fixed_th_disch: float | None = None

    # How calculate_metrics/inference pick the operating point when no fixed
    # threshold is set: "precision_recall" maximises F1 (handles class imbalance,
    # the default); "youden"/"min_distance" balance sensitivity vs specificity.
    threshold_method: ThresholdMethod = "precision_recall"

    # training
    learning_rate_mort: float = 1e-5
    learning_rate_disch: float = 1e-5
    epochs: int = 1000
    batch_size: int = 100
    early_stopping_patience: int = 20
    # Validation metric EarlyStopping/ModelCheckpoint track (the value used to
    # decide the "best" epoch and when to stop). Default "loss" (minimised).
    monitor_metric: MonitorMetric = "loss"

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

    @model_validator(mode="after")
    def _check_fixed_thresholds(self) -> PADSConfig:
        # Fixed thresholds only make sense as a pair (one per model); a single one
        # would leave the other silently data-derived, which is a footgun.
        if (self.fixed_th_mort is None) != (self.fixed_th_disch is None):
            raise ValueError(
                "fixed_th_mort and fixed_th_disch must be set together (or neither)."
            )
        return self

    @property
    def fixed_thresholds(self) -> tuple[float, float] | None:
        """The (mortality, discharge) fixed operating point, or None if not set."""
        if self.fixed_th_mort is None or self.fixed_th_disch is None:
            return None
        return (self.fixed_th_mort, self.fixed_th_disch)

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

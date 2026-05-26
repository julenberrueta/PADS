"""Dataset schema, feature lists, and preprocessing rules."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

N_TIME_OFFSETS = 48

IMPUTE_FIRST_ROW = [
    "gcs_min", "meanbp_min", "bilirubin_max", "platelet_min", "creatinine_max",
]

NAN_RULES: dict[str, str] = {
    "pao2fio2ratio_novent": "zero",
    "pao2fio2ratio_vent":   "zero",
    "gcs_min":              "prev_value",
    "rate_norepinephrine":  "zero",
    "rate_epinephrine":     "zero",
    "rate_dopamine":        "zero",
    "rate_dobutamine":      "zero",
    "meanbp_min":           "prev_value",
    "bilirubin_max":        "prev_value",
    "platelet_min":         "prev_value",
    "creatinine_max":       "prev_value",
}

OUTLIER_CORRECTION: dict[str, tuple[float, float]] = {
    "rate_epinephrine":     (0, 10),
    "rate_norepinephrine":  (0, 5),
    "rate_dopamine":        (0, 50),
    "rate_dobutamine":      (0, 40),
    "meanbp_min":           (10, 150),
    "pao2fio2ratio_novent": (50, 600),
    "pao2fio2ratio_vent":   (50, 600),
    "bilirubin_max":        (0.1, 70),
    "creatinine_max":       (0.2, 20),
    "platelet_min":         (5, 2000),
}

MANDATORY_COLUMNS = [
    "stay_id", "hr",
    "rate_epinephrine", "rate_norepinephrine", "rate_dopamine", "rate_dobutamine",
    "meanbp_min", "pao2fio2ratio_novent", "pao2fio2ratio_vent", "gcs_min",
    "bilirubin_max", "creatinine_max", "platelet_min",
    "admission_age", "icu_expire_flag",
    "admission_type_Medical", "admission_type_ScheduledSurgical",
    "admission_type_UnscheduledSurgical", "charlson_comorbidity_index",
]

FEATURES = [
    "rate_epinephrine", "rate_norepinephrine", "rate_dopamine", "rate_dobutamine",
    "meanbp_min", "pao2fio2ratio_novent", "pao2fio2ratio_vent", "gcs_min",
    "bilirubin_max", "creatinine_max", "platelet_min",
    "admission_age", "charlson_comorbidity_index",
    "admission_type_Medical", "admission_type_ScheduledSurgical",
    "admission_type_UnscheduledSurgical",
]


class DatasetValidationError(ValueError):
    """Raised when an input dataset does not match the expected schema."""


def validate_dataset(path: str | Path) -> pd.DataFrame:
    """Validate that the CSV/Parquet at `path` matches the expected schema, then return it.

    Raises DatasetValidationError on any missing column or non-numeric dtype.
    """
    from pads.data.loader import load_dataframe
    
    path = Path(path)
    if not path.is_file():
        raise DatasetValidationError(f"Dataset not found: {path}")

    df = load_dataframe(path)
    missing = [c for c in MANDATORY_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetValidationError(f"Missing mandatory columns: {missing}")

    non_numeric = [c for c in MANDATORY_COLUMNS if not is_numeric_dtype(df[c])]
    if non_numeric:
        raise DatasetValidationError(f"Non-numeric mandatory columns: {non_numeric}")

    return df

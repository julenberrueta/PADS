from __future__ import annotations

import pandas as pd
import pytest

from pads.data.schema import DatasetValidationError, validate_dataset


def test_validate_passes_on_valid_csv(synthetic_csv_path):
    df = validate_dataset(synthetic_csv_path)
    assert isinstance(df, pd.DataFrame)
    assert "stay_id" in df.columns


def test_validate_raises_on_missing_file(tmp_path):
    with pytest.raises(DatasetValidationError):
        validate_dataset(tmp_path / "does_not_exist.csv")


def test_validate_raises_on_missing_column(tmp_path, tiny_stay_df):
    p = tmp_path / "bad.csv"
    tiny_stay_df.drop(columns=["gcs_min"]).to_csv(p, index=False)
    with pytest.raises(DatasetValidationError, match="gcs_min"):
        validate_dataset(p)


def test_validate_raises_on_non_numeric(tmp_path, tiny_stay_df):
    p = tmp_path / "bad.csv"
    df = tiny_stay_df.copy()
    df["meanbp_min"] = "not_a_number"
    df.to_csv(p, index=False)
    with pytest.raises(DatasetValidationError, match="meanbp_min"):
        validate_dataset(p)

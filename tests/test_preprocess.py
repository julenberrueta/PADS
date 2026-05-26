from __future__ import annotations

import numpy as np
import pandas as pd

from pads.data.preprocess import apply_nan_rules, correct_outliers, impute_first_row, preprocess


def test_outliers_are_clipped():
    df = pd.DataFrame(
        {"meanbp_min": [-10, 50, 500], "stay_id": [1, 1, 1], "hr": [0, 1, 2]}
    )
    out = correct_outliers(df)
    assert out["meanbp_min"].tolist() == [10, 50, 150]


def test_impute_first_row_uses_medians():
    df = pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 2],
            "hr": [0, 1, 0, 1],
            "gcs_min": [np.nan, 12.0, np.nan, 10.0],
        }
    )
    out = impute_first_row(df, {"gcs_min": 7.0})
    assert out.loc[0, "gcs_min"] == 7.0
    assert out.loc[2, "gcs_min"] == 7.0
    # second row left untouched even if NaN — only hr==0 gets imputed here
    assert out.loc[1, "gcs_min"] == 12.0


def test_apply_nan_rules_zero_and_ffill():
    df = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 2, 2],
            "hr": [0, 1, 2, 0, 1],
            "rate_epinephrine": [np.nan, np.nan, 0.5, np.nan, 0.0],
            "gcs_min": [10.0, np.nan, np.nan, 12.0, np.nan],
        }
    )
    out = apply_nan_rules(df, show_progress=False)
    # zero-fill
    assert out["rate_epinephrine"].iloc[0] == 0
    # ffill within stay
    assert out["gcs_min"].iloc[1] == 10.0
    assert out["gcs_min"].iloc[2] == 10.0
    assert out["gcs_min"].iloc[4] == 12.0  # ffilled from stay 2's first row


def test_preprocess_pipeline_sorts_and_fills(tiny_stay_df):
    df = tiny_stay_df.sample(frac=1, random_state=1).reset_index(drop=True)
    medians = {c: 1.0 for c in ["gcs_min", "meanbp_min", "bilirubin_max", "platelet_min", "creatinine_max"]}
    out = preprocess(df, medians)
    # sorted by (stay_id, hr)
    assert out["stay_id"].is_monotonic_increasing
    grouped = out.groupby("stay_id")["hr"]
    assert all(g.is_monotonic_increasing for _, g in grouped)
    # no NaN remains in known feature columns
    for c in ["rate_epinephrine", "gcs_min", "meanbp_min"]:
        assert out[c].isna().sum() == 0

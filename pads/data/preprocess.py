"""Outlier correction, first-row imputation, NaN rule application."""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from pads.data.schema import NAN_RULES, OUTLIER_CORRECTION


def correct_outliers(df: pd.DataFrame) -> pd.DataFrame:
    for feat, (lo, hi) in OUTLIER_CORRECTION.items():
        if feat in df.columns:
            df[feat] = df[feat].clip(lo, hi)
    return df


def impute_first_row(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """Fill NaNs in the first hour of each stay using global medians."""
    mask = df["hr"] == 0
    for col, value in medians.items():
        if col in df.columns:
            df.loc[mask, col] = df.loc[mask, col].fillna(value)
    return df


def apply_nan_rules(
    df: pd.DataFrame,
    rules: dict[str, str] = NAN_RULES,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Apply per-feature NaN imputation rules: 'zero', 'prev_value', or a constant."""
    zero_cols = [k for k, v in rules.items() if v == "zero" and k in df.columns]
    ffill_cols = [k for k, v in rules.items() if v == "prev_value" and k in df.columns]
    other = [(k, v) for k, v in rules.items() if v not in ("zero", "prev_value") and k in df.columns]

    pbar = tqdm if show_progress else (lambda x, **_: x)

    for c in pbar(zero_cols, desc="fill zeros"):
        df[c] = df[c].fillna(0)
    if ffill_cols:
        df[ffill_cols] = df.groupby("stay_id")[ffill_cols].ffill()
    for c, v in pbar(other, desc="fill values"):
        df[c] = df[c].fillna(v)
    return df


def preprocess(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """Sort, clip outliers, impute first row, apply NaN rules."""
    df = df.sort_values(["stay_id", "hr"]).reset_index(drop=True)
    df = correct_outliers(df)
    df = impute_first_row(df, medians)
    df = apply_nan_rules(df)
    return df

"""Time-series windowing for mortality and discharge datasets.

Conventions preserved from code_v3 to keep saved pickles compatible:
- mortality windows: time axis is *most-recent-first*  (consumer reverses with [::-1])
- discharge  windows: time axis is *oldest-first*       (consumer uses as-is)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

from pads.data.schema import FEATURES, N_TIME_OFFSETS

_FIXED_COLS = ["stay_id"]


# --- mortality / rolling-window dataset ---------------------------------------


def _stay_rolling_windows(stay_df: pd.DataFrame, n: int = N_TIME_OFFSETS) -> np.ndarray:
    """For a single stay, return rolling windows of shape (n_windows, n, len(FEATURES)+1).

    Time axis ordering matches the v3 merge convention: index 0 = most recent hour.
    """
    g = stay_df.sort_values("hr")
    arr = g[_FIXED_COLS + FEATURES].to_numpy()
    los = arr.shape[0]
    if los < n:
        return np.empty((0, n, arr.shape[1]))
    # sliding_window_view over rows
    win = sliding_window_view(arr, window_shape=n, axis=0)   # (los-n+1, n_cols, n)
    win = np.swapaxes(win, 1, 2)                              # (los-n+1, n, n_cols)
    return win[:, ::-1, :].copy()                             # most-recent-first


def build_rolling_windows(df: pd.DataFrame, n: int = N_TIME_OFFSETS) -> dict[int, np.ndarray]:
    """Build rolling-window arrays for every stay with los >= n.

    Returns dict[stay_id, ndarray of shape (n_windows, n, len(FEATURES)+1)].
    """
    out: dict[int, np.ndarray] = {}
    groups = df.groupby("stay_id", sort=False)
    for sid, g in tqdm(groups, desc="rolling windows", total=groups.ngroups):
        win = _stay_rolling_windows(g, n=n)
        if win.shape[0] > 0:
            out[int(sid)] = win
    return out


# --- discharge / 5-anchor windows ---------------------------------------------


def _stay_discharge_chunks(
    stay_df: pd.DataFrame, n: int = N_TIME_OFFSETS
) -> tuple[np.ndarray, np.ndarray]:
    """For a single stay, return up to 5 discharge anchor windows and their outcomes.

    Time axis ordering is *as-is* (ascending hr), matching v3.
    """
    g = stay_df.sort_values("hr").reset_index(drop=True)
    los = int(g["los"].iloc[0])
    if los < n:
        return np.empty((0, n, len(FEATURES) + 1)), np.empty((0, 1))

    half_los = los // 2
    cols = _FIXED_COLS + FEATURES
    n_cols = len(cols)

    anchors: list[pd.DataFrame] = [g[(g["hr"] >= 0) & (g["hr"] < n)]]
    if los >= n + 24:
        anchors.append(g[(g["hr"] >= 24) & (g["hr"] < n + 24)])
    if half_los >= n:
        anchors.append(g[(g["hr"] >= half_los - n) & (g["hr"] < half_los)])
        anchors.append(g[(g["hr"] >= half_los) & (g["hr"] < half_los + n)])
    anchors.append(g[g["hr"] > los - n])

    arrs, outs = [], []
    for a in anchors:
        if len(a) != n:
            continue
        arrs.append(a[cols].to_numpy().reshape(1, n, n_cols))
        outs.append(a["disch_48h"].to_numpy()[-1:].reshape(-1, 1))

    if not arrs:
        return np.empty((0, n, n_cols)), np.empty((0, 1))
    return np.vstack(arrs), np.vstack(outs)


def build_discharge_windows(
    df: pd.DataFrame, n: int = N_TIME_OFFSETS
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Build the 5-anchor discharge windows for every eligible stay."""
    all_data: dict[int, np.ndarray] = {}
    outcomes: dict[int, np.ndarray] = {}
    groups = df.groupby("stay_id", sort=False)
    for sid, g in tqdm(groups, desc="discharge chunks", total=groups.ngroups):
        arrs, outs = _stay_discharge_chunks(g, n=n)
        if arrs.shape[0] > 0:
            all_data[int(sid)] = arrs
            outcomes[int(sid)] = outs
    return all_data, outcomes


# --- helpers shared by both -----------------------------------------------------


def disch_label_series(df: pd.DataFrame, n: int = N_TIME_OFFSETS) -> pd.Series:
    """Return the 0/1 `disch_48h` flag for each row: 1 if within the last `n` hours of the stay."""
    return (df["hr"] >= df["los"] - (n - 1)).astype(int)


def disch_outcomes_by_stay(
    df: pd.DataFrame, n: int = N_TIME_OFFSETS
) -> dict[int, list[int]]:
    """Per-stay list of `disch_48h` flags, one per rolling window.

    The label for window k is the `disch_48h` of that window's most-recent row.
    `build_rolling_windows` slides over rows *by position*, so window k ends at
    row k+n-1 and there are (n_rows - n + 1) windows. We therefore take the
    `disch_48h` of rows[n-1:] after sorting by hr — which aligns one-to-one with
    the windows regardless of whether a stay's hours are contiguous or start at 0.

    (The previous `df["hr"] >= n-1` filter desynced from the windows whenever a
    stay had gaps in `hr` or didn't start at 0, producing length mismatches
    downstream in inference.)
    """
    out: dict[int, list[int]] = {}
    for sid, g in df.groupby("stay_id", sort=False):
        labels = g.sort_values("hr")["disch_48h"].to_numpy()[n - 1:]
        if labels.size > 0:
            out[int(sid)] = labels.tolist()
    return out


def stay_to_mortality_outcome(df: pd.DataFrame) -> dict[int, int]:
    """One mortality label per stay."""
    s = df[["stay_id", "icu_expire_flag"]].drop_duplicates().set_index("stay_id")["icu_expire_flag"]
    return {int(k): int(v) for k, v in s.items()}

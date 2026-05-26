from __future__ import annotations

from pads.data.schema import FEATURES, N_TIME_OFFSETS
from pads.data.windowing import (
    build_discharge_windows,
    build_rolling_windows,
    disch_label_series,
    disch_outcomes_by_stay,
    stay_to_mortality_outcome,
)


def _enrich_for_discharge(df, n=N_TIME_OFFSETS):
    df = df.copy()
    df["los"] = df.groupby("stay_id")["hr"].transform("max")
    df["half_los"] = (df["los"] // 2).astype(int)
    df["disch_48h"] = (df["hr"] >= df["los"] - (n - 1)).astype(int)
    return df


def test_rolling_windows_shape(tiny_stay_df):
    df = tiny_stay_df.copy()
    df["los"] = df.groupby("stay_id")["hr"].transform("max")
    out = build_rolling_windows(df)
    # 2 stays of 50 hours each, n=48 → 50 - 48 + 1 = 3 windows per stay... but los is 49 (hr=0..49)
    los_per_stay = df.groupby("stay_id")["hr"].max().to_dict()
    for sid, arr in out.items():
        expected_windows = los_per_stay[sid] - N_TIME_OFFSETS + 2  # because we have hr=0..los
        assert arr.shape == (expected_windows, N_TIME_OFFSETS, len(FEATURES) + 1)


def test_rolling_window_time_axis_is_most_recent_first(tiny_stay_df):
    """First time index should correspond to the latest hour in the window."""
    df = tiny_stay_df.copy()
    df["los"] = df.groupby("stay_id")["hr"].transform("max")
    # mark a feature with the hr value so we can check ordering
    df["rate_epinephrine"] = df["hr"].astype(float)
    out = build_rolling_windows(df)
    sid = next(iter(out))
    # FEATURES list has rate_epinephrine at position 0, stay_id is col 0 → rate_epinephrine col 1
    epi_idx = 1 + FEATURES.index("rate_epinephrine")
    # last window: hours should span [los-47, los] but reversed
    last_window = out[sid][-1, :, epi_idx]
    # time index 0 = most recent hour = highest value
    assert last_window[0] > last_window[-1]
    assert last_window[0] == max(df[df["stay_id"] == sid]["hr"])


def test_discharge_windows_shapes(tiny_stay_df):
    df = _enrich_for_discharge(tiny_stay_df)
    data, outcomes = build_discharge_windows(df)
    for sid in data:
        # each window: (k, n, n_features+1); outcomes: (k, 1)
        assert data[sid].shape[1] == N_TIME_OFFSETS
        assert data[sid].shape[2] == len(FEATURES) + 1
        assert outcomes[sid].shape == (data[sid].shape[0], 1)


def test_disch_label_series_marks_last_n_hours():
    import pandas as pd

    df = pd.DataFrame({"stay_id": [1] * 60, "hr": list(range(60))})
    df["los"] = 59
    labels = disch_label_series(df, n=N_TIME_OFFSETS).tolist()
    # last 48 hours (hr 12..59) are 1
    assert labels[:12] == [0] * 12
    assert labels[12:] == [1] * 48


def test_disch_outcomes_align_with_rolling_windows(tiny_stay_df):
    """One discharge label per rolling window, for every stay."""
    df = _enrich_for_discharge(tiny_stay_df)
    windows = build_rolling_windows(df)
    outcomes = disch_outcomes_by_stay(df, n=N_TIME_OFFSETS)
    assert outcomes.keys() == windows.keys()
    for sid in windows:
        assert len(outcomes[sid]) == windows[sid].shape[0], f"mismatch for stay {sid}"


def test_disch_outcomes_align_when_hr_not_contiguous():
    """Regression: a stay with gaps in `hr` (or not starting at 0) must still
    yield exactly one label per rolling window — the old `hr >= n-1` filter did not."""
    import pandas as pd

    n = N_TIME_OFFSETS
    # 60 rows but hr starts at 10 and skips a couple of hours → non-contiguous.
    hrs = [h for h in range(10, 75) if h not in (20, 41)][:60]
    df = pd.DataFrame({"stay_id": [7] * 60, "hr": hrs})
    for col in FEATURES:
        df[col] = 1.0
    df["los"] = df["hr"].max()
    df["disch_48h"] = (df["hr"] >= df["los"] - (n - 1)).astype(int)

    windows = build_rolling_windows(df)
    outcomes = disch_outcomes_by_stay(df, n=n)
    assert outcomes.keys() == windows.keys()
    for sid in windows:
        assert len(outcomes[sid]) == windows[sid].shape[0]


def test_stay_mortality_outcome():
    import pandas as pd
    df = pd.DataFrame(
        {"stay_id": [1, 1, 2, 2], "hr": [0, 1, 0, 1], "icu_expire_flag": [1, 1, 0, 0]}
    )
    assert stay_to_mortality_outcome(df) == {1: 1, 2: 0}

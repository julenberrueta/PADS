"""Stay-level 3-way splits and 2-column label encoding."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from pads.data.schema import FEATURES, N_TIME_OFFSETS


def _can_stratify(labels: np.ndarray | None) -> bool:
    """True when every class has >=2 members (sklearn's stratify requirement)."""
    if labels is None:
        return False
    _, counts = np.unique(labels, return_counts=True)
    return bool(counts.min() >= 2)


def _three_way_indices(
    labels: np.ndarray | None, n: int, *, val_size: float, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 3-way split of `n` items -> (train, val, test) index arrays."""
    idx = np.arange(n)
    train_val_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=labels if _can_stratify(labels) else None,
    )
    # val fraction is relative to the remaining (train+val) pool.
    rel_val = val_size / (1.0 - test_size)
    sub = labels[train_val_idx] if labels is not None else None
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=rel_val,
        random_state=seed,
        stratify=sub if _can_stratify(sub) else None,
    )
    return train_idx, val_idx, test_idx


def split_stays(
    stay_ids: list[int],
    outcomes: list[int] | np.ndarray | None = None,
    *,
    groups: list | np.ndarray | None = None,
    val_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Stratified 3-way stay-level split -> (train, val, test).

    Splitting on stay_id (never on individual windows) keeps every window of a
    patient in the same fold, so there is no leakage across train/val/test.

    `outcomes` is the per-stay binary label aligned to `stay_ids`; when given and
    every class has >=2 members the split is stratified by it, otherwise it falls
    back to a plain random split. `val_size`/`test_size` are fractions of the
    whole; the default 0.1/0.2 yields 70/10/20.

    `groups` is a per-stay group id aligned to `stay_ids` (e.g. a hospital
    episode id). When given, the split is performed on whole groups so every
    stay sharing a group lands in the same fold — preventing leakage across
    stays of the same hospital episode. Stratification then uses the group's
    label (1 if any stay in the group is positive).
    """
    stay_ids = list(stay_ids)
    strat = np.asarray(outcomes) if outcomes is not None else None

    if groups is None:
        tr, va, te = _three_way_indices(
            strat, len(stay_ids), val_size=val_size, test_size=test_size, seed=seed,
        )
        pick = lambda ix: [stay_ids[i] for i in ix]
        return pick(tr), pick(va), pick(te)

    groups = np.asarray(groups)
    uniq = np.unique(groups)  # sorted -> deterministic
    grp_label = (
        np.array([strat[groups == g].max() for g in uniq]) if strat is not None else None
    )
    tr, va, te = _three_way_indices(
        grp_label, len(uniq), val_size=val_size, test_size=test_size, seed=seed,
    )
    sel = lambda ix: [s for s, g in zip(stay_ids, groups) if g in set(uniq[ix])]
    return sel(tr), sel(va), sel(te)


def build_xy_rolling(
    data: dict[int, np.ndarray], outcome: dict[int, int], stay_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """For mortality: one window per stay (the last one), one label.

    Fed most-recent-first (timestep 0 = newest hour) — the orientation the
    shipped base model was trained on. See windowing.build_rolling_windows.
    """
    keep = [s for s in stay_ids if s in data]
    X = np.empty((len(keep), N_TIME_OFFSETS, len(FEATURES)))
    y = np.empty((len(keep), 1))
    for i, sid in enumerate(keep):
        X[i] = data[sid][-1, :, 1:]
        y[i] = outcome[sid]
    return X, y


def build_xy_discharge(
    data: dict[int, np.ndarray], outcome: dict[int, np.ndarray], stay_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """For discharge: all anchor windows per stay (time axis as stored)."""
    keep = [s for s in stay_ids if s in data and s in outcome]
    n_rows = sum(data[s].shape[0] for s in keep)
    X = np.empty((n_rows, N_TIME_OFFSETS, len(FEATURES)))
    y = np.empty((n_rows, 1))
    i = 0
    for sid in keep:
        for s in range(data[sid].shape[0]):
            X[i] = data[sid][s, :, 1:]
            y[i] = outcome[sid][s]
            i += 1
    return X[:i], y[:i]


def encode_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero NaNs and one-hot the binary label into 2 columns (col0=neg, col1=pos)."""
    X = np.nan_to_num(X)
    y2 = np.zeros((len(y), 2))
    y2[:, 0] = (y[:, 0] == 0).astype(int)
    y2[:, 1] = (y[:, 0] == 1).astype(int)
    return X, y2

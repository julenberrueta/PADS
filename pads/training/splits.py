"""Train/val/test splits and 2-column label encoding."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from pads.data.schema import FEATURES, N_TIME_OFFSETS


def split_stays(stay_ids: list[int], *, test_size: float = 0.2, seed: int = 42):
    return train_test_split(stay_ids, test_size=test_size, random_state=seed)


def build_xy_rolling(
    data: dict[int, np.ndarray], outcome: dict[int, int], stay_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """For mortality: one window per stay (the last one, time-reversed), one label."""
    keep = [s for s in stay_ids if s in data]
    X = np.empty((len(keep), N_TIME_OFFSETS, len(FEATURES)))
    y = np.empty((len(keep), 1))
    for i, sid in enumerate(keep):
        X[i] = data[sid][-1, ::-1, 1:]
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


def prepare_train_val_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Half of `*_test` becomes val. NaNs zeroed. Labels one-hot in 2 cols.

    Returns (X_tr, y_tr_2c, X_te, y_te_2c, X_val, y_val_2c).
    """
    rng = np.random.default_rng(seed)
    n = X_test.shape[0]
    val_idx = rng.integers(0, n, size=n // 2)
    mask = np.zeros(n, dtype=bool)
    mask[val_idx] = True

    X_val = X_test[mask]
    y_val = y_test[mask]
    X_te = X_test[~mask]
    y_te = y_test[~mask]

    X_train = np.nan_to_num(X_train)
    X_val = np.nan_to_num(X_val)
    X_te = np.nan_to_num(X_te)

    def _to_2col(y):
        y2 = np.hstack([y, y])
        y2[:, 0] = (y[:, 0] == 0).astype(int)
        y2[:, 1] = (y[:, 0] == 1).astype(int)
        return y2

    return X_train, _to_2col(y_train), X_te, _to_2col(y_te), X_val, _to_2col(y_val)

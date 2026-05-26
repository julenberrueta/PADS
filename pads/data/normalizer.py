"""MinMax normalisation fitted on the first time slice of the windowed data."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def fit_and_save(X: np.ndarray, path: str | Path) -> MinMaxScaler:
    """Fit a MinMaxScaler on `X[:, 0, :]` and persist it. Returns the fitted scaler."""
    norm = MinMaxScaler().fit(X[:, 0, :])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(norm, f)
    return norm


def load(path: str | Path) -> MinMaxScaler:
    with open(path, "rb") as f:
        return pickle.load(f)


def transform(X: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Apply a fitted scaler to a 3-D windowed array (N, T, F)."""
    n, t, f = X.shape
    return scaler.transform(X.reshape(n * t, f)).reshape(n, t, f)


def fit_or_load_and_transform(
    X: np.ndarray, path: str | Path, *, load_existing: bool, save: bool = True
) -> np.ndarray:
    """Convenience wrapper used by training/inference."""
    if load_existing:
        scaler = load(path)
    else:
        scaler = MinMaxScaler().fit(X[:, 0, :])
        if save:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(scaler, f)
    return transform(X, scaler)

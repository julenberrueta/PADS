"""Training loop: callbacks, class weights, fit."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    TerminateOnNaN,
)

from pads import tracking

ModelKind = Literal["mortality", "discharge"]


class MLflowEpochLogger(Callback):
    """Stream per-epoch metrics to MLflow (no-op if MLflow disabled).

    Logs `{prefix}/{metric}` for every key in `logs` at the end of each epoch,
    using `step=epoch` so the MLflow UI shows them as a time series.
    """

    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if not logs or not tracking.enabled():
            return
        tracking.log_metrics(
            {f"{self.prefix}/{k}": float(v) for k, v in logs.items()},
            step=epoch,
        )


def class_weights(y_train_2c: np.ndarray, kind: ModelKind) -> dict[int, float]:
    """Compute per-class weights using the strategy from code_v3.

    - discharge: y_train.shape[0] / (2 * y_train[:, k].sum())
    - mortality: sklearn 'balanced' x (0.8, 1.5)  (matches code_v3 behaviour)
    """
    if kind == "discharge":
        n = y_train_2c.shape[0]
        return {
            0: n / (2 * y_train_2c[:, 0].sum()),
            1: n / (2 * y_train_2c[:, 1].sum()),
        }

    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_2c[:, 0]),
        y=y_train_2c[:, 0],
    )
    return {0: float(cw[0]) * 0.8, 1: float(cw[1]) * 1.5}


def fit(
    model: tf.keras.Model,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    kind: ModelKind,
    log_dir: str | Path,
    epochs: int = 1000,
    batch_size: int = 100,
    early_stopping_patience: int = 50,
    verbose: int = 2,
) -> tf.keras.callbacks.History:
    log_dir = Path(log_dir) / model_name
    log_dir.mkdir(parents=True, exist_ok=True)

    prefix = "mort" if kind == "mortality" else "disch"
    callbacks = [
        TensorBoard(log_dir=str(log_dir), write_graph=True),
        EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            mode="min",
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=str(log_dir / f"{model_name}_best.keras"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        # Stop immediately if the loss goes to NaN/Inf instead of running every
        # epoch and silently saving a model with NaN weights.
        TerminateOnNaN(),
        MLflowEpochLogger(prefix=prefix),
    ]

    history = model.fit(
        X_train,
        y_train,
        class_weight=class_weights(y_train, kind),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )
    assert_finite_weights(model, model_name)
    return history


def assert_finite_weights(model: tf.keras.Model, name: str) -> None:
    """Raise if any weight is NaN/Inf — i.e. training diverged.

    Guards against persisting a broken model. Divergence to NaN typically
    happens on very small or degenerate datasets (e.g. the synthetic smoke-test
    set) rather than with real cohorts; the message points the user there.
    """
    for w in model.weights:
        arr = w.numpy()
        if not np.isfinite(arr).all():
            n_bad = int((~np.isfinite(arr)).sum())
            raise ValueError(
                f"Training of '{name}' diverged: {n_bad}/{arr.size} non-finite values in "
                f"weight '{getattr(w, 'name', '?')}' (loss went to NaN/Inf; stopped by "
                f"TerminateOnNaN). This usually means the dataset is too small or "
                f"degenerate for this configuration (the synthetic dataset is for smoke "
                f"tests only). Try real/larger data, fewer epochs, a lower learning rate, "
                f"or a different --retrain_type."
            )


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu() -> list:
    """Enable memory growth on every visible GPU. Safe to call when no GPU is present."""
    import contextlib

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        # Ignore if memory growth was already configured on this device.
        with contextlib.suppress(RuntimeError):
            tf.config.experimental.set_memory_growth(gpu, True)
    return gpus

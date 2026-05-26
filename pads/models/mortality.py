"""Mortality LSTM model architecture and compilation."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.optimizers import Adam

from pads.models.layers import SoftmaxTemperature


def build_mortality_model(input_shape: tuple[int, ...]) -> tf.keras.Model:
    """Build (uncompiled) LSTM mortality model. `input_shape` is the full X shape (N, T, F)."""
    return tf.keras.Sequential(
        [
            layers.LSTM(100, input_shape=input_shape[1:]),
            layers.Dense(20, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(10, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(2, name="logits"),
            SoftmaxTemperature(temperature=1),
        ]
    )


def compile_mortality_model(model: tf.keras.Model, learning_rate: float = 1e-5) -> tf.keras.Model:
    model.compile(
        loss=BinaryFocalCrossentropy(apply_class_balancing=True),
        optimizer=Adam(learning_rate=learning_rate),
        weighted_metrics=[
            "accuracy",
            "AUC",
            "precision",
            "recall",
            tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
            tf.keras.metrics.F1Score(name="f1_score", average="weighted"),
        ],
    )
    return model

"""Load and save Keras models with PADS custom objects, plus retrain-strategy freezing."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import tensorflow as tf

from pads.models.discharge import build_discharge_model
from pads.models.layers import CUSTOM_OBJECTS
from pads.models.mortality import build_mortality_model

ModelKind = Literal["mortality", "discharge"]
RetrainType = Literal["full", "dense", "lstm", "scratch"]


def load_model(path: str | Path) -> tf.keras.Model:
    return tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)


def save_model(model: tf.keras.Model, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_or_create_model(
    retrain_type: RetrainType,
    kind: ModelKind,
    base_model_path: str | Path,
    input_shape: tuple[int, ...],
) -> tf.keras.Model:
    """Return a model ready for retraining.

    `retrain_type`:
      - "scratch": build a fresh model from scratch (base model is ignored)
      - "full":    load the base model with all layers trainable
      - "dense":   load and freeze the LSTM (layer 0)
      - "lstm":    load and freeze every layer except the LSTM (layer 0)
    """
    if retrain_type == "scratch":
        return (
            build_mortality_model(input_shape)
            if kind == "mortality"
            else build_discharge_model(input_shape)
        )

    model = load_model(base_model_path)
    if retrain_type == "dense":
        model.layers[0].trainable = False
    elif retrain_type == "lstm":
        for layer in model.layers[1:]:
            layer.trainable = False
    return model

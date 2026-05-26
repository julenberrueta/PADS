"""Custom Keras layers."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="pads", name="softmax_temperature")
def softmax_temperature(x: tf.Tensor, temperature: float = 1.0) -> tf.Tensor:
    """Temperature-scaled softmax along the last axis."""
    exp_x = tf.exp(x / temperature)
    return exp_x / tf.reduce_sum(exp_x, axis=-1, keepdims=True)


@tf.keras.utils.register_keras_serializable(package="pads", name="SoftmaxTemperature")
class SoftmaxTemperature(layers.Layer):
    """Softmax with a learnable-free temperature parameter."""

    def __init__(self, temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        exp_x = tf.exp(inputs / self.temperature)
        return exp_x / tf.reduce_sum(exp_x, axis=-1, keepdims=True)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({"temperature": self.temperature})
        return cfg


CUSTOM_OBJECTS = {
    "softmax_temperature": softmax_temperature,
    "SoftmaxTemperature": SoftmaxTemperature,
}

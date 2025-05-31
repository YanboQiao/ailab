"""
TensorFlow / Keras 版轻量 CNN
"""
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, models


def build_tf_model(
        channels: tuple[int, int] = (32, 64),
        fc_dim: int = 128,
        num_classes: int = 10
) -> tf.keras.Model:
    c1, c2 = channels
    inputs = layers.Input(shape=(28, 28, 1))
    x = inputs

    # --- 卷积块 1
    x = layers.Conv2D(c1, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(c1, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.25)(x)
    # --- 卷积块 2
    x = layers.Conv2D(c2, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(c2, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(fc_dim, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="CNN_TF")
    return model

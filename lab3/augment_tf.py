# augment_tf.py
"""
TensorFlow 数据增强 & Dataset 构建
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers


def get_augment_layer() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
    ], name="augmentation")


def get_datasets(
        data_dir: str | Path = "./data",
        batch_size: int = 128,
        val_split: float = 0.1
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    构建 train / val / test 三个 tf.data.Dataset
    """
    path = Path(data_dir) / "mnist.npz"
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
        path=path if path.exists() else None
    )

    x_train = x_train[..., None].astype("float32") / 255.0
    x_test = x_test[..., None].astype("float32") / 255.0

    val_size = int(len(x_train) * val_split)
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]

    aug = get_augment_layer()

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(buffer_size=10000) \
        .map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds

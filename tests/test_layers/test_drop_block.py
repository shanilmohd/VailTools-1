import numpy as np
from tensorflow import keras

from vailtools import layers


def test_drop_block_1d():
    model = keras.models.Sequential(
        [
            layers.DropBlock1D(rate=0.3, block_size=4),
            keras.layers.Conv1D(filters=16, kernel_size=3, padding="same"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16)), np.random.random((32, 16, 16)))
    model.save("tmp.ckpt")


def test_drop_block_2d():
    model = keras.models.Sequential(
        [
            layers.DropBlock2D(rate=0.3, block_size=4),
            keras.layers.Conv2D(filters=16, kernel_size=3, padding="same"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 16)))

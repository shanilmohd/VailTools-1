import numpy as np
from tensorflow import keras

from vailtools import layers


def test_coodinate_channel_1d():
    model = keras.models.Sequential([
        layers.CoordinateChannel1D(),
        keras.layers.Conv1D(filters=16, kernel_size=3, padding="same")
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((32, 16, 16)), np.random.random((32, 16, 16))
    )
    model.summary()


def test_coodinate_channel_2d():
    model = keras.models.Sequential([
        layers.CoordinateChannel2D(),
        keras.layers.Conv2D(filters=16, kernel_size=3, padding="same")
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 16))
    )
    model.summary()


def test_coodinate_channel_3d():
    model = keras.models.Sequential([
        layers.CoordinateChannel3D(),
        keras.layers.Conv3D(filters=16, kernel_size=3, padding="same")
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((32, 16, 16, 16, 16)), np.random.random((32, 16, 16, 16, 16))
    )
    model.summary()

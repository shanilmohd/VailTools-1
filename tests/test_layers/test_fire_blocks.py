import numpy as np
from tensorflow import keras

from vailtools import layers


def test_fire_1d():
    model = keras.models.Sequential([layers.Fire1D(filters=16)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16)), np.random.random((32, 16, 32)))
    model.summary()


def test_fire_2d():
    model = keras.models.Sequential([layers.Fire2D(filters=16)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 32)))
    model.summary()


def test_fire_3d():
    model = keras.models.Sequential([layers.Fire3D(filters=16)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((32, 16, 16, 16, 16)), np.random.random((32, 16, 16, 16, 32))
    )
    model.summary()

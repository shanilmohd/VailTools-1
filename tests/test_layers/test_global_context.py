import numpy as np
from tensorflow import keras

from vailtools import layers


def test_global_context_1d():
    model = keras.models.Sequential([layers.GlobalContext1D(filters=16)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16)), np.random.random((32, 16, 16)))
    model.summary()


def test_global_context_2d():
    model = keras.models.Sequential([layers.GlobalContext2D(filters=16)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 16)))
    model.summary()


def test_global_context_3d():
    model = keras.models.Sequential([layers.GlobalContext3D(filters=16)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((32, 16, 16, 16, 16)), np.random.random((32, 16, 16, 16, 16))
    )
    model.summary()

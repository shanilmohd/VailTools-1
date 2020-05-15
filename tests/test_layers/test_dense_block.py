import numpy as np
from tensorflow import keras

from vailtools import layers


def test_dense_1d():
    model = keras.models.Sequential([layers.Dense1D(filters=8, depth=2)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16)), np.random.random((4, 16, 16 + 8 * 2)))


def test_dense_2d():
    model = keras.models.Sequential([layers.Dense2D(filters=8, depth=2)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((4, 16, 16, 16)), np.random.random((4, 16, 16, 16 + 8 * 2))
    )


def test_dense_3d():
    model = keras.models.Sequential([layers.Dense3D(filters=8, depth=2)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((4, 16, 16, 16, 16)),
        np.random.random((4, 16, 16, 16, 16 + 8 * 2)),
    )

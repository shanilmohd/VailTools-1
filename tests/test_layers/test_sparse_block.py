import numpy as np
from tensorflow import keras

from vailtools import layers


def test_sparse_1d():
    model = keras.models.Sequential([layers.Sparse1D(filters=8)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16)), np.random.random((4, 16, 8)))


def test_sparse_2d():
    model = keras.models.Sequential([layers.Sparse2D(filters=8)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16, 16)), np.random.random((4, 16, 16, 8)))


def test_sparse_3d():
    model = keras.models.Sequential([layers.Sparse3D(filters=8)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((4, 16, 16, 16, 16)), np.random.random((4, 16, 16, 16, 8))
    )

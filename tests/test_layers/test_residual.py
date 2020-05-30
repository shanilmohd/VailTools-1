import numpy as np
from tensorflow import keras

from vailtools import layers


def test_residual_1d():
    model = keras.models.Sequential([layers.Residual1D(filters=4)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16)), np.random.random((4, 16, 4)))


def test_residual_2d():
    model = keras.models.Sequential([layers.Residual2D(filters=4)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16, 16)), np.random.random((4, 16, 16, 4)))


def test_residual_3d():
    model = keras.models.Sequential([layers.Residual3D(filters=4)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((4, 16, 16, 16, 16)), np.random.random((4, 16, 16, 16, 4))
    )


def test_residual_bottleneck_1d():
    model = keras.models.Sequential([layers.ResidualBottleneck1D(filters=4)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16)), np.random.random((4, 16, 4)))


def test_residual_bottleneck_2d():
    model = keras.models.Sequential([layers.ResidualBottleneck2D(filters=4)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16, 16)), np.random.random((4, 16, 16, 4)))


def test_residual_bottleneck_3d():
    model = keras.models.Sequential([layers.ResidualBottleneck3D(filters=4)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((4, 16, 16, 16, 16)), np.random.random((4, 16, 16, 16, 4))
    )

import numpy as np
from tensorflow import keras

from vailtools import layers


def test_dilation_block():
    model = keras.models.Sequential([layers.DilationBlock(filters=16)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16, 16)), np.random.random((4, 16, 16, 16)))


def test_residual_block():
    model = keras.models.Sequential([layers.ResidualBlock(filters=4)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16, 16)), np.random.random((4, 16, 16, 16 + 4)))


def test_residual_bottleneck_block():
    model = keras.models.Sequential([layers.ResidualBottleneckBlock(filters=4)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16, 16)), np.random.random((4, 16, 16, 16 + 4)))


def test_squeeze_excite_block():
    model = keras.models.Sequential([layers.SqueezeExciteBlock(width=16)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16, 16)), np.random.random((4, 16, 16, 16)))

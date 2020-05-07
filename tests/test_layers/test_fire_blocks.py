import numpy as np
from tensorflow import keras

from vailtools import layers


def test_fire_block_1d():
    model = keras.models.Sequential([layers.FireBlock1D(filters=16)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16)), np.random.random((32, 16, 32)))
    model.summary()


def test_fire_block_2d():
    model = keras.models.Sequential([layers.FireBlock2D(filters=16)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 32)))
    model.summary()


def test_fire_block_3d():
    model = keras.models.Sequential([layers.FireBlock3D(filters=16)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16, 16, 16)), np.random.random((32, 16, 16, 16, 32)))
    model.summary()

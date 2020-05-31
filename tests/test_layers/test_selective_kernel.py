import numpy as np
from tensorflow import keras

from vailtools import layers


def test_selective_kernel_1d():
    model = keras.models.Sequential([layers.SelectiveKernel1D(filters=8)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 8, 8)), np.random.random((4, 8, 8)))


def test_selective_kernel_2d():
    model = keras.models.Sequential([layers.SelectiveKernel2D(filters=8)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 8, 8, 8)), np.random.random((4, 8, 8, 8)))


def test_selective_kernel_3d():
    model = keras.models.Sequential([layers.SelectiveKernel3D(filters=8)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 8, 8, 8, 8)), np.random.random((4, 8, 8, 8, 8)))

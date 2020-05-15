import numpy as np
from tensorflow import keras

import vailtools.layers


def test_fractal_1d():
    model = keras.models.Sequential(
        [vailtools.layers.fractal_block.Fractal1D(filters=8, depth=2)]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16)), np.random.random((4, 16, 8 * 3)))


def test_fractal_2d():
    model = keras.models.Sequential(
        [vailtools.layers.fractal_block.Fractal2D(filters=8, depth=2)]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16, 16)), np.random.random((4, 16, 16, 8 * 3)))


def test_fractal_3d():
    model = keras.models.Sequential(
        [vailtools.layers.fractal_block.Fractal3D(filters=8, depth=2)]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((4, 16, 16, 16, 16)), np.random.random((4, 16, 16, 16, 8 * 3)),
    )

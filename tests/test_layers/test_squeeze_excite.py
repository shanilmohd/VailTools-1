import numpy as np
from tensorflow import keras

import vailtools.layers


def test_squeeze_excite_1d():
    model = keras.models.Sequential(
        [vailtools.layers.squeeze_excite.SqueezeExcite1D(width=16)]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16)), np.random.random((4, 16, 16)))


def test_squeeze_excite_2d():
    model = keras.models.Sequential(
        [vailtools.layers.squeeze_excite.SqueezeExcite2D(width=16)]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16, 16)), np.random.random((4, 16, 16, 16)))


def test_squeeze_excite_3d():
    model = keras.models.Sequential(
        [vailtools.layers.squeeze_excite.SqueezeExcite3D(width=16)]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((4, 16, 16, 16, 16)), np.random.random((4, 16, 16, 16, 16))
    )

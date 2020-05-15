import numpy as np
from tensorflow import keras

from vailtools import activations


def test_mish_dense():
    model = keras.models.Sequential([keras.layers.Dense(10, activation="mish")])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16)), np.random.random((32, 10)))


def test_swish_dense():
    model = keras.models.Sequential([keras.layers.Dense(10), activations.Swish()])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16)), np.random.random((32, 10)))


def test_swish_conv_1d():
    model = keras.models.Sequential(
        [keras.layers.Conv1D(10, 3, padding="same"), activations.Swish()]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16)), np.random.random((32, 16, 10)))


def test_swish_conv_2d():
    model = keras.models.Sequential(
        [keras.layers.Conv2D(10, 3, padding="same"), activations.Swish()]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 10)))


def test_swish_conv_3d():
    model = keras.models.Sequential(
        [keras.layers.Conv3D(10, 3, padding="same"), activations.Swish()]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((32, 16, 16, 16, 16)), np.random.random((32, 16, 16, 16, 10))
    )


def test_mish_rnn():
    model = keras.models.Sequential(
        [keras.layers.SimpleRNN(10, return_sequences=True), activations.Mish()]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16)), np.random.random((32, 16, 10)))

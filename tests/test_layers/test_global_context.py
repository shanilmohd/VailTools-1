import numpy as np
from tensorflow import keras

from vailtools import layers


def test_global_context_1d():
    model = keras.models.Sequential([layers.GlobalContext1D(filters=16)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16)), np.random.random((32, 16, 16)))


def test_global_context_1d_mismatch():
    model = keras.models.Sequential([layers.GlobalContext1D(filters=16, project_inputs=True)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 20, 20)), np.random.random((32, 20, 16)))


def test_global_context_2d():
    model = keras.models.Sequential([layers.GlobalContext2D(filters=16)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 16)))


def test_global_context_2d_mismatch():
    model = keras.models.Sequential([layers.GlobalContext2D(filters=16, project_inputs=True)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 20, 40, 20)), np.random.random((32, 20, 40, 16)))


def test_global_context_3d():
    model = keras.models.Sequential([layers.GlobalContext3D(filters=16)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((32, 16, 16, 16, 16)), np.random.random((32, 16, 16, 16, 16))
    )


def test_global_context_3d_mismatch():
    model = keras.models.Sequential([layers.GlobalContext3D(filters=16, project_inputs=True)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((32, 20, 10, 30, 5)), np.random.random((32, 20, 10, 30, 16))
    )
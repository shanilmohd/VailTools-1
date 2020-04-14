import numpy as np
from tensorflow import keras

from vailtools import layers


def test_dense_block():
    model = keras.models.Sequential([layers.DenseBlock(filters=32, depth=2)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 16 + 32 * 2))
    )
    model.summary()


def test_dilation_block():
    model = keras.models.Sequential([layers.DilationBlock(filters=16)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 16)))
    model.summary()


def test_fractal_block():
    model = keras.models.Sequential([layers.FractalBlock(filters=32, depth=2)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 32 * 3))
    )
    model.summary()


def test_fire_block():
    model = keras.models.Sequential([layers.FireBlock(filters=16)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 32)))
    model.summary()


def test_global_context_block():
    model = keras.models.Sequential([layers.GlobalContextBlock(filters=16)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 16)))
    model.summary()


def test_residual_block():
    model = keras.models.Sequential([layers.ResidualBlock(filters=32)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 16 + 32))
    )
    model.summary()


def test_residual_bottleneck_block():
    model = keras.models.Sequential([layers.ResidualBottleneckBlock(filters=32)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 16 + 32))
    )
    model.summary()


def test_sparse_block():
    model = keras.models.Sequential([layers.SparseBlock(filters=32)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 32)))
    model.summary()


def test_squeeze_excite_block():
    model = keras.models.Sequential([layers.SqueezeExciteBlock(width=16)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 16)))
    model.summary()

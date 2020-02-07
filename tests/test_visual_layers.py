import numpy as np
from tensorflow import keras

from vailtools.layers import ResidualBlock, ResidualBottleneckBlock, DenseBlock, FractalBlock, DilationBlock


def test_residual_block():
    model = keras.models.Sequential([ResidualBlock(filters=32)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 16 + 32)))
    model.summary()


def test_residual_bottleneck_block():
    model = keras.models.Sequential([ResidualBottleneckBlock(filters=32)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 16 + 32)))
    model.summary()


def test_dense_block():
    model = keras.models.Sequential([DenseBlock(filters=32, depth=2)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 16 + 32 * 2)))
    model.summary()


def test_fractal_block():
    model = keras.models.Sequential([FractalBlock(filters=32, depth=2)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 32 * 3)))
    model.summary()


def test_dilation_block():
    model = keras.models.Sequential([DilationBlock(filters=16)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(np.random.random((32, 16, 16, 16)), np.random.random((32, 16, 16, 16)))
    model.summary()

import numpy as np

from tensorflow import keras
from vailtools.layers import SnailAttentionBlock, SnailDenseBlock, SnailTCBlock


def test_snail_attention():
    model = keras.models.Sequential([SnailAttentionBlock(key_size=32, value_size=32)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16)), np.random.random((32, 16, 16 + 32)))
    model.summary()


def test_snail_dense():
    model = keras.models.Sequential([SnailDenseBlock(filters=32, dilation_rate=1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16)), np.random.random((32, 16, 16 + 32)))
    model.summary()


def test_snail_dense_chain():
    model = keras.models.Sequential(
        [
            SnailDenseBlock(filters=16, dilation_rate=1),
            SnailDenseBlock(filters=16, dilation_rate=2),
            SnailDenseBlock(filters=16, dilation_rate=4),
            SnailDenseBlock(filters=16, dilation_rate=8),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16)), np.random.random((32, 16, 16 + 16 * 4)))
    model.summary()


def test_snail_tc():
    model = keras.models.Sequential([SnailTCBlock(sequence_length=16, filters=16)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((32, 16, 16)), np.random.random((32, 16, 16 + 16 * 4)))
    model.summary()

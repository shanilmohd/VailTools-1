import numpy as np
from tensorflow import keras

from vailtools.layers import SnailAttentionBlock, SnailDenseBlock, SnailTCBlock


def test_snail_attention():
    model = keras.models.Sequential([SnailAttentionBlock(key_size=4, value_size=4)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16)), np.random.random((4, 16, 16 + 4)))


def test_snail_dense():
    model = keras.models.Sequential([SnailDenseBlock(filters=4, dilation_rate=1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16)), np.random.random((4, 16, 16 + 4)))


def test_snail_dense_chain():
    model = keras.models.Sequential(
        [
            SnailDenseBlock(filters=16, dilation_rate=1),
            SnailDenseBlock(filters=16, dilation_rate=2),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16)), np.random.random((4, 16, 16 + 16 * 2)))


def test_snail_tc():
    model = keras.models.Sequential([SnailTCBlock(sequence_length=16, filters=16)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16)), np.random.random((4, 16, 16 + 16 * 4)))

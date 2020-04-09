import numpy as np
from tensorflow import keras
from vailtools.layers import WaveNetBlock


def test_wave_net_block():
    input_ = keras.layers.Input((None, 16))
    skip_out, residual = WaveNetBlock(filters=32)(input_)
    m1 = keras.models.Model(input_, (skip_out, residual))
    m1.compile(optimizer="adam", loss="mse")
    m1.fit(
        np.random.random((32, 16, 16)),
        (np.random.random((32, 16, 32)), np.random.random((32, 16, 32))),
    )
    m1.summary()

import numpy as np
from tensorflow import keras

from vailtools.layers import WaveNetBlock


def test_wave_net_block():
    m1 = keras.models.Sequential([WaveNetBlock(filters=32)])
    m1.compile(optimizer='adam', loss='mse')
    m1.fit(np.random.random((32, 16, 16)), np.random.random((32, 16, 16 + 32)))
    m1.summary()

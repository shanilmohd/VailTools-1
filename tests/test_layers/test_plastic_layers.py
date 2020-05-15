import numpy as np
from tensorflow import keras

from vailtools.layers import NMPlasticRNN, PlasticGRU, PlasticLSTM, PlasticRNN


def test_plastic_rnn():
    model = keras.models.Sequential(
        [PlasticRNN(units=16, batch_input_shape=(4, 16, 16))]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16)), np.random.random((4, 16)))
    model.summary()


def test_plastic_gru():
    model = keras.models.Sequential(
        [PlasticGRU(units=16, batch_input_shape=(4, 16, 16))]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16)), np.random.random((4, 16)))
    model.summary()


def test_plastic_lstm():
    model = keras.models.Sequential(
        [PlasticLSTM(units=16, batch_input_shape=(4, 16, 16))]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16)), np.random.random((4, 16)))
    model.summary()


def test_nm_plastic_rnn():
    model = keras.models.Sequential(
        [NMPlasticRNN(units=16, batch_input_shape=(4, 16, 16))]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16)), np.random.random((4, 16)))
    model.summary()

import numpy as np
from tensorflow import keras

from vailtools.layers import WaveNetBlock


def test_wave_net_block():
    input_ = keras.layers.Input((None, 16))
    skip_out, residual = WaveNetBlock(filters=4)(input_)
    model = keras.models.Model(input_, (skip_out, residual))
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((4, 16, 16)),
        (np.random.random((4, 16, 4)), np.random.random((4, 16, 4))),
    )
    model.save("tmp.ckpt")
    del model
    model = keras.models.load_model("tmp.ckpt")
    model.predict(np.random.random((4, 16, 16)))

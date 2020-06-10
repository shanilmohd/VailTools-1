import numpy as np
from tensorflow import keras

from vailtools import layers


def test_coodinate_channel_1d():
    model = keras.models.Sequential(
        [
            layers.CoordinateChannel1D(input_shape=(None, 16)),
            keras.layers.Conv1D(filters=16, kernel_size=3, padding="same"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16)), np.random.random((4, 16, 16)))
    model.save("tmp.ckpt")
    model = keras.models.load_model("tmp.ckpt")
    model.predict(np.random.random((4, 16, 16)))


def test_coodinate_channel_1d_functional():
    ins = keras.layers.Input((None, 16))
    pred = layers.CoordinateChannel1D()(ins)
    pred = keras.layers.Conv1D(filters=16, kernel_size=3, padding="same")(pred)
    model = keras.models.Model(ins, pred)

    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16)), np.random.random((4, 16, 16)))
    model.save("tmp.ckpt")
    model = keras.models.load_model("tmp.ckpt")
    model.predict(np.random.random((4, 16, 16)))


def test_coodinate_channel_2d():
    model = keras.models.Sequential(
        [
            layers.CoordinateChannel2D(input_shape=(None, 16, 16)),
            keras.layers.Conv2D(filters=16, kernel_size=3, padding="same"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((4, 16, 16, 16)), np.random.random((4, 16, 16, 16)))


def test_coodinate_channel_3d():
    model = keras.models.Sequential(
        [
            layers.CoordinateChannel3D(input_shape=(None, 16, 16, 16)),
            keras.layers.Conv3D(filters=16, kernel_size=3, padding="same"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        np.random.random((4, 16, 16, 16, 16)), np.random.random((4, 16, 16, 16, 16))
    )

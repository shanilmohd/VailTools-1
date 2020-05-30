import numpy as np

from vailtools import networks

train_x = np.random.random(size=(4, 32, 32, 1))


def test_restrict_net():
    model = networks.restrict_net(
        final_activation="sigmoid", output_channels=1, input_shape=(32, 32, 1),
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(train_x, train_x, epochs=1)


def test_u_net():
    model = networks.u_net(
        final_activation="sigmoid", output_channels=1, input_shape=(32, 32, 1),
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(train_x, train_x, epochs=1)


def test_res_u_net():
    model = networks.res_u_net(
        final_activation="sigmoid", output_channels=1, input_shape=(32, 32, 1),
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(train_x, train_x, epochs=1)

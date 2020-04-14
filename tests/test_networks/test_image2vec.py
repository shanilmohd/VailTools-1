import numpy as np

from vailtools import networks

train_x = np.random.random(size=(32, 28, 28, 1))
train_y = np.random.randint(10, size=32)


def test_res_net():
    model = networks.res_net(
        final_activation="sigmoid", num_classes=10, input_shape=(28, 28, 1),
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.summary()
    model.fit(train_x, train_y, epochs=1)

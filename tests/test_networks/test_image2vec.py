from tensorflow.keras.datasets import mnist

from vailtools import networks

(train_x, train_y), _ = mnist.load_data()
train_x = train_x[..., None].astype(float)

train_x, train_y = train_x[:2048], train_y[:2048]


def test_res_net():
    model = networks.res_net(
        final_activation='sigmoid',
        num_classes=10,
        input_shape=(28, 28, 1),
    )
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.summary()
    model.fit(train_x, train_y, epochs=3)

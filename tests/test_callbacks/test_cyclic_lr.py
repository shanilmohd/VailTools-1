from tensorflow import keras
from tensorflow.keras.datasets import mnist

from vailtools.callbacks import CyclicLRScheduler

(train_x, train_y), _ = mnist.load_data()
train_x = train_x[..., None].astype(float)
train_x, train_y = train_x[:2048], train_y[:2048]


def test_cyclic_lr_scheduler():
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv2D(10, kernel_size=3, padding='same', activation='relu'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Activation('softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.fit(
        train_x,
        keras.utils.to_categorical(train_y),
        epochs=5,
        callbacks=[CyclicLRScheduler(total_steps=5, cycles=1)],
    )

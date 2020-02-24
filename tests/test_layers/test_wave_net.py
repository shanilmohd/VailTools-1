import numpy as np
from tensorflow import keras

from vailtools.layers import WaveNetBlock
from vailtools.networks.seq2seq import wave_net


def test_wave_net_block():
    m1 = keras.models.Sequential([WaveNetBlock(filters=32)])
    m1.compile(optimizer='adam', loss='mse')
    m1.fit(np.random.random((32, 16, 16)), np.random.random((32, 16, 16 + 32)))
    m1.summary()


def test_wave_net_integration(max_features=2048, maxlen=80, batch_size=32):
    model = wave_net(
        embedding_input_dim=max_features,
        final_activation='sigmoid',
        input_shape=(maxlen,),
        flatten_output=True,
        output_channels=1,
    )
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    x_train = np.random.randint(0, max_features, (batch_size, maxlen))
    y_train = np.random.randint(0, 2, size=(batch_size, 1))

    model.fit(x_train, y_train, batch_size=batch_size)
    model.summary()

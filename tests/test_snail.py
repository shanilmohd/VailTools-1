import keras
import numpy as np

from vailtools.layers import SnailAttentionBlock, SnailDenseBlock, SnailTCBlock
from vailtools.networks.seq2seq import snail_control


def test_snail_attention():
    m1 = keras.models.Sequential([SnailAttentionBlock(key_size=32, value_size=32)])
    m1.compile(optimizer='adam', loss='mse')
    m1.fit(np.random.random((32, 16, 16)), np.random.random((32, 16, 16 + 32)))
    m1.summary()


def test_snail_dense():
    m2 = keras.models.Sequential([SnailDenseBlock(filters=32, dilation_rate=1)])
    m2.compile(optimizer='adam', loss='mse')
    m2.fit(np.random.random((32, 16, 16)), np.random.random((32, 16, 16 + 32)))
    m2.summary()


def test_snail_dense_chain():
    m3 = keras.models.Sequential([
        SnailDenseBlock(filters=16, dilation_rate=1),
        SnailDenseBlock(filters=16, dilation_rate=2),
        SnailDenseBlock(filters=16, dilation_rate=4),
        SnailDenseBlock(filters=16, dilation_rate=8),
    ])
    m3.compile(optimizer='adam', loss='mse')
    m3.fit(np.random.random((32, 16, 16)), np.random.random((32, 16, 16 + 16 * 4)))
    m3.summary()


def test_snail_tc():
    m4 = keras.models.Sequential([SnailTCBlock(sequence_length=16, filters=16)])
    m4.compile(optimizer='adam', loss='mse')
    m4.fit(np.random.random((32, 16, 16)), np.random.random((32, 16, 16 + 16 * 4)))
    m4.summary()


def test_snail_integration(max_features=20000, maxlen=80, batch_size=32):
    model = snail_control(
        embedding_input_dim=max_features,
        final_activation='sigmoid',
        input_shape=(maxlen,),
        output_size=1,
    )
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    x_train = np.random.randint(0, max_features, (batch_size, maxlen))
    y_train = np.random.randint(0, 2, size=(batch_size, 1))

    model.fit(x_train, y_train, batch_size=batch_size)
    model.summary()

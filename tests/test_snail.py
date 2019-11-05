import keras
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence

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
    # cut texts after this number of words (among top max_features most common words)
    print('Loading data...')
    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    np.load = np_load_old
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    model = snail_control(
        embedding_input_dim=max_features,
        final_activation='sigmoid',
        input_shape=(maxlen,),
        output_size=1,
    )
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    print('Train...')
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=5,
        validation_data=(x_test, y_test),
    )
    score, acc = model.evaluate(
        x_test,
        y_test,
        batch_size=batch_size,
    )
    print('Test score:', score)
    print('Test accuracy:', acc)

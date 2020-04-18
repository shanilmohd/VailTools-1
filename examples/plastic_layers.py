"""
Adapted from https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
Intended to provide a small, self-contained test case for ensuring that the tensorflow
implementation of plastic layers is functioning.
Note that this only ensures that the computation graphs are correctly constructed,
allowing the layers to be used more widely for comprehensive testing.
"""
import numpy as np
from tensorflow.keras.datasets import reuters
from tensorflow.keras.layers import Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential

from vailtools.layers import PlasticRNN


def main(max_words=1000, max_len=128, batch_size=32, epochs=5):
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words, maxlen=max_len, test_split=0.2)

    x_train = ragged_to_non_ragged(x_train, max_len=max_len)
    x_test = ragged_to_non_ragged(x_test, max_len=max_len)
    x_train = x_train[:-(len(x_train) % batch_size)]
    x_test = x_test[:-(len(x_test) % batch_size)]
    print(f"Input Train Shape: {x_train.shape}")
    print(f"Input Test Shape:  {x_test.shape}")

    y_train = y_train[:-(len(y_train) % batch_size)]
    y_test = y_test[:-(len(y_test) % batch_size)]
    print(f"Output Train Shape: {y_train.shape}")
    print(f"Output Test Shape:  {y_test.shape}")

    num_classes = np.max(y_train) + 1
    print(num_classes, 'classes')

    print('Building model...')
    model = Sequential([
        Embedding(
            input_dim=max_words + 1,
            output_dim=64,
            input_length=max_len,
            batch_input_shape=(batch_size, max_len),  # Must specify full batch shape for plastic layers
        ),
        PlasticRNN(128),
        Dropout(0.5),
        Dense(
            num_classes,
            activation='softmax',
        ),
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.10989,  # Plastic layers require a fixed batch size.
    )
    results = model.evaluate(
        x_test,
        y_test,
        batch_size=batch_size,
        verbose=1,
    )
    print('Test loss:', results[0])
    print('Test accuracy:', results[1])


def ragged_to_non_ragged(ragged, max_len=None, pad_side="left", pad_value=0, dtype=int):
    assert pad_side in {"left", "right"}

    if max_len is None:
        max_len = np.max([len(x) for x in ragged])

    non_ragged = np.zeros((len(ragged), max_len), dtype=dtype)
    non_ragged[:] = pad_value
    for i, row in enumerate(ragged):
        if len(row) > max_len:
            row = row[:max_len]

        if pad_side == "left":
            non_ragged[i, -len(row):] = np.array(row)
        else:
            non_ragged[i, :len(row)] = np.array(row)

    return non_ragged


if __name__ == '__main__':
    main()

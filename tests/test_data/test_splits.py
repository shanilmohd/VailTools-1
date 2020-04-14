from tensorflow.keras.datasets import mnist

from vailtools.data import splits


def test_split_data():
    train, test = mnist.load_data()

    (train, val, test) = splits.split_data(*train)

    assert len(train.x) > len(val.x)
    assert len(train.x) > len(test.x)
    assert len(val.x) == len(test.x)

    assert len(train.x) == len(train.y)
    assert len(val.x) == len(val.y)
    assert len(test.x) == len(test.y)

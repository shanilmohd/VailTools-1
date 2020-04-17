import numpy as np
import tensorflow as tf

from vailtools.data.generate import pickup_sticks_image
from vailtools.losses import lovasz
from vailtools.utils import to_categorical


def test_lovasz_hinge():
    _, labels = pickup_sticks_image()  # (256, 256, 1)
    sparse_labels = labels[np.newaxis]  # (1, 256, 256, 1)
    sparse_binary_labels = sparse_labels > 0
    sparse_binary_labels = tf.convert_to_tensor(sparse_binary_labels > 0, tf.float32)
    loss = lovasz.lovasz_hinge(sparse_binary_labels, sparse_binary_labels, ignore=0)
    assert loss.numpy() == 0


def test_lovasz_softmax():
    _, labels = pickup_sticks_image()  # (256, 256, 1)
    sparse_labels = labels[np.newaxis]   # (1, 256, 256, 1)
    dense_labels = np.squeeze(to_categorical(sparse_labels), axis=-2)   # (1, 256, 256, classes)
    sparse_labels = tf.convert_to_tensor(sparse_labels, tf.float32)
    dense_labels = tf.convert_to_tensor(dense_labels, tf.float32)
    loss = lovasz.lovasz_softmax(sparse_labels, dense_labels, ignore=0, classes="all")
    assert loss.numpy() == 0

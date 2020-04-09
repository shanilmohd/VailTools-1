import numpy as np

import tensorflow as tf
from vailtools.data.generate import pickup_sticks_image
from vailtools.losses import lovasz


def test_lovasz_hinge():
    _, labels = pickup_sticks_image()
    labels = labels[np.newaxis]
    labels = tf.convert_to_tensor(labels > 0, tf.float32)
    loss = lovasz.lovasz_hinge(labels, labels, ignore=0)
    assert loss.numpy() == 0


def test_lovasz_softmax():
    _, labels = pickup_sticks_image()
    labels = labels[np.newaxis]
    labels = tf.convert_to_tensor(labels, tf.float32)
    loss = lovasz.lovasz_hinge(labels, labels, ignore=0)
    assert loss.numpy() == 0

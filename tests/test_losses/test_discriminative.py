import numpy as np
import tensorflow as tf

from vailtools.data.generate import pickup_sticks_image
from vailtools.losses import discriminative
from vailtools.utils import to_categorical


def test_discriminative_full():
    # (256, 256, 1)
    _, labels = pickup_sticks_image()
    # (1, 256, 256, 1)
    sparse_labels = labels[np.newaxis]
    # (1, 256, 256, classes)
    dense_labels = np.squeeze(to_categorical(sparse_labels), axis=-2)

    sparse_labels = tf.convert_to_tensor(sparse_labels, tf.float32)
    dense_labels = tf.convert_to_tensor(dense_labels, tf.float32)

    loss = discriminative.discriminative_loss(sparse_labels, dense_labels)

    assert loss.numpy() >= 0


def test_discriminative_sample():
    # (256, 256, 1)
    _, labels = pickup_sticks_image()
    # (256, 256, classes)
    dense_labels = np.squeeze(to_categorical(labels), axis=-2)

    sparse_labels = tf.convert_to_tensor(labels, tf.float32)
    dense_labels = tf.convert_to_tensor(dense_labels, tf.float32)

    loss = discriminative.sample_loss((sparse_labels, dense_labels))

    assert loss.numpy() >= 0


def test_compute_centroids():
    # (256, 256, 1)
    _, labels = pickup_sticks_image()
    # (256, 256, classes)
    dense_labels = np.squeeze(to_categorical(labels), axis=-2)

    sparse_labels = tf.convert_to_tensor(labels, tf.float32)
    dense_labels = tf.convert_to_tensor(dense_labels, tf.float32)

    centroids = discriminative.compute_centroids(sparse_labels, dense_labels)

    assert np.all(np.logical_not(np.isnan(centroids.numpy())))


def test_var_loss():
    # (256, 256, 1)
    _, labels = pickup_sticks_image()
    # (256, 256, classes)
    dense_labels = np.squeeze(to_categorical(labels), axis=-2)

    sparse_labels = tf.convert_to_tensor(labels, tf.float32)
    dense_labels = tf.convert_to_tensor(dense_labels, tf.float32)

    centroids = discriminative.compute_centroids(sparse_labels, dense_labels)
    loss = discriminative.var_loss(sparse_labels, dense_labels, centroids)

    assert loss.numpy() == 0


def test_dist_loss():
    # (256, 256, 1)
    _, labels = pickup_sticks_image()
    # (256, 256, classes)
    dense_labels = np.squeeze(to_categorical(labels), axis=-2)

    sparse_labels = tf.convert_to_tensor(labels, tf.float32)
    dense_labels = tf.convert_to_tensor(dense_labels, tf.float32)

    centroids = discriminative.compute_centroids(sparse_labels, dense_labels)
    loss = discriminative.dist_loss(centroids)

    assert loss.numpy() >= 0


def test_reg_loss():
    # (256, 256, 1)
    _, labels = pickup_sticks_image()
    # (256, 256, classes)
    dense_labels = np.squeeze(to_categorical(labels), axis=-2)

    sparse_labels = tf.convert_to_tensor(labels, tf.float32)
    dense_labels = tf.convert_to_tensor(dense_labels, tf.float32)

    centroids = discriminative.compute_centroids(sparse_labels, dense_labels)
    loss = discriminative.reg_loss(centroids)

    assert loss.numpy() > 0

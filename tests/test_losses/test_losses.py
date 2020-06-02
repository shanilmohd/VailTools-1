import numpy as np
import tensorflow as tf

from vailtools import losses


def test_iou_loss_perfect():
    y = tf.convert_to_tensor(np.random.randint(2, size=(4, 10)), dtype=tf.float32)
    loss = losses.iou_loss(y, y).numpy()
    assert np.allclose(loss, np.zeros_like(loss))


def test_iou_loss_anti_perfect():
    y = np.random.randint(2, size=(4, 10))
    x = np.logical_not(y)
    loss = losses.iou_loss(
        tf.convert_to_tensor(y, dtype=tf.float32),
        tf.convert_to_tensor(x, dtype=tf.float32),
    ).numpy()
    assert np.allclose(loss, np.ones_like(loss))


def test_class_mean_iou_loss_perfect():
    y = np.random.randint(100, size=(4, 10, 10))
    y = y == np.max(y, axis=-1, keepdims=True)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    loss = losses.class_mean_iou_loss(y, y).numpy()
    assert np.allclose(loss, np.zeros_like(loss))


def test_class_mean_iou_loss_anti_perfect():
    y = np.random.randint(100, size=(4, 10, 10))
    y = y == np.max(y, axis=-1, keepdims=True)
    x = np.roll(y, 1, axis=-1)
    loss = losses.iou_loss(
        tf.convert_to_tensor(y, dtype=tf.float32),
        tf.convert_to_tensor(x, dtype=tf.float32),
    ).numpy()
    assert np.allclose(loss, np.ones_like(loss))


def test_dice_loss_perfect():
    y = tf.convert_to_tensor(np.random.randint(2, size=(4, 10)), dtype=tf.float32)
    loss = losses.dice_loss(y, y).numpy()
    assert np.allclose(loss, np.zeros_like(loss))


def test_dice_loss_anti_perfect():
    y = np.random.randint(2, size=(4, 10))
    x = np.logical_not(y)
    loss = losses.dice_loss(
        tf.convert_to_tensor(y, dtype=tf.float32),
        tf.convert_to_tensor(x, dtype=tf.float32),
    ).numpy()
    assert np.allclose(loss, np.ones_like(loss))


def test_ewc_loss_perfect():
    y = np.random.randint(100, size=(3, 5))
    y = tf.convert_to_tensor(y == np.max(y, axis=-1, keepdims=True), dtype=tf.float32)
    weights = np.random.random(size=(5, 5))
    l_func = losses.error_weighted_categorical_crossentropy(weights / weights.sum())
    loss = l_func(y, y).numpy()
    assert np.allclose(loss, np.zeros_like(loss))


def test_ewc_loss_anti_perfect():
    y = np.random.randint(100, size=(3, 5))
    y = y == np.max(y, axis=-1, keepdims=True)
    x = np.roll(y, 1, axis=-1)
    weights = np.random.random(size=(5, 5))
    l_func = losses.error_weighted_categorical_crossentropy(weights / weights.sum())
    loss = l_func(
        tf.convert_to_tensor(y, dtype=tf.float32),
        tf.convert_to_tensor(x, dtype=tf.float32),
    ).numpy()
    assert np.all(loss > 0)


def test_cwc_loss_perfect():
    y = np.random.randint(100, size=(3, 5))
    y = tf.convert_to_tensor(y == np.max(y, axis=-1, keepdims=True), dtype=tf.float32)
    weights = np.random.random(size=(5,))
    l_func = losses.class_weighted_categorical_crossentropy(weights / weights.sum())
    loss = l_func(y, y).numpy()
    assert np.abs(loss - np.zeros_like(loss)).max() < 1e-6


def test_cwc_loss_anti_perfect():
    y = np.random.randint(100, size=(3, 5))
    y = y == np.max(y, axis=-1, keepdims=True)
    x = np.roll(y, 1, axis=-1)
    weights = np.random.random(size=(5,))
    l_func = losses.class_weighted_categorical_crossentropy(weights / weights.sum())
    loss = l_func(
        tf.convert_to_tensor(y, dtype=tf.float32),
        tf.convert_to_tensor(x, dtype=tf.float32),
    ).numpy()
    assert np.all(loss > 0)


def test_mixed_loss_perfect():
    y = np.random.random(size=(3, 5))
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    l_func = losses.mixed_l1_l2_loss()
    loss = l_func(y, y).numpy()
    assert np.allclose(loss, np.zeros_like(loss))


def test_mixed_loss_anti_perfect():
    y = np.random.random(size=(3, 5))
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    x = np.random.random(size=(3, 5))
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    l_func = losses.mixed_l1_l2_loss()
    loss = l_func(y, x,).numpy()
    assert np.all(loss > 0)


def test_focal_loss_perfect():
    y = np.random.randint(100, size=(4, 10, 10))
    y = y == np.max(y, axis=-1, keepdims=True)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    loss = losses.focal_loss()(y, y).numpy()
    assert np.allclose(loss, np.zeros_like(loss))


def test_focal_loss_anti_perfect():
    y = np.random.randint(100, size=(4, 10, 10))
    y = y == np.max(y, axis=-1, keepdims=True)
    x = np.roll(y, 1, axis=-1)
    loss = losses.focal_loss()(
        tf.convert_to_tensor(y, dtype=tf.float32),
        tf.convert_to_tensor(x, dtype=tf.float32),
    ).numpy()
    assert np.all(loss > 0)

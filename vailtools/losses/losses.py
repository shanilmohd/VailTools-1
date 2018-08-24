"""
Provides loss functions suitable for training Keras models.
"""

from itertools import product

import keras.backend as K
import tensorflow as tf
from keras.losses import mse, mae

from .. import metrics


def iou_loss(y_true, y_pred):
    """
    Args:
        y_true: (keras.backend.Tensor) Target values.
        y_pred: (keras.backend.Tensor) Predicted values.

    Returns: (keras.backend.Tensor)
        IoU loss for each sample.
    """
    return 1. - metrics.iou_score(y_true, y_pred)


def class_mean_iou_loss(y_true, y_pred):
    """
    Args:
        y_true: (keras.backend.Tensor) Target values.
        y_pred: (keras.backend.Tensor) Predicted values.

    Returns: (keras.backend.Tensor)
        Mean IoU loss over classes for each sample.
    """
    return 1. - metrics.class_mean_iou_score(y_true, y_pred)


def dice_loss(y_true, y_pred):
    """
    Args:
        y_true: (keras.backend.Tensor) Target values.
        y_pred: (keras.backend.Tensor) Predicted values.

    Returns: (keras.backend.Tensor)
        Dice loss for each sample.
    """
    return 1. - metrics.dice_score(y_true, y_pred)


def error_weighted_categorical_crossentropy(weights):
    """
    Adapted from https://github.com/keras-team/keras/issues/6218

    Args:
        weights: (numpy.ndarray) Should have shape (C, C), where C is the number of classes.
    """
    def wcc(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())  # Clip for numerical stability

        final_mask = K.zeros_like(y_pred[:, 0])
        y_pred_max = K.max(y_pred, axis=-1, keepdims=True)
        y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
        for c_p, c_t in product(range(len(weights)), repeat=2):
            final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        return K.categorical_crossentropy(y_pred, y_true) * final_mask
    return wcc


def class_weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Args:
        weights: (numpy.ndarray) Should have shape (C,), where C is the number of classes.
    """
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())  # Clip for numerical stability
        return -K.sum(y_true * K.log(y_pred) * weights, axis=-1)
    return loss


def focal_loss(gamma=2., alpha=0.25):
    """
    Adapted from https://github.com/Atomwh/FocalLoss_Keras/blob/master/focalloss.py

    Args:
        gamma: (float) Reduces the loss from easy samples, allowing emphasis to be placed on hard samples.
        alpha: (float) Controls the relative contribution of the positive and negative classes.
    """
    def focal_loss_(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())  # Clip for numerical stability
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return (
            - K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))
            - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
        )
    return focal_loss_


def mixed_l1_l2_loss(alpha=0.5):
    """
    A simple mixing of the standard L1 and L2 losses.

    Args:
        alpha: (float) Controls the mixing of the losses.

    Returns: (tensorflow.Tensor)
        Prediction loss/error.
    """
    def loss_(y_true, y_pred):
        return alpha * mse(y_true, y_pred) + (1 - alpha) * mae(y_true, y_pred)
    return loss_

"""
Provides metrics suitable for monitoring and evaluating Keras models.
"""
import tensorflow.keras.backend as K


def iou_score(y_true, y_pred):
    """
    Intersection over Union score, also known as the Jaccard index.

    Args:
        y_true: (keras.backend.Tensor) Target values.
        y_pred: (keras.backend.Tensor) Predicted values.

    Returns: (keras.backend.Tensor)
        IoU score for each sample.
    """
    intersection = K.sum(K.batch_flatten(y_true * y_pred), axis=-1)
    union = K.sum(K.batch_flatten(y_true + y_pred), axis=-1) - intersection
    return (intersection + K.epsilon()) / (union + K.epsilon())


def class_mean_iou_score(y_true, y_pred):
    """
    Average of IoU scores for each class.

    Args:
        y_true: (keras.backend.Tensor) Target values.
        y_pred: (keras.backend.Tensor) Predicted values.

    Returns: keras.backend.Tensor
    """
    batch, *spatial, channels = K.shape(y_true)
    new_shape = (batch, K.prod(spatial), channels)
    y_true, y_pred = K.reshape(y_true, new_shape), K.reshape(y_pred, new_shape)
    intersection = K.sum(y_true * y_pred, axis=-2)
    union = K.sum(y_true + y_pred, axis=-2) - intersection
    return K.mean((intersection + K.epsilon()) / (union + K.epsilon()), axis=-1)


def dice_score(y_true, y_pred):
    """
    A.K.A. the F1 score.

    Args:
        y_true: (keras.backend.Tensor) Target values.
        y_pred: (keras.backend.Tensor) Predicted values.

    Returns: keras.backend.Tensor
    """
    numerator = 2.0 * K.sum(K.batch_flatten(y_true * y_pred), axis=-1)
    denominator = K.sum(K.batch_flatten(y_true + y_pred), axis=-1)
    return (numerator + K.epsilon()) / (denominator + K.epsilon())

"""
Provides metrics suitable for monitoring and evaluating Keras models.
"""
import keras.backend as K


def iou_score(y_true, y_pred):
    """
    Intersection over Union score, also known as the Jaccard index,
    implemented using the Keras backend.

    This should be benchmarked against the dice score using a third metric to
    see whether the use of the maximum operation harms speed or quality.

    Args:
        y_true: (keras.backend.Tensor) Target values.
        y_pred: (keras.backend.Tensor) Predicted values.

    Returns: (keras.backend.Tensor)
        IoU score for each sample.
    """
    intersection = K.sum(K.batch_flatten(y_true * y_pred), axis=-1)
    union = K.sum(K.batch_flatten(y_true + y_pred), axis=-1) - intersection
    return intersection / union


def class_mean_iou_score(y_true, y_pred):
    """
    Intersection over Union score, also known as the Jaccard index,
    implemented using the Keras backend.

    This should be benchmarked against the dice score using a third metric to
    see whether the use of the maximum operation harms speed or quality.

    Args:
        y_true: (keras.backend.Tensor) Target values.
        y_pred: (keras.backend.Tensor) Predicted values.

    Returns: (keras.backend.Tensor)
        Mean IoU score over classes for each sample.
    """
    old_shape = K.shape(y_true)
    new_shape = (old_shape[0], K.prod(old_shape[1:-1]), old_shape[-1])
    y_true, y_pred = K.reshape(y_true, new_shape), K.reshape(y_pred, new_shape)
    # Compute IoU for each class
    intersection = K.sum(y_true * y_pred, axis=-2)
    union = K.sum(K.maximum(y_true, y_pred), axis=-2)
    return K.mean(intersection / union, axis=-1)


def dice_score(y_true, y_pred):
    """
    Dice score, also known as the F1 score, implemented using the Keras backend.

    Args:
        y_true: (keras.backend.Tensor) Target values.
        y_pred: (keras.backend.Tensor) Predicted values.

    Returns: (keras.backend.Tensor)
        Dice score for each sample.
    """
    numerator = 2. * K.sum(K.batch_flatten(y_true * y_pred), axis=-1)
    denominator = K.sum(K.batch_flatten(y_true + y_pred), axis=-1)
    return numerator / denominator

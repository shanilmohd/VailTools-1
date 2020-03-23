"""
Provides metrics for evaluating artificial neural networks.
Note that all functions operate on and return numpy.ndarrays.
"""

import numpy as np


def iou_score(y_true, y_pred):
    """
    Intersection over Union score implemented using numpy.

    Args:
        y_true: (numpy.ndarray) Target values.
        y_pred: (numpy.ndarray) Predicted values.

    Returns: (numpy.ndarray)
        IoU score for each sample.
    """
    intersection = np.sum(np.reshape(y_true * y_pred, (-1, y_true.shape[-1])), axis=-1)
    union = np.sum(
        np.reshape(np.maximum(y_true, y_pred), (-1, y_true.shape[-1])), axis=-1
    )
    return intersection / union

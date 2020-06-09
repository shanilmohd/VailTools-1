import numpy as np
from tensorflow.keras.utils import get_custom_objects


def register_custom_objects(objects):
    """
    Registers custom objects with tensorflow.keras.
    Should prevent the end user from needing to manually declare custom objects
    when saving and loading models made by or using VaiLTools
    
    Todo: May want to ensure that builtin objects are not overwritten?

    Args:
        objects: Iterable of custom objects to be registered.
    """
    get_custom_objects().update({x.__name__: x for x in objects})


def to_categorical(x, num_classes=None):
    """
    Converts an N dimensional integer encoded numpy array into an N+1 dimensional
    one-hot encoded numpy array.

    Args:
        x: numpy.ndarray
        num_classes: int

    Returns: numpy.ndarray
    """
    x = x.astype(int)
    if num_classes is None:
        num_classes = np.max(x) + 1
    return np.eye(num_classes)[x.astype(int)]

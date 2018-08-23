"""
Provides factory functions for the creation of various building blocks for Keras models.
"""


import numpy as np
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D


def residual_block(
        x,
        activation='selu',
        filter_shape=(3, 3),
        filters=16,
        merge=None,
        project=False,
):
    """
    Implements the two convolution residual block described in
        https://arxiv.org/pdf/1512.03385.pdf

    Args:
        x: (keras.backend.Tensor) Symbolic input tensor
        activation: (str) Usually 'relu', 'elu', or 'selu'
        filter_shape: (tuple[int]) Dimensions of the convolution filters
        filters: (int) Number of filters used in each convolution
        merge: (keras.layers.Layer) Used to merge the residual connection, usually Concatenate or Add
        project: (bool) Toggle application of a 1x1 convolution without non-linearity to the residual connection

    Returns: (keras.backend.Tensor)
        Symbolic output tensor
    """
    if merge is None:
        merge = Add()

    pred = Conv2D(filters, filter_shape, padding='same')(x)
    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)

    pred = Conv2D(filters, filter_shape, padding='same')(pred)
    pred = BatchNormalization()(pred)

    if project:
        x = Conv2D(filters, (1, 1))(x)
        x = BatchNormalization()(x)
    pred = merge([x, pred])
    return Activation(activation)(pred)


def residual_bottlneck_block(
        x,
        activation='selu',
        filter_shape=(3, 3),
        filters=16,
        merge=None,
        neck_filters=None,
        project=False,
):
    """
    Implements the three convolution bottleneck residual block described in
        https://arxiv.org/pdf/1512.03385.pdf

    Args:
        x: (keras.backend.Tensor) Symbolic input tensor
        activation: (str) Usually 'relu', 'elu', or 'selu'
        filter_shape: (tuple[int]) Dimensions of the convolution filters
        filters: (int) Number of filters used in output convolution
        merge: (keras.layers.Layer) Used to merge the residual connection, usually Concatenate or Add
        neck_filters: (int) Number of filters used in bottleneck convolutions
        project: (bool) Toggle application of a 1x1 convolution without non-linearity to the residual connection

    Returns: (keras.backend.Tensor)
        Symbolic output tensor
    """
    if merge is None:
        merge = Add()
    if neck_filters is None:
        neck_filters = max(filters // 4, 1)

    pred = Conv2D(neck_filters, (1, 1))(x)
    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)

    pred = Conv2D(neck_filters, filter_shape, padding='same')(pred)
    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)

    pred = Conv2D(filters, (1, 1))(pred)
    pred = BatchNormalization()(pred)

    if project:
        x = Conv2D(filters, (1, 1))(x)
        x = BatchNormalization()(x)

    pred = merge([x, pred])

    return Activation(activation)(pred)


def dense_block(
        x,
        activation='selu',
        depth=2,
        filter_shape=(3, 3),
        filters=16,
        merge=None,
):
    """
    Implements a densely connected convolution block, as described in https://arxiv.org/abs/1608.06993

    Args:
        x: (keras.backend.Tensor) Symbolic input tensor
        activation: (str) Usually 'relu', 'elu', or 'selu'
        depth: (int) Number of convolutions to use in block construction
        filter_shape: (tuple[int]) Dimensions of the convolution filters
        filters: (int) Number of filters used in output convolution
        merge: (keras.layers.Layer) Used to merge the residual connection, usually Concatenate or Add

    Returns: (keras.backend.Tensor)
        Symbolic output tensor
    """
    if merge is None:
        merge = Concatenate()
    depth = max(depth, 1)

    inputs = [x]
    for _ in range(depth):
        pred = merge(inputs)
        pred = BatchNormalization()(pred)
        pred = Activation(activation)(pred)
        pred = Conv2D(filters=filters, kernel_size=filter_shape, padding='same')(pred)
        inputs.append(pred)
    return pred


def sparse_block(
        x,
        activation='selu',
        depth=2,
        filter_shape=(3, 3),
        filters=16,
        merge=None,
):
    """
    Implements a sparsely connected convolution block, as described in https://arxiv.org/abs/1801.05895

    Args:
        x: (keras.backend.Tensor) Symbolic input tensor
        activation: (str) Usually 'relu', 'elu', or 'selu'
        depth: (int) Number of convolutions to use in block construction
        filter_shape: (tuple[int]) Dimensions of the convolution filters
        filters: (int) Number of filters used in output convolution
        merge: (keras.layers.Layer) Used to merge the residual connection, usually Concatenate or Add

    Returns: (keras.backend.Tensor)
        Symbolic output tensor
    """
    if merge is None:
        merge = Concatenate()
    depth = max(depth, 1)

    inputs = [x]
    for i in range(depth):
        inds = [-2**j for j in range(1 + int(np.log2(i + 1)))]

        pred = merge([inputs[ind] for ind in inds])
        pred = BatchNormalization()(pred)
        pred = Activation(activation)(pred)
        pred = Conv2D(filters=filters, kernel_size=filter_shape, padding='same')(pred)
        inputs.append(pred)
    return pred


def fractal_block(
        x,
        activation='selu',
        depth=2,
        filter_shape=(3, 3),
        filters=16,
        merge=None,
):
    """
    Implements a fractal convolution block, as described in https://arxiv.org/abs/1605.07648

    Args:
        x: (keras.backend.Tensor) Symbolic input tensor
        activation: (str) Usually 'relu', 'elu', or 'selu'
        depth: (int) Number of convolutions to use in block construction
        filter_shape: (tuple[int]) Dimensions of the convolution filters
        filters: (int) Number of filters used in output convolution
        merge: (keras.layers.Layer) Used to merge the residual connection, usually Concatenate or Add

    Returns: (keras.backend.Tensor)
        Symbolic output tensor
    """
    pass


def dilation_block(
        x,
        activation='selu',
        dilations=None,
        filter_shape=(3, 3),
        filters=16,
        merge=None,
):
    """
    Inspired by architectures using dilated convolutions such as:
        https://arxiv.org/abs/1511.07122
        https://arxiv.org/abs/1710.02224
        https://arxiv.org/abs/1802.10062

    Args:
        x: (keras.backend.Tensor) Symbolic input tensor
        activation: (str) Usually 'relu', 'elu', or 'selu'
        dilations: (tuple[int]) Dilation factors to use for parallel convolutions
        filter_shape: (tuple[int]) Dimensions of the convolution filters
        filters: (int) Number of filters used in output convolution
        merge: (keras.layers.Layer) Used to merge the residual connection, usually Concatenate or Add

    Returns: (keras.backend.Tensor)
        Symbolic output tensor
    """
    if dilations is None:
        dilations = tuple(2 ** x for x in range(4))
    if merge is None:
        merge = Add()

    pred = BatchNormalization()(x)
    pred = Activation(activation)(pred)
    preds = [Conv2D(filters, filter_shape, dilation_rate=d, padding='same')(pred) for d in dilations]
    preds = [BatchNormalization()(p) for p in preds]
    return merge(preds)


def residual_dilation_block(
        x,
        activation='selu',
        dilations=None,
        filter_shape=(3, 3),
        filters=16,
        merge=None,
        project=False,
):
    """
    Implements a residual block where the fundamental unit is a dilation block rather than a simple convolution.

    Args:
        x: (keras.backend.Tensor) Symbolic input tensor
        activation: (str) Usually 'relu', 'elu', or 'selu'
        dilations: (tuple[int]) Dilation factors to use for parallel convolutions
        filter_shape: (tuple[int]) Dimensions of the convolution filters
        filters: (int) Number of filters used in output convolution
        merge: (keras.layers.Layer) Used to merge the residual connection, usually Concatenate or Add
        project: (bool) Toggle application of a 1x1 convolution without non-linearity to the residual connection

    Returns: (keras.backend.Tensor)
        Symbolic output tensor
    """
    if dilations is None:
        dilations = tuple(2 ** x for x in range(4))
    if merge is None:
        merge = Add()

    pred = dilation_block(
        x,
        activation=activation,
        dilations=dilations,
        filter_shape=filter_shape,
        filters=filters,
        merge=merge,
    )
    pred = dilation_block(
        pred,
        activation=activation,
        dilations=dilations,
        filter_shape=filter_shape,
        filters=filters,
        merge=merge,
    )

    if project:
        x = Conv2D(filters, 1)(x)
        x = BatchNormalization()(x)
    return merge([pred, x])

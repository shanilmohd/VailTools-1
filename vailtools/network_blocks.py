"""
Provides factory functions for the creation of various building blocks for Keras models.
"""


import numpy as np
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv1D, Conv2D, Multiply


def residual_block(
        x,
        activation='selu',
        filters=16,
        kernel_size=(3, 3),
        merge=None,
        padding='same',
        project=False,
):
    """
    Implements the two convolution residual block described in
        https://arxiv.org/pdf/1512.03385.pdf

    Args:
        x: (keras.backend.Tensor)
            Symbolic input tensor.
        activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.
        filters: (int)
            Number of filters used in each convolution.
        kernel_size: (tuple[int] or int)
            Dimensions of the convolution filters.
        merge: (keras.layers.Layer)
            Keras layer that merges the input and output branches of a residual block.
        padding: (str)
            Padding strategy applied during convolution operations.
        project: (bool)
            Toggle application of a 1x1 convolution without non-linearity to the residual connection.

    Returns: (keras.backend.Tensor)
        Symbolic output tensor
    """
    if merge is None:
        merge = Add()

    pred = Conv2D(filters, kernel_size=kernel_size, padding=padding)(x)
    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)

    pred = Conv2D(filters, kernel_size=kernel_size, padding=padding)(pred)
    pred = BatchNormalization()(pred)

    if project:
        x = Conv2D(filters, kernel_size=(1, 1))(x)
        x = BatchNormalization()(x)
    pred = merge([x, pred])
    return Activation(activation)(pred)


def residual_bottlneck_block(
        x,
        activation='selu',
        kernel_size=(3, 3),
        filters=16,
        merge=None,
        neck_filters=None,
        padding='same',
        project=False,
):
    """
    Implements the three convolution bottleneck residual block described in
        https://arxiv.org/pdf/1512.03385.pdf

    Args:
        x: (keras.backend.Tensor)
            Symbolic input tensor.
        activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.
        filters: (int)
            Number of filters used in each convolution.
        kernel_size: (tuple[int] or int)
            Dimensions of the convolution filters.
        merge: (keras.layers.Layer)
            Keras layer that merges the input and output branches of a residual block.
        neck_filters: (int)
            Number of filters used in bottleneck convolutions.
        padding: (str)
            Padding strategy applied during convolution operations.
        project: (bool)
            Toggle application of a 1x1 convolution without non-linearity to the residual connection.

    Returns: (keras.backend.Tensor)
        Symbolic output tensor
    """
    if merge is None:
        merge = Add()
    if neck_filters is None:
        neck_filters = max(filters // 4, 1)

    pred = Conv2D(neck_filters, kernel_size=(1, 1))(x)
    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)

    pred = Conv2D(neck_filters, kernel_size=kernel_size, padding=padding)(pred)
    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)

    pred = Conv2D(filters, kernel_size=(1, 1))(pred)
    pred = BatchNormalization()(pred)

    if project:
        x = Conv2D(filters, kernel_size=(1, 1))(x)
        x = BatchNormalization()(x)
    pred = merge([x, pred])
    return Activation(activation)(pred)


def dense_block(
        x,
        activation='selu',
        depth=2,
        filters=16,
        kernel_size=(3, 3),
        merge=None,
        padding='same',
):
    """
    Implements a densely connected convolution block, as described in https://arxiv.org/abs/1608.06993

    Args:
        x: (keras.backend.Tensor)
            Symbolic input tensor.
        activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.
        depth: (int)
            Number of convolutions used in block construction.
        filters: (int)
            Number of filters used in each convolution.
        kernel_size: (tuple[int] or int)
            Dimensions of the convolution filters.
        merge: (keras.layers.Layer)
            Keras layer that merges the input and output branches of a residual block.
        padding: (str)
            Padding strategy applied during convolution operations.

    Returns: (keras.backend.Tensor)
        Symbolic output tensor
    """
    if merge is None:
        merge = Concatenate()
    depth = max(depth, 1)

    inputs = [x]
    pred = x
    for _ in range(depth):
        pred = merge(inputs)
        pred = BatchNormalization()(pred)
        pred = Activation(activation)(pred)
        pred = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(pred)
        inputs.append(pred)
    return pred


def sparse_block(
        x,
        activation='selu',
        depth=4,
        filters=16,
        kernel_size=(3, 3),
        merge=None,
        padding='same',
):
    """
    Implements a sparsely connected convolution block, as described in https://arxiv.org/abs/1801.05895

    Args:
        x: (keras.backend.Tensor)
            Symbolic input tensor.
        activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.
        depth: (int)
            Number of convolutions used in block construction.
        filters: (int)
            Number of filters used in each convolution.
        kernel_size: (tuple[int] or int)
            Dimensions of the convolution filters.
        merge: (keras.layers.Layer)
            Keras layer that merges the input and output branches of a residual block.
        padding: (str)
            Padding strategy applied during convolution operations.

    Returns: (keras.backend.Tensor)
        Symbolic output tensor
    """
    if merge is None:
        merge = Concatenate()
    depth = max(depth, 1)

    inputs = [x]
    pred = x
    for i in range(depth):
        inds = [-2**j for j in range(1 + int(np.log2(i + 1)))]

        pred = merge([inputs[ind] for ind in inds])
        pred = BatchNormalization()(pred)
        pred = Activation(activation)(pred)
        pred = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(pred)
        inputs.append(pred)
    return pred


def fractal_block():
    """
    Will implement a fractal convolution block, as described in https://arxiv.org/abs/1605.07648
    """
    raise NotImplementedError('The fractal_block implementation has not been completed!')


def dilation_block(
        x,
        activation='selu',
        dilations=None,
        filters=16,
        kernel_size=(3, 3),
        merge=None,
        padding='same',
):
    """
    Inspired by architectures using dilated convolutions such as:
        https://arxiv.org/abs/1511.07122
        https://arxiv.org/abs/1710.02224
        https://arxiv.org/abs/1802.10062

    Args:
        x: (keras.backend.Tensor)
            Symbolic input tensor.
        activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.
        dilations: (tuple[int])
            Dilation rates used for parallel convolutions.
        filters: (int)
            Number of filters used in each convolution.
        kernel_size: (tuple[int] or int)
            Dimensions of the convolution filters.
        merge: (keras.layers.Layer)
            Keras layer that merges the input and output branches of a residual block.
        padding: (str)
            Padding strategy applied during convolution operations.

    Returns: (keras.backend.Tensor)
        Symbolic output tensor
    """
    if dilations is None:
        dilations = tuple(2 ** x for x in range(4))
    if merge is None:
        merge = Add()

    pred = BatchNormalization()(x)
    pred = Activation(activation)(pred)
    preds = [Conv2D(filters, kernel_size=kernel_size, dilation_rate=d, padding=padding)(pred) for d in dilations]
    preds = [BatchNormalization()(p) for p in preds]
    return merge(preds)


def residual_dilation_block(
        x,
        activation='selu',
        dilations=None,
        filters=16,
        kernel_size=(3, 3),
        merge=None,
        padding='same',
        project=False,
):
    """
    Implements a residual block where the fundamental unit is a dilation block rather than a simple convolution.

    Args:
        x: (keras.backend.Tensor)
            Symbolic input tensor.
        activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.
        dilations: (tuple[int])
            Dilation rates used for parallel convolutions.
        filters: (int)
            Number of filters used in each convolution.
        kernel_size: (tuple[int] or int)
            Dimensions of the convolution filters.
        merge: (keras.layers.Layer)
            Keras layer that merges the input and output branches of a residual block.
        padding: (str)
            Padding strategy applied during convolution operations.
        project: (bool)
            Toggle application of a 1x1 convolution without non-linearity to the residual connection

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
        filters=filters,
        kernel_size=kernel_size,
        merge=merge,
        padding=padding,
    )
    pred = dilation_block(
        pred,
        activation=activation,
        dilations=dilations,
        filters=filters,
        kernel_size=kernel_size,
        merge=merge,
        padding=padding,
    )

    if project:
        x = Conv2D(filters, 1)(x)
        x = BatchNormalization()(x)
    return merge([pred, x])


def wavenet_block(
    x,
    activation='tanh',
    dilation_rate=1,
    filters=16,
    gate_activation='sigmoid',
    gate_merge=None,
    kernel_size=(3, 3),
    residual_merge=None,
):
    """
    Implements the basic building block of the WaveNet architecture,
    as described in https://arxiv.org/abs/1609.03499


    Args:
        x: (keras.backend.Tensor)
            Symbolic input tensor.
        activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.
            Applied to the non-gate branch of a gated activation unit.
        dilation_rate: (int)
            Dilation rate used in convolutions.
        filters: (int)
            Number of filters used in convolutions.
        gate_activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.
            Applied to the gate branch of a gated activation unit
        gate_merge: (keras.layers.Layer)
            Keras layer that merges the gate and non-gate branch of a gated activation unit.
        kernel_size: (tuple[int] or int)
            Dimensions of the convolution filters.
        residual_merge: (keras.layers.Layer)
            Keras layer that merges the input and output branches of a residual block.

    Returns: (keras.backend.Tensor)
        Symbolic output tensor
    """
    if gate_merge is None:
        gate_merge = Multiply()
    if residual_merge is None:
        residual_merge = Add()

    pred = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(x)
    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)

    gate = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(x)
    gate = BatchNormalization()(gate)
    gate = Activation(gate_activation)(gate)

    gate_activation = gate_merge([pred, gate])

    skip_out = Conv1D(filters=filters, kernel_size=1, dilation_rate=dilation_rate)(gate_activation)
    unit_pred = residual_merge([x, skip_out])
    return unit_pred, skip_out

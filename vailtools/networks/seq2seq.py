from itertools import cycle

from tensorflow.keras import layers
from tensorflow.keras.models import Model

from ..layers import SnailAttentionBlock, SnailTCBlock, WaveNetBlock


def snail_mdp(
        embedding_input_dim=None,
        embedding_output_dim=24,
        filters=32,
        final_activation='linear',
        input_shape=(None, None),
        key_size=32,
        output_size=10,
        return_sequences=False,
        sequence_length=32,
        value_size=32,
):
    """
    Settings taken from Section C.1 of https://arxiv.org/abs/1707.03141

    Args:
        embedding_input_dim:
        embedding_output_dim:
        filters:
        final_activation:
        input_shape:
        key_size:
        output_size:
        return_sequences:
        sequence_length:
        value_size:

    Returns:
    """
    inputs = layers.Input(shape=input_shape)
    if embedding_input_dim and embedding_output_dim:
        pred = layers.Embedding(embedding_input_dim, embedding_output_dim)(inputs)
    else:
        pred = inputs

    pred = SnailTCBlock(sequence_length=sequence_length, filters=filters)(pred)
    pred = SnailTCBlock(sequence_length=sequence_length, filters=filters)(pred)
    pred = SnailAttentionBlock(key_size=key_size, value_size=value_size)(pred)

    # Used to reduce model output and apply a final activation
    pred = layers.LSTM(
        output_size,
        return_sequences=return_sequences,
        activation=final_activation,
    )(pred)
    return Model(inputs=inputs, outputs=pred)


def snail_control(
        embedding_input_dim=None,
        embedding_output_dim=24,
        filters=32,
        final_activation='linear',
        input_shape=(None, None),
        key_size=16,
        output_size=10,
        return_sequences=False,
        sequence_length=32,
        value_size=16,
):
    """
    Settings taken from Section C.2 of https://arxiv.org/abs/1707.03141

    Args:
        embedding_input_dim:
        embedding_output_dim:
        filters:
        final_activation:
        input_shape:
        key_size:
        output_size:
        return_sequences:
        sequence_length:
        value_size:

    Returns:
    """
    inputs = layers.Input(shape=input_shape)
    if embedding_input_dim and embedding_output_dim:
        pred = layers.Embedding(embedding_input_dim, embedding_output_dim)(inputs)
    else:
        pred = inputs

    pred = SnailAttentionBlock(key_size=key_size, value_size=value_size)(pred)
    pred = SnailTCBlock(sequence_length=sequence_length, filters=filters)(pred)
    pred = SnailTCBlock(sequence_length=sequence_length, filters=filters)(pred)
    pred = SnailAttentionBlock(key_size=key_size, value_size=value_size)(pred)

    # Used to reduce model output and apply a final activation
    pred = layers.LSTM(
        output_size,
        return_sequences=return_sequences,
        activation=final_activation,
    )(pred)
    return Model(inputs=inputs, outputs=pred)


def snail_visual(
        embedding_input_dim=None,
        embedding_output_dim=24,
        filters=32,
        final_activation='linear',
        input_shape=(None, None),
        key_size=16,
        output_size=10,
        return_sequences=False,
        sequence_length=32,
        value_size=16,
):
    """
    Settings taken from Section C.3 of https://arxiv.org/abs/1707.03141

    Args:
        embedding_input_dim:
        embedding_output_dim:
        filters:
        final_activation:
        input_shape:
        key_size:
        output_size:
        return_sequences:
        sequence_length:
        value_size:

    Returns:
    """
    inputs = layers.Input(shape=input_shape)
    if embedding_input_dim and embedding_output_dim:
        pred = layers.Embedding(embedding_input_dim, embedding_output_dim)(inputs)
    else:
        pred = inputs

    pred = SnailTCBlock(sequence_length=sequence_length, filters=filters)(pred)
    pred = SnailAttentionBlock(key_size=key_size, value_size=value_size)(pred)
    pred = SnailTCBlock(sequence_length=sequence_length, filters=filters)(pred)
    pred = SnailAttentionBlock(key_size=key_size, value_size=value_size)(pred)

    # Used to reduce model output and apply a final activation
    pred = layers.LSTM(
        output_size,
        return_sequences=return_sequences,
        activation=final_activation,
    )(pred)

    return Model(inputs=inputs, outputs=pred)


def wave_net(
        activation='tanh',
        bias_initializer='zeros',
        depth=10,
        dilation_rates=None,
        embedding_input_dim=None,
        embedding_output_dim=24,
        filters=16,
        final_activation='softmax',
        flatten_output=False,
        gate_activation='sigmoid',
        gate_merge=layers.Multiply,
        input_shape=(None, None),
        kernel_initializer='glorot_uniform',
        kernel_size=3,
        output_channels=1,
        skip_merge=layers.Concatenate,
        tail_activation='relu',
):
    """
    An implementation of WaveNet, described in https://arxiv.org/abs/1609.03499, using Keras.
    Works on time series data with dimensions (samples, time steps, features).
    
    Args:
        activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.
            Activation applied to non-gate portion of a gated activation unit.
        bias_initializer: (str or Callable)
            Name or instance of a keras.initializers.Initializer.
        depth: (int)
            Number of consecutive gated residual blocks used in model construction.
        dilation_rates: (tuple[int])
            Sequence of dilation rates used cyclically during the creation of gated residual blocks.
        embedding_input_dim:

        embedding_output_dim:

        filters: (int)
            Number of filters used in each convolution operation.
        final_activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function
            Final operation of the network, determines the possible range of network outputs.
        flatten_output: (bool)
            Toggles the use of a global average pooling operation to remove the time dimension from the outputs.
        gate_activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.
            Activation applied to the gate portion of each gated activation unit.
        gate_merge: tensorflow.keras.layers.Merge
            Layer that merges the forget gate and value branch.
            Common options are Add, Concatenate, Maximum, and Multiply.
        input_shape: (tuple[int or None])
            Specifies the time steps and features dimensions of the input data, does not include the samples dimension.
        kernel_initializer: (str or Callable)
            Name or instance of a keras.initializers.Initializer.
        kernel_size: (int)
            Determines the length of the 1D kernels used in each convolution operation.
            Name or instance of a keras optimizer that will be used for training.
        output_channels: (int)
            Number of output channels/features.
        skip_merge: tensorflow.keras.layers.Merge
            Layer that handles skip connection merge behavior.
            Common options are Add, Concatenate, Maximum, and Multiply.
        tail_activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.

    Returns: (keras.models.Model)
        A compiled WaveNet
    """
    if dilation_rates is None:
        dilation_rates = tuple(2 ** x for x in range(10))

    inputs = layers.Input(shape=input_shape)

    if embedding_input_dim and embedding_output_dim:
        pred = layers.Embedding(embedding_input_dim, embedding_output_dim)(inputs)
    else:
        pred = inputs

    pred = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='causal')(pred)

    for i, dilation_rate in zip(range(depth), cycle(dilation_rates)):
        pred = WaveNetBlock(
            activation=activation,
            bias_initializer=bias_initializer,
            dilation_rate=dilation_rate,
            filters=filters,
            gate_activation=gate_activation,
            gate_merge=gate_merge,
            kernel_initializer=kernel_initializer,
            kernel_size=kernel_size,
            skip_merge=skip_merge,
        )(pred)

    pred = layers.BatchNormalization()(pred)
    pred = layers.Activation(tail_activation)(pred)
    pred = layers.Conv1D(
        bias_initializer=bias_initializer,
        filters=filters,
        kernel_initializer=kernel_initializer,
        kernel_size=1,
    )(pred)

    pred = layers.BatchNormalization()(pred)
    pred = layers.Activation(tail_activation)(pred)
    pred = layers.Conv1D(
        activation=final_activation,
        bias_initializer=bias_initializer,
        filters=output_channels,
        kernel_initializer=kernel_initializer,
        kernel_size=1,
    )(pred)

    if flatten_output:
        pred = layers.GlobalAvgPool1D()(pred)

    return Model(inputs=inputs, outputs=pred)

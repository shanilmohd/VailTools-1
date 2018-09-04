from itertools import cycle

from keras.layers import Activation, Add, BatchNormalization, Conv1D, Input, Multiply
from keras.models import Model
from keras.optimizers import SGD

from ..network_blocks import wavenet_block


def wave_net(
    activation='tanh',
    depth=10,
    dilation_rates=None,
    filters=16,
    final_activation='softmax',
    gate_activation='sigmoid',
    gate_merge=None,
    input_shape=(None, None),
    kernel_size=3,
    loss='mse',
    optimizer=None,
    output_channels=1,
    residual_merge=None,
    tail_activation='relu',
):
    """
    An implementation of WaveNet, as described in https://arxiv.org/abs/1609.03499, using Keras.
    Works on time series data with dimensions (samples, time steps, features).
    
    Args:
        activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.
            Activation applied to non-gate portion of a gated activation unit.
        depth: (int)
            Number of consecutive gated residual blocks used in model construction.
        dilation_rates: (tuple[int])
            Sequence of dilation rates used cyclically during the creation of gated residual blocks.
        filters: (int)
            Number of filters used in each convolution operation.
        final_activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function
            Final operation of the network, determines the possible range of network outputs.
        gate_activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.
            Activation applied to the gate portion of each gated activation unit.
        gate_merge: (keras.layers.layer)
            Keras layer to merge the prediction branch and gate branch of a gated activation unit.
        input_shape: (tuple[int or None])
            Specifies the time steps and features dimensions of the input data, does not include the samples dimension.
        kernel_size: (int)
            Determines the length of the 1D kernels used in each convolution operation.
        loss: (str)
            Name of a keras loss function or an instance of a  keras/Tensorflow loss function.
        optimizer: (str or keras.optimizers.Optimizer)
            Name or instance of a keras optimizer that will be used for training.
        output_channels: (int)
            Number of output channels/features.
        residual_merge: (keras.layers.layer)
            Keras layer that merges the input and output branches of a residual block, usually keras.layers.Add.
        tail_activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.

    Returns: (keras.models.Model)
        A compiled WaveNet
    """
    if dilation_rates is None:
        dilation_rates = tuple(2 ** x for x in range(10))
    if gate_merge is None:
        gate_merge = Multiply()
    if optimizer is None:
        optimizer = SGD(momentum=0.9)
    if residual_merge is None:
        residual_merge = Add()

    input_ = Input(shape=input_shape)
    pred = Conv1D(filters=filters, kernel_size=kernel_size, padding='causal')(input_)

    skip_connections = []
    for i, dilation_rate in zip(range(depth), cycle(dilation_rates)):
        pred, skip_out = wavenet_block(
            pred,
            activation=activation,
            dilation_rate=dilation_rate,
            filters=filters,
            gate_activation=gate_activation,
            gate_merge=gate_merge,
            kernel_size=kernel_size,
            residual_merge=residual_merge,
        )
        skip_connections.append(skip_out)
    pred = residual_merge(skip_connections)

    pred = BatchNormalization()(pred)
    pred = Activation(tail_activation)(pred)
    pred = Conv1D(filters=filters, kernel_size=1)(pred)

    pred = BatchNormalization()(pred)
    pred = Activation(tail_activation)(pred)
    pred = Conv1D(filters=output_channels, kernel_size=1)(pred)

    pred = Activation(final_activation)(pred)

    model = Model(inputs=input_, outputs=pred)
    model.compile(optimizer=optimizer, loss=loss)
    return model

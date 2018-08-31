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
    input_shape=(None, None, None),
    kernel_size=3,
    loss='mse',
    optimizer=None,
    output_channels=1,
    residual_merge=None,
    tail_activation='relu',
):
    """
    An implementation of WaveNet, as described in https://arxiv.org/abs/1609.03499, using Keras.
    
    Args:
        activation: (str or keras.)
        depth: (int)
        dilation_rates: (tuple[int])
        filters: (int)
        final_activation: (str)
        gate_activation: (str)
        gate_merge: (keras.layers.layer)
        input_shape: (tuple[int or None])
        kernel_size: (int)
        loss: (str)
        optimizer: (keras.optimizers.Optimizer)
        output_channels: (int)
        residual_merge: (keras.layers.layer)
        tail_activation: (str)

    Returns: (keras.models.Model)
        A compiled WaveNet
    """
    if dilation_rates is None:
        dilation_rates = [2 ** x for x in range(10)]
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

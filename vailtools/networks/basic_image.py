"""
Several architectures designed for image to image mappings.
"""


from keras.layers import Activation, Add, BatchNormalization, Concatenate, \
    Conv2D, GaussianNoise, Input, MaxPool2D, UpSampling2D
from keras.models import Model
from keras.optimizers import SGD

from .. import network_blocks


def restrict_net(
        activation='selu',
        depth=4,
        filters=16,
        final_activation='selu',
        input_shape=(None, None, None),
        loss=None,
        noise_std=0.1,
        optimizer=None,
        output_channels=1,
):
    """A U-Net without skip connections.

    Args:
        activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.
            Applied throughout the network, except for the final activation.
        depth: (int)
            Number of levels used in the construction of the restrictive/reconstituting paths.
        filters: (int)
            Number of filters used in convolutions, altered by spatial resampling operations.
        final_activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function
            Final operation of the network, determines the possible range of network outputs.
        input_shape: (tuple[int or None])
            Specifies the dimensions of the input data, does not include the samples dimension.
        loss: (str)
            Name of a keras loss function or an instance of a  keras/Tensorflow loss function.
        noise_std: (float)
            Standard deviation of an additive 0-mean Gaussian noise applied to network inputs.
        optimizer: (str or keras.optimizers.Optimizer)
            Name or instance of a keras optimizer that will be used for training.
        output_channels: (int)
            Number of output channels/features.

    Returns: (keras.models.Model)
        A compiled and ready-to-use Restrict-Net.
    """
    if loss is None:
        loss = 'mse'
    if optimizer is None:
        optimizer = SGD(momentum=0.9)

    inputs = Input(shape=input_shape)
    pred = GaussianNoise(stddev=noise_std)(inputs)

    # Restriction
    for _ in range(depth):
        pred = BatchNormalization()(pred)
        pred = Activation(activation)(pred)
        pred = Conv2D(filters, (3, 3), padding='same')(pred)

        pred = BatchNormalization()(pred)
        pred = Activation(activation)(pred)
        pred = Conv2D(filters, (3, 3), padding='same')(pred)

        pred = MaxPool2D()(pred)
        filters *= 2

    # Transition
    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)
    pred = Conv2D(filters, (3, 3), padding='same')(pred)

    # Reconstitution
    for _ in range(depth):
        pred = UpSampling2D()(pred)
        filters //= 2
        pred = Conv2D(filters, (3, 3), padding='same')(pred)

        pred = BatchNormalization()(pred)
        pred = Activation(activation)(pred)
        pred = Conv2D(filters, (3, 3), padding='same')(pred)

        pred = BatchNormalization()(pred)
        pred = Activation(activation)(pred)
        pred = Conv2D(filters, (3, 3), padding='same')(pred)

    # Ensure the correct number of output channels and apply the final activation
    pred = Conv2D(output_channels, (1, 1))(pred)
    pred = BatchNormalization()(pred)
    pred = Activation(final_activation)(pred)

    model = Model(inputs=inputs, outputs=pred)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def u_net(
        activation='selu',
        depth=4,
        filters=16,
        final_activation='selu',
        input_shape=(None, None, None),
        loss=None,
        noise_std=0.1,
        optimizer=None,
        output_channels=1,
):
    """A Keras implementation of the U-Net architecture.
     See https://arxiv.org/pdf/1505.04597.pdf for details.

    Deviations:
        - Uses a BN-activation-conv structure rather than conv-activation
        - Uses padded convolutions to simplify dimension arithmetic
        - Does not use reflection padding on inputs
        - Cropping is not used on the cross connections
        - Uses 3x3 up-conv, rather than 2x2

    Args:
        activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.
            Applied throughout the network, except for the final activation.
        depth: (int)
            Number of levels used in the construction of the restrictive/reconstituting paths.
        filters: (int)
            Number of filters used in convolutions, altered by spatial resampling operations.
        final_activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function
            Final operation of the network, determines the possible range of network outputs.
        input_shape: (tuple[int or None])
            Specifies the dimensions of the input data, does not include the samples dimension.
        loss: (str)
            Name of a keras loss function or an instance of a  keras/Tensorflow loss function.
        noise_std: (float)
            Standard deviation of an additive 0-mean Gaussian noise applied to network inputs.
        optimizer: (str or keras.optimizers.Optimizer)
            Name or instance of a keras optimizer that will be used for training.
        output_channels: (int)
            Number of output channels/features.

    Returns: (keras.models.Model)
        A compiled and ready-to-use U-Net.
    """
    if loss is None:
        loss = 'mse'
    if optimizer is None:
        optimizer = SGD(momentum=0.9)

    inputs = Input(shape=input_shape)
    pred = GaussianNoise(stddev=noise_std)(inputs)

    # Restriction
    crosses = []
    for _ in range(depth):
        pred = BatchNormalization()(pred)
        pred = Activation(activation)(pred)
        pred = Conv2D(filters, (3, 3), padding='same')(pred)

        pred = BatchNormalization()(pred)
        pred = Activation(activation)(pred)
        pred = Conv2D(filters, (3, 3), padding='same')(pred)

        crosses.append(pred)

        pred = MaxPool2D()(pred)
        filters *= 2

    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)
    pred = Conv2D(filters, (3, 3), padding='same')(pred)

    # Reconstitution
    for cross in crosses[::-1]:
        pred = UpSampling2D()(pred)
        filters //= 2
        pred = Conv2D(filters, (3, 3), padding='same')(pred)

        pred = Concatenate()([pred, cross])
        pred = BatchNormalization()(pred)
        pred = Activation(activation)(pred)
        pred = Conv2D(filters, (3, 3), padding='same')(pred)

        pred = BatchNormalization()(pred)
        pred = Activation(activation)(pred)
        pred = Conv2D(filters, (3, 3), padding='same')(pred)

    pred = Conv2D(output_channels, (1, 1))(pred)
    pred = BatchNormalization()(pred)
    pred = Activation(final_activation)(pred)

    model = Model(inputs=inputs, outputs=pred)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def res_u_net(
        activation='selu',
        depth=4,
        filters=16,
        final_activation='selu',
        input_shape=(None, None, None),
        loss=None,
        merge=None,
        noise_std=0.1,
        optimizer=None,
        output_channels=1,
):
    """A U-Net with residual blocks at each level.

    Args:
        activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.
            Applied throughout the network, except for the final activation.
        depth: (int)
            Number of levels used in the construction of the restrictive/reconstituting paths.
        filters: (int)
            Number of filters used in convolutions, altered by spatial resampling operations.
        final_activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function
            Final operation of the network, determines the possible range of network outputs.
        input_shape: (tuple[int or None])
            Specifies the dimensions of the input data, does not include the samples dimension.
        loss: (str)
            Name of a keras loss function or an instance of a  keras/Tensorflow loss function.
        merge: (keras.layers.layer)
            Keras layer that merges the input and output branches of a residual block, usually keras.layers.Add.
        noise_std: (float)
            Standard deviation of an additive 0-mean Gaussian noise applied to network inputs.
        optimizer: (str or keras.optimizers.Optimizer)
            Name or instance of a keras optimizer that will be used for training.
        output_channels: (int)
            Number of output channels/features.

    Returns: (keras.models.Model)
        A compiled and ready-to-use Residual U-Net.
    """
    if loss is None:
        loss = 'mse'
    if merge is None:
        merge = Add()
    if optimizer is None:
        optimizer = SGD(momentum=0.9)

    inputs = Input(shape=input_shape)
    pred = GaussianNoise(stddev=noise_std)(inputs)

    # Restriction
    crosses = []
    for _ in range(depth):
        pred = network_blocks.residual_block(
            pred,
            filters=filters,
            activation=activation,
            project=True,
            merge=merge,
        )

        crosses.append(pred)

        pred = MaxPool2D()(pred)
        filters *= 2

    pred = network_blocks.residual_block(
        pred,
        filters=filters,
        activation=activation,
        project=True,
        merge=merge,
    )

    # Reconstitution
    for cross in crosses[::-1]:
        pred = UpSampling2D()(pred)
        filters //= 2
        pred = Conv2D(filters, (3, 3), padding='same')(pred)

        pred = Concatenate()([pred, cross])
        pred = network_blocks.residual_block(
            pred,
            filters=filters,
            activation=activation,
            project=True,
            merge=merge,
        )

    pred = Conv2D(output_channels, (1, 1))(pred)
    pred = BatchNormalization()(pred)
    pred = Activation(final_activation)(pred)

    model = Model(inputs=inputs, outputs=pred)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def dilated_net(
        activation='selu',
        depth=3,
        filters=32,
        final_activation='sigmoid',
        input_shape=(None, None, None),
        loss=None,
        merge=None,
        noise_std=0.1,
        optimizer=None,
        output_channels=1,
):
    """A neural network primarily composed of dilated convolutions.
    Uses exponentially dilated convolutions to operate on multi-scale features.
    No up-sampling or down-sampling is used, since sequential dilated convolutions
    have extremely large effective receptive fields.

    Args:
        activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.
            Applied throughout the network, except for the final activation.
        depth: (int)
            Number of levels used in the construction of the restrictive/reconstituting paths.
        filters: (int)
            Number of filters used in convolutions, altered by spatial resampling operations.
        final_activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function
            Final operation of the network, determines the possible range of network outputs.
        input_shape: (tuple[int or None])
            Specifies the dimensions of the input data, does not include the samples dimension.
        loss: (str)
            Name of a keras loss function or an instance of a  keras/Tensorflow loss function.
        merge: (keras.layers.layer)
            Keras layer that merges the input and output branches of a residual block, usually keras.layers.Add.
        noise_std: (float)
            Standard deviation of an additive 0-mean Gaussian noise applied to network inputs.
        optimizer: (str or keras.optimizers.Optimizer)
            Name or instance of a keras optimizer that will be used for training.
        output_channels: (int)
            Number of output channels/features.

    Returns: (keras.models.Model)
        A compiled and ready-to-use Residual-U-Net.
    """
    if loss is None:
        loss = 'mse'
    if merge is None:
        merge = Add()
    if optimizer is None:
        optimizer = SGD(momentum=0.9)

    inputs = Input(shape=input_shape)
    pred = GaussianNoise(stddev=noise_std)(inputs)

    for _ in range(depth):
        pred = network_blocks.dilation_block(
            pred,
            filters=filters,
            activation=activation,
            merge=merge,
        )

    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)
    pred = Conv2D(filters, (3, 3), padding='same')(pred)

    pred = Conv2D(output_channels, (1, 1))(pred)
    pred = BatchNormalization()(pred)
    pred = Activation(final_activation)(pred)

    model = Model(inputs=inputs, outputs=pred)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def res_dilated_net(
        activation='selu',
        depth=3,
        filters=32,
        final_activation='sigmoid',
        input_shape=(None, None, None),
        loss=None,
        merge=None,
        noise_std=0.1,
        optimizer=None,
        output_channels=1,
):
    """A neural network primarily composed of dilated convolutions.
    Uses exponentially dilated convolutions to operate on multi-scale features.
    No up-sampling or down-sampling is used, since sequential dilated convolutions
    have extremely large effective receptive fields.

    Args:
        activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.
            Applied throughout the network, except for the final activation.
        depth: (int)
            Number of levels used in the construction of the restrictive/reconstituting paths.
        filters: (int)
            Number of filters used in convolutions, altered by spatial resampling operations.
        final_activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function
            Final operation of the network, determines the possible range of network outputs.
        input_shape: (tuple[int or None])
            Specifies the dimensions of the input data, does not include the samples dimension.
        loss: (str)
            Name of a keras loss function or an instance of a  keras/Tensorflow loss function.
        merge: (keras.layers.layer)
            Keras layer that merges the input and output branches of a residual block, usually keras.layers.Add.
        noise_std: (float)
            Standard deviation of an additive 0-mean Gaussian noise applied to network inputs.
        optimizer: (str or keras.optimizers.Optimizer)
            Name or instance of a keras optimizer that will be used for training.
        output_channels: (int)
            Number of output channels/features.

    Returns: (keras.models.Model)
        A compiled and ready-to-use Residual-U-Net.
    """
    if loss is None:
        loss = 'mse'
    if merge is None:
        merge = Add()
    if optimizer is None:
        optimizer = SGD(momentum=0.9)

    inputs = Input(shape=input_shape)
    pred = GaussianNoise(stddev=noise_std)(inputs)

    pred = Conv2D(filters, (1, 1))(pred)
    for _ in range(depth):
        pred = network_blocks.residual_dilation_block(
            pred,
            filters=filters,
            activation=activation,
            merge=merge,
        )

    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)
    pred = Conv2D(filters, (3, 3), padding='same')(pred)

    pred = Conv2D(output_channels, (1, 1))(pred)
    pred = BatchNormalization()(pred)
    pred = Activation(final_activation)(pred)

    model = Model(inputs=inputs, outputs=pred)
    model.compile(optimizer=optimizer, loss=loss)
    return model

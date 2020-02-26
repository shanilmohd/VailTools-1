"""
Neural networks for learning mappings from images to images.
"""

from tensorflow.keras import layers as k_layers
from tensorflow.keras.models import Model

from .. import layers


def restrict_net(
        activation='selu',
        bias_initializer='zeros',
        depth=4,
        filters=16,
        final_activation='selu',
        input_shape=(None, None, None),
        kernel_initializer='glorot_uniform',
        noise_std=0.1,
        output_channels=1,
):
    """A U-Net without skip connections.

    Args:
        activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.
            Applied throughout the network, except for the final activation.
        bias_initializer: (str or Callable)
            Name or instance of a keras.initializers.Initializer.
        depth: (int)
            Number of levels used in the construction of the restrictive/reconstituting paths.
        filters: (int)
            Number of filters used in convolutions, altered by spatial resampling operations.
        final_activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function
            Final operation of the network, determines the possible range of network outputs.
        input_shape: (tuple[int or None])
            Specifies the dimensions of the input data, does not include the samples dimension.
        kernel_initializer: (str or Callable)
            Name or instance of a keras.initializers.Initializer.
        noise_std: (float)
            Standard deviation of an additive 0-mean Gaussian noise applied to network inputs.
        output_channels: (int)
            Number of output channels/features.

    Returns: (keras.models.Model)
        A compiled and ready-to-use Restrict-Net.
    """
    inputs = k_layers.Input(shape=input_shape)
    pred = k_layers.GaussianNoise(stddev=noise_std)(inputs)

    # Restriction
    for _ in range(depth):
        pred = k_layers.BatchNormalization()(pred)
        pred = k_layers.Activation(activation)(pred)
        pred = k_layers.Conv2D(
            bias_initializer=bias_initializer,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=(3, 3),
            padding='same',
        )(pred)

        pred = k_layers.BatchNormalization()(pred)
        pred = k_layers.Activation(activation)(pred)
        pred = k_layers.Conv2D(
            bias_initializer=bias_initializer,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=(3, 3),
            padding='same',
        )(pred)

        pred = k_layers.MaxPool2D()(pred)
        filters *= 2

    # Transition
    pred = k_layers.BatchNormalization()(pred)
    pred = k_layers.Activation(activation)(pred)
    pred = k_layers.Conv2D(
        bias_initializer=bias_initializer,
        filters=filters,
        kernel_initializer=kernel_initializer,
        kernel_size=(3, 3),
        padding='same',
    )(pred)

    # Reconstitution
    for _ in range(depth):
        pred = k_layers.UpSampling2D()(pred)
        filters //= 2
        pred = k_layers.Conv2D(
            bias_initializer=bias_initializer,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=(3, 3),
            padding='same',
        )(pred)

        pred = k_layers.BatchNormalization()(pred)
        pred = k_layers.Activation(activation)(pred)
        pred = k_layers.Conv2D(
            bias_initializer=bias_initializer,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=(3, 3),
            padding='same',
        )(pred)

        pred = k_layers.BatchNormalization()(pred)
        pred = k_layers.Activation(activation)(pred)
        pred = k_layers.Conv2D(
            bias_initializer=bias_initializer,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=(3, 3),
            padding='same',
        )(pred)

    # Ensure the correct number of output channels and apply the final activation
    pred = k_layers.Conv2D(
        bias_initializer=bias_initializer,
        filters=output_channels,
        kernel_initializer=kernel_initializer,
        kernel_size=(1, 1),
        padding='same',
    )(pred)
    pred = k_layers.BatchNormalization()(pred)
    pred = k_layers.Activation(final_activation)(pred)
    return Model(inputs=inputs, outputs=pred)


def u_net(
        activation='selu',
        bias_initializer='zeros',
        cross_merge=k_layers.Concatenate,
        depth=4,
        filters=16,
        final_activation='selu',
        input_shape=(None, None, None),
        kernel_initializer='glorot_uniform',
        noise_std=0.1,
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
        bias_initializer: (str or Callable)
            Name or instance of a keras.initializers.Initializer.
        cross_merge: tensorflow.keras.layers.Merge
            Layer that merges the cross connections into the second half of the U-Net.
            Common options are Add, Concatenate, Maximum, and Multiply.
        depth: (int)
            Number of levels used in the construction of the restrictive/reconstituting paths.
        filters: (int)
            Number of filters used in convolutions, altered by spatial resampling operations.
        final_activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function
            Final operation of the network, determines the possible range of network outputs.
        input_shape: (tuple[int or None])
            Specifies the dimensions of the input data, does not include the samples dimension.
        kernel_initializer: (str or Callable)
            Name or instance of a keras.initializers.Initializer.
        noise_std: (float)
            Standard deviation of an additive 0-mean Gaussian noise applied to network inputs.
        output_channels: (int)
            Number of output channels/features.

    Returns: (keras.models.Model)
        A compiled and ready-to-use U-Net.
    """
    inputs = k_layers.Input(shape=input_shape)
    pred = k_layers.GaussianNoise(stddev=noise_std)(inputs)

    # Restriction
    crosses = []
    for _ in range(depth):
        pred = k_layers.BatchNormalization()(pred)
        pred = k_layers.Activation(activation)(pred)
        pred = k_layers.Conv2D(
            bias_initializer=bias_initializer,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=(3, 3),
            padding='same',
        )(pred)

        pred = k_layers.BatchNormalization()(pred)
        pred = k_layers.Activation(activation)(pred)
        pred = k_layers.Conv2D(filters, (3, 3), padding='same')(pred)

        crosses.append(pred)

        pred = k_layers.MaxPool2D()(pred)
        filters *= 2

    pred = k_layers.BatchNormalization()(pred)
    pred = k_layers.Activation(activation)(pred)
    pred = k_layers.Conv2D(
        bias_initializer=bias_initializer,
        filters=filters,
        kernel_initializer=kernel_initializer,
        kernel_size=(3, 3),
        padding='same',
    )(pred)

    # Reconstitution
    for cross in crosses[::-1]:
        pred = k_layers.UpSampling2D()(pred)
        filters //= 2
        pred = k_layers.Conv2D(
            bias_initializer=bias_initializer,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=(3, 3),
            padding='same',
        )(pred)

        pred = cross_merge()([pred, cross])
        pred = k_layers.BatchNormalization()(pred)
        pred = k_layers.Activation(activation)(pred)
        pred = k_layers.Conv2D(
            bias_initializer=bias_initializer,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=(3, 3),
            padding='same',
        )(pred)

        pred = k_layers.BatchNormalization()(pred)
        pred = k_layers.Activation(activation)(pred)
        pred = k_layers.Conv2D(
            bias_initializer=bias_initializer,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=(3, 3),
            padding='same',
        )(pred)

    pred = k_layers.Conv2D(
        bias_initializer=bias_initializer,
        filters=output_channels,
        kernel_initializer=kernel_initializer,
        kernel_size=(1, 1),
        padding='same',
    )(pred)
    pred = k_layers.BatchNormalization()(pred)
    pred = k_layers.Activation(final_activation)(pred)
    return Model(inputs=inputs, outputs=pred)


def res_u_net(
        activation='selu',
        bias_initializer='zeros',
        coord_features=False,
        cross_merge=k_layers.Concatenate,
        depth=4,
        filters=16,
        final_activation='selu',
        input_shape=(None, None, None),
        kernel_initializer='glorot_uniform',
        noise_std=0.1,
        output_channels=1,
        residual_merge=k_layers.Add,
):
    """A U-Net with residual blocks at each level.

    Args:
        activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function.
            Applied throughout the network, except for the final activation.
        bias_initializer: (str or Callable)
            Name or instance of a keras.initializers.Initializer.
        coord_features: (bool)
            Adds coordinate feature channels to the input, allowing the network
            to better handle spatially varying relationships.
        cross_merge: tensorflow.keras.layers.Merge
            Layer that merges the cross connections into the second half of the U-Net.
            Common options are Add, Concatenate, Maximum, and Multiply.
        depth: (int)
            Number of levels used in the construction of the restrictive/reconstituting paths.
        filters: (int)
            Number of filters used in convolutions, altered by spatial resampling operations.
        final_activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function
            Final operation of the network, determines the possible range of network outputs.
        input_shape: (tuple[int or None])
            Specifies the dimensions of the input data, does not include the samples dimension.
        kernel_initializer: (str or Callable)
            Name or instance of a keras.initializers.Initializer.
        noise_std: (float)
            Standard deviation of an additive 0-mean Gaussian noise applied to network inputs.
        output_channels: (int)
            Number of output channels/features.
        residual_merge: tensorflow.keras.layers.Merge
            Layer that merges the residual connections of residual blocks.
            Common options are Add, Concatenate, Maximum, and Multiply.

    Returns: (keras.models.Model)
        A compiled and ready-to-use Residual U-Net.
    """
    inputs = k_layers.Input(shape=input_shape)
    pred = k_layers.GaussianNoise(stddev=noise_std)(inputs)

    # TODO: Add back in when CoordinateChannel2D is ported to TF2
    # if coord_features:
    #     pred = layers.CoordinateChannel2D()(pred)

    # Restriction
    crosses = []
    for _ in range(depth):
        pred = layers.ResidualBlock(
            activation=activation,
            bias_initializer=bias_initializer,
            filters=filters,
            kernel_initializer=kernel_initializer,
            merge=residual_merge,
            residual_projection=True,
        )(pred)

        crosses.append(pred)

        pred = k_layers.MaxPool2D()(pred)
        filters *= 2

    pred = layers.ResidualBlock(
        activation=activation,
        bias_initializer=bias_initializer,
        filters=filters,
        kernel_initializer=kernel_initializer,
        merge=residual_merge,
        residual_projection=True,
    )(pred)

    # Reconstitution
    for cross in crosses[::-1]:
        pred = k_layers.UpSampling2D()(pred)
        filters //= 2
        pred = k_layers.Conv2D(filters, (3, 3), padding='same')(pred)

        pred = cross_merge()([pred, cross])
        pred = layers.ResidualBlock(
            activation=activation,
            bias_initializer=bias_initializer,
            filters=filters,
            kernel_initializer=kernel_initializer,
            merge=residual_merge,
            residual_projection=True,
        )(pred)

    pred = k_layers.Conv2D(output_channels, (1, 1))(pred)
    pred = k_layers.BatchNormalization()(pred)
    pred = k_layers.Activation(final_activation)(pred)
    return Model(inputs=inputs, outputs=pred)


def dilated_net(
        activation='selu',
        bias_initializer='zeros',
        depth=3,
        filters=32,
        final_activation='sigmoid',
        input_shape=(None, None, None),
        kernel_initializer='glorot_uniform',
        merge=k_layers.Add,
        noise_std=0.1,
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
        bias_initializer: (str or Callable)
            Name or instance of a keras.initializers.Initializer.
        depth: (int)
            Number of levels used in the construction of the restrictive/reconstituting paths.
        filters: (int)
            Number of filters used in convolutions, altered by spatial resampling operations.
        final_activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function
            Final operation of the network, determines the possible range of network outputs.
        input_shape: (tuple[int or None])
            Specifies the dimensions of the input data, does not include the samples dimension.
        kernel_initializer: (str or Callable)
            Name or instance of a keras.initializers.Initializer.
        merge: (keras.layers.layer)
            Keras layer that merges the input and output branches of a residual block, usually keras.layers.Add.
        noise_std: (float)
            Standard deviation of an additive 0-mean Gaussian noise applied to network inputs.
        output_channels: (int)
            Number of output channels/features.

    Returns: (keras.models.Model)
        A compiled and ready-to-use Residual-U-Net.
    """
    inputs = k_layers.Input(shape=input_shape)
    pred = k_layers.GaussianNoise(stddev=noise_std)(inputs)

    for _ in range(depth):
        pred = layers.DilationBlock(
            activation=activation,
            bias_initializer=bias_initializer,
            filters=filters,
            kernel_initializer=kernel_initializer,
            merge=merge,
        )(pred)

    pred = k_layers.BatchNormalization()(pred)
    pred = k_layers.Activation(activation)(pred)
    pred = k_layers.Conv2D(
        bias_initializer=bias_initializer,
        filters=filters,
        kernel_initializer=kernel_initializer,
        kernel_size=(3, 3),
        padding='same',
    )(pred)

    pred = k_layers.Conv2D(
        filters=output_channels,
        kernel_size=(1, 1),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(pred)
    pred = k_layers.BatchNormalization()(pred)
    pred = k_layers.Activation(final_activation)(pred)
    return Model(inputs=inputs, outputs=pred)


def res_dilated_net(
        activation='selu',
        bias_initializer='zeros',
        depth=3,
        filters=32,
        final_activation='sigmoid',
        input_shape=(None, None, None),
        kernel_initializer='glorot_uniform',
        merge=k_layers.Add,
        noise_std=0.1,
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
        bias_initializer: (str or Callable)
            Name or instance of a keras.initializers.Initializer.
        depth: (int)
            Number of levels used in the construction of the restrictive/reconstituting paths.
        filters: (int)
            Number of filters used in convolutions, altered by spatial resampling operations.
        final_activation: (str or Callable)
            Name of a keras activation function or an instance of a keras/Tensorflow activation function
            Final operation of the network, determines the possible range of network outputs.
        input_shape: (tuple[int or None])
            Specifies the dimensions of the input data, does not include the samples dimension.
        kernel_initializer: (str or Callable)
            Name or instance of a keras.initializers.Initializer.
        merge: (keras.layers.layer)
            Keras layer that merges the input and output branches of a residual block, usually keras.layers.Add.
        noise_std: (float)
            Standard deviation of an additive 0-mean Gaussian noise applied to network inputs.
        output_channels: (int)
            Number of output channels/features.

    Returns: (keras.models.Model)
        A compiled and ready-to-use Residual-U-Net.
    """
    inputs = k_layers.Input(shape=input_shape)
    pred = k_layers.GaussianNoise(stddev=noise_std)(inputs)

    pred = k_layers.Conv2D(filters, (1, 1))(pred)
    for _ in range(depth):
        pred = layers.DilationBlock(
            activation=activation,
            bias_initializer=bias_initializer,
            filters=filters,
            kernel_initializer=kernel_initializer,
            merge=merge,
            skip_connection=True,
        )(pred)

    pred = k_layers.BatchNormalization()(pred)
    pred = k_layers.Activation(activation)(pred)
    pred = k_layers.Conv2D(
        bias_initializer=bias_initializer,
        filters=filters,
        kernel_initializer=kernel_initializer,
        kernel_size=(3, 3),
        padding='same',
    )(pred)

    pred = k_layers.Conv2D(
        bias_initializer=bias_initializer,
        filters=output_channels,
        kernel_initializer=kernel_initializer,
        kernel_size=(1, 1),
    )(pred)
    pred = k_layers.BatchNormalization()(pred)
    pred = k_layers.Activation(final_activation)(pred)
    return Model(inputs=inputs, outputs=pred)

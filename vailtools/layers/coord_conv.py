"""
Borrowed from https://github.com/titu1994/keras-coordconv on 2020/04/05
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputSpec, Layer

from ..utils import register_custom_objects


class CoordinateChannel(Layer):
    """
    Adds coordinate channels to the input tensor.

    Args:
        rank: int
            Number of spatial dimensions in the input data tensor, e.g. 2 for 2D convolution.
        use_radius: bool
            Flag to toggle addition of a radius channel in addition to the coordinate channels.
        data_format: str, options = {"channels_last", "channels_first", None}
            Ordering of the dimensions in the inputs.
            "channels_last"  -> (batch, ..., channels)
            "channels_first" -> (batch, channels, ...)
            Defaults to the Keras image_data_format set in ~/.keras/keras.json.
            If you never set it, then it will be "channels_last".

    Input shape:
        ND tensor with shape (samples, channels, *) if data_format is "channels_first"
        or
        ND tensor with shape (samples, *, channels) if data_format is "channels_last"

    Output shape:
        ND tensor with shape (samples, channels + N - 2, *) if data_format is "channels_first"
        or
        ND tensor with shape (samples, *, channels + N - 2) if data_format is "channels_last"

    References:
        - [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)
    """

    def __init__(self, rank, use_radius=False, data_format=None, **kwargs):
        super().__init__(**kwargs, dynamic=True)

        if data_format not in [None, "channels_first", "channels_last"]:
            raise ValueError(
                f'data_format must be  one of {{"channels_last", "channels_first", None}}, received {data_format}!'
            )

        self.rank = rank
        self.use_radius = use_radius
        self.data_format = data_format or K.image_data_format()
        self.axis = 1 if self.data_format == "channels_first" else -1

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        if len(input_shape) < 2:
            raise ValueError(
                f"Inputs must be at least rank 3 tensors, received rank {len(input_shape)} input shape!"
            )
        input_dim = input_shape[self.axis]

        self.input_spec = InputSpec(min_ndim=self.rank + 2, axes={self.axis: input_dim})
        self.built = True

    def call(self, inputs, training=None, mask=None):
        input_shape = K.shape(inputs)
        if self.rank == 1:
            if self.data_format == "channels_first":
                inputs = K.permute_dimensions(inputs, [0, 2, 1])
                input_shape = K.shape(inputs)
            input_shape = [input_shape[i] for i in range(3)]
            batch_shape, dim, channels = input_shape

            channels_1 = K.arange(dim, dtype=K.floatx())
            channels_1 = K.reshape(channels_1, (1, -1, 1))
            channels_1 = tf.broadcast_to(channels_1, (batch_shape, dim, 1))
            channels_1 = channels_1 / K.cast(dim - 1, K.floatx())
            channels_1 = (channels_1 * 2) - 1

            outputs = K.concatenate([inputs, channels_1])

            if self.use_radius:
                rr = K.sqrt(K.square(channels_1))
                outputs = K.concatenate([outputs, rr])

            if self.data_format == "channels_first":
                outputs = K.permute_dimensions(outputs, [0, 2, 1])

        elif self.rank == 2:
            if self.data_format == "channels_first":
                inputs = K.permute_dimensions(inputs, [0, 2, 3, 1])
                input_shape = K.shape(inputs)
            input_shape = [input_shape[i] for i in range(4)]
            batch_shape, dim1, dim2, channels = input_shape

            channels_1 = K.arange(dim1, dtype=K.floatx())
            channels_1 = K.reshape(channels_1, (1, -1, 1, 1))
            channels_1 = tf.broadcast_to(channels_1, (batch_shape, dim1, dim2, 1))
            channels_1 = channels_1 / K.cast(dim1 - 1, K.floatx())
            channels_1 = 2 * channels_1 - 1

            channels_2 = K.arange(dim2, dtype=K.floatx())
            channels_2 = K.reshape(channels_2, (1, 1, -1, 1))
            channels_2 = tf.broadcast_to(channels_2, (batch_shape, dim1, dim2, 1))
            channels_2 = channels_2 / K.cast(dim2 - 1, K.floatx())
            channels_2 = 2 * channels_2 - 1

            outputs = K.concatenate([inputs, channels_1, channels_2])

            if self.use_radius:
                rr = K.sqrt(K.square(channels_1) + K.square(channels_2))
                outputs = K.concatenate([outputs, rr])

            if self.data_format == "channels_first":
                outputs = K.permute_dimensions(outputs, [0, 3, 1, 2])

        elif self.rank == 3:
            if self.data_format == "channels_first":
                inputs = K.permute_dimensions(inputs, [0, 2, 3, 4, 1])
                input_shape = K.shape(inputs)

            input_shape = [input_shape[i] for i in range(5)]
            batch_shape, dim1, dim2, dim3, channels = input_shape

            channels_1 = K.arange(dim1, dtype=K.floatx())
            channels_1 = K.reshape(channels_1, (1, -1, 1, 1, 1))
            channels_1 = tf.broadcast_to(channels_1, (batch_shape, dim1, dim2, dim3, 1))
            channels_1 = channels_1 / K.cast(dim1 - 1, K.floatx())
            channels_1 = 2 * channels_1 - 1

            channels_2 = K.arange(dim2, dtype=K.floatx())
            channels_2 = K.reshape(channels_2, (1, 1, -1, 1, 1))
            channels_2 = tf.broadcast_to(channels_2, (batch_shape, dim1, dim2, dim3, 1))
            channels_2 = channels_2 / K.cast(dim2 - 1, K.floatx())
            channels_2 = 2 * channels_2 - 1

            channels_3 = K.arange(dim3, dtype=K.floatx())
            channels_3 = K.reshape(channels_3, (1, 1, 1, -1, 1))
            channels_3 = tf.broadcast_to(channels_3, (batch_shape, dim1, dim2, dim3, 1))
            channels_3 = channels_3 / K.cast(dim3 - 1, K.floatx())
            channels_3 = 2 * channels_3 - 1

            outputs = K.concatenate([inputs, channels_1, channels_2, channels_3])

            if self.use_radius:
                rr = K.sqrt(
                    K.square(channels_1) + K.square(channels_2) + K.square(channels_3)
                )
                outputs = K.concatenate([outputs, rr])

            if self.data_format == "channels_first":
                outputs = K.permute_dimensions(outputs, [0, 4, 1, 2, 3])
        else:
            raise ValueError(
                f"Only ranks 1, 2, and 3 are supported, received rank = {self.rank}!"
            )

        return outputs

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[self.axis]

        channel_count = self.rank + self.use_radius

        output_shape = list(input_shape)
        output_shape[self.axis] = input_shape[self.axis] + channel_count
        return tuple(output_shape)

    def get_config(self):
        config = {
            "rank": self.rank,
            "use_radius": self.use_radius,
            "data_format": self.data_format,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class CoordinateChannel1D(CoordinateChannel):
    """
    Adds coordinate channels to the input tensor.

    Args:
        use_radius: bool
            Flag to toggle addition of a radius channel in addition to the coordinate channels.
        data_format: str, options = {"channels_last", "channels_first", None}
            Ordering of the dimensions in the inputs.
            "channels_last"  -> (batch, ..., channels)
            "channels_first" -> (batch, channels, ...)
            Defaults to the Keras image_data_format set in ~/.keras/keras.json.
            If you never set it, then it will be "channels_last".

    Input shape:
        3D tensor with shape (samples, channels, *) if data_format is "channels_first"
        or
        3D tensor with shape (samples, *, channels) if data_format is "channels_last"

    Output shape:
        3D tensor with shape (samples, channels + N - 2, *) if data_format is "channels_first"
        or
        3D tensor with shape (samples, *, channels + N - 2) if data_format is "channels_last"

    References:
        - [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)
    """

    def __init__(self, use_radius=False, data_format=None, **kwargs):
        super().__init__(
            rank=1, use_radius=use_radius, data_format=data_format, **kwargs
        )

    def get_config(self):
        config = super().get_config()
        config.pop("rank")
        return config


class CoordinateChannel2D(CoordinateChannel):
    """
    Adds coordinate channels to the input tensor.

    Args:
        use_radius: bool
            Flag to toggle addition of a radius channel in addition to the coordinate channels.
        data_format: str, options = {"channels_last", "channels_first", None}
            Ordering of the dimensions in the inputs.
            "channels_last"  -> (batch, ..., channels)
            "channels_first" -> (batch, channels, ...)
            Defaults to the Keras image_data_format set in ~/.keras/keras.json.
            If you never set it, then it will be "channels_last".

    Input shape:
        4D tensor with shape (samples, channels, *) if data_format is "channels_first"
        or
        4D tensor with shape (samples, *, channels) if data_format is "channels_last"

    Output shape:
        4D tensor with shape (samples, channels + N - 2, *) if data_format is "channels_first"
        or
        4D tensor with shape (samples, *, channels + N - 2) if data_format is "channels_last"

    References:
        - [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)
    """

    def __init__(self, use_radius=False, data_format=None, **kwargs):
        super().__init__(
            rank=2, use_radius=use_radius, data_format=data_format, **kwargs
        )

    def get_config(self):
        config = super().get_config()
        config.pop("rank")
        return config


class CoordinateChannel3D(CoordinateChannel):
    """
    Adds coordinate channels to the input tensor.

    Args:
        use_radius: bool
            Flag to toggle addition of a radius channel in addition to the coordinate channels.
        data_format: str, options = {"channels_last", "channels_first", None}
            Ordering of the dimensions in the inputs.
            "channels_last"  -> (batch, ..., channels)
            "channels_first" -> (batch, channels, ...)
            Defaults to the Keras image_data_format set in ~/.keras/keras.json.
            If you never set it, then it will be "channels_last".

    Input shape:
        5D tensor with shape (samples, channels, *) if data_format is "channels_first"
        or
        5D tensor with shape (samples, *, channels) if data_format is "channels_last"

    Output shape:
        5D tensor with shape (samples, channels + N - 2, *) if data_format is "channels_first"
        or
        5D tensor with shape (samples, *, channels + N - 2) if data_format is "channels_last"

    References:
        - [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)
    """

    def __init__(self, use_radius=False, data_format=None, **kwargs):
        super().__init__(
            rank=3, use_radius=use_radius, data_format=data_format, **kwargs
        )

    def get_config(self):
        config = super().get_config()
        config.pop("rank")
        return config


register_custom_objects([CoordinateChannel1D, CoordinateChannel2D, CoordinateChannel3D])

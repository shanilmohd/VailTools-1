"""
Implementations of the Fire module discussed in https://arxiv.org/abs/1602.07360
for sequence data, image data, and volumetric data.
"""

from tensorflow.keras import layers

from ..utils import register_custom_objects


class FireBlock1D(layers.Layer):
    """
    Accepts sequence data, i.e. dimensions (batch, time, features).

    Reference:
        https://arxiv.org/abs/1602.07360
    """

    def __init__(
        self,
        activation="relu",
        bias_initializer="zeros",
        e1_filters=None,
        e3_filters=None,
        filters=16,
        kernel_initializer="glorot_uniform",
        kernel_size=3,
        merge=layers.Concatenate,
        padding="same",
        s1_filters=None,
        **kwargs,
    ):
        """
        Args:
            activation: (str or Callable)
                Name or instance of a keras activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            e1_filters: (None or int)
                Number of filters used in the 1x1 expand convolution.
            e3_filters: (None or int)
                Number of filters used in the 3x3 expand convolution.
            filters: (None or int)
                Number of filters used in the expand convolutions.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            kernel_size: (tuple[int] or int)
                Convolution filter dimensions.
            merge: (keras.layers.Layer)
                Layer used to merge branches of computation.
            padding: (str)
                Convolution padding strategy.
            s1_filters: (None or int)
                Number of filters used in the 1x1 squeeze convolution.
            **kwargs:
        """
        super().__init__(**kwargs)
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.e1_filters = e1_filters or filters
        self.e3_filters = e3_filters or filters
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.padding = padding
        self.s1_filters = s1_filters or filters // 4

        self.squeeze = layers.Conv1D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.s1_filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=1,
            padding=self.padding,
        )
        self.expand_1 = layers.Conv1D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.e1_filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=1,
            padding=self.padding,
        )
        self.expand_3 = layers.Conv1D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.e3_filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.merge = merge()

    def call(self, inputs, **kwargs):
        pred = self.squeeze(inputs)
        return self.merge([self.expand_1(pred), self.expand_3(pred)])


class FireBlock2D(layers.Layer):
    """
    Accepts image data, i.e. dimensions (batch, width, height, features).

    Reference:
        https://arxiv.org/abs/1602.07360
    """

    def __init__(
        self,
        activation="relu",
        bias_initializer="zeros",
        e1_filters=None,
        e3_filters=None,
        filters=16,
        kernel_initializer="glorot_uniform",
        kernel_size=(3, 3),
        merge=layers.Concatenate,
        padding="same",
        s1_filters=None,
        **kwargs,
    ):
        """
        Args:
            activation: (str or Callable)
                Name or instance of a keras activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            e1_filters: (None or int)
                Number of filters used in the 1x1 expand convolution.
            e3_filters: (None or int)
                Number of filters used in the 3x3 expand convolution.
            filters: (None or int)
                Number of filters used in the expand convolutions.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            kernel_size: (tuple[int] or int)
                Convolution filter dimensions.
            merge: (keras.layers.Layer)
                Layer used to merge branches of computation.
            padding: (str)
                Convolution padding strategy.
            s1_filters: (None or int)
                Number of filters used in the 1x1 squeeze convolution.
            **kwargs:
        """
        super().__init__(**kwargs)
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.e1_filters = e1_filters or filters
        self.e3_filters = e3_filters or filters
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.padding = padding
        self.s1_filters = s1_filters or filters // 4

        self.squeeze = layers.Conv2D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.s1_filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=(1, 1),
            padding=self.padding,
        )
        self.expand_1 = layers.Conv2D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.e1_filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=(1, 1),
            padding=self.padding,
        )
        self.expand_3 = layers.Conv2D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.e3_filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.merge = merge()

    def call(self, inputs, **kwargs):
        pred = self.squeeze(inputs)
        return self.merge([self.expand_1(pred), self.expand_3(pred)])


class FireBlock3D(layers.Layer):
    """
    Accepts volumetric data, i.e. dimensions (batch, width, height, depth, features).

    Reference:
        https://arxiv.org/abs/1602.07360
    """

    def __init__(
        self,
        activation="relu",
        bias_initializer="zeros",
        e1_filters=None,
        e3_filters=None,
        filters=16,
        kernel_initializer="glorot_uniform",
        kernel_size=(3, 3, 3),
        merge=layers.Concatenate,
        padding="same",
        s1_filters=None,
        **kwargs,
    ):
        """
        Args:
            activation: (str or Callable)
                Name or instance of a keras activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            e1_filters: (None or int)
                Number of filters used in the 1x1 expand convolution.
            e3_filters: (None or int)
                Number of filters used in the 3x3 expand convolution.
            filters: (None or int)
                Number of filters used in the expand convolutions.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            kernel_size: (tuple[int] or int)
                Convolution filter dimensions.
            merge: (keras.layers.Layer)
                Layer used to merge branches of computation.
            padding: (str)
                Convolution padding strategy.
            s1_filters: (None or int)
                Number of filters used in the 1x1 squeeze convolution.
            **kwargs:
        """
        super().__init__(**kwargs)
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.e1_filters = e1_filters or filters
        self.e3_filters = e3_filters or filters
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.padding = padding
        self.s1_filters = s1_filters or filters // 4

        self.squeeze = layers.Conv3D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.s1_filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=(1, 1, 1),
            padding=self.padding,
        )
        self.expand_1 = layers.Conv3D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.e1_filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=(1, 1, 1),
            padding=self.padding,
        )
        self.expand_3 = layers.Conv3D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.e3_filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.merge = merge()

    def call(self, inputs, **kwargs):
        pred = self.squeeze(inputs)
        return self.merge([self.expand_1(pred), self.expand_3(pred)])


register_custom_objects(
    [FireBlock1D, FireBlock2D, FireBlock3D,]
)

from tensorflow.keras import layers

from ..utils import register_custom_objects


class FireModule(layers.Layer):
    """
    Implements the Fire module discussed in https://arxiv.org/abs/1602.07360.
    Can be specialized to operate on sequence, image, or volumetric data by
    passing an appropriate primary_layer.
    """

    def __init__(
        self,
        primary_layer,
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

        self.squeeze = primary_layer(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.s1_filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=1,
            padding=self.padding,
        )
        self.expand_1 = primary_layer(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.e1_filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=1,
            padding=self.padding,
        )
        self.expand_3 = primary_layer(
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


class Fire1D(FireModule):
    """
    Implements the Fire module discussed in https://arxiv.org/abs/1602.07360.
    Accepts sequence data, i.e. dimensions (batch size, time, features).
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
        super().__init__(
            layers.Conv1D,
            activation=activation,
            bias_initializer=bias_initializer,
            e1_filters=e1_filters,
            e3_filters=e3_filters,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=kernel_size,
            merge=merge,
            padding=padding,
            s1_filters=s1_filters,
            **kwargs,
        )


class Fire2D(FireModule):
    """
    Implements the Fire module discussed in https://arxiv.org/abs/1602.07360.
    Accepts image data, i.e. dimensions (batch size, width, height, features).
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
        super().__init__(
            layers.Conv2D,
            activation=activation,
            bias_initializer=bias_initializer,
            e1_filters=e1_filters,
            e3_filters=e3_filters,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=kernel_size,
            merge=merge,
            padding=padding,
            s1_filters=s1_filters,
            **kwargs,
        )


class Fire3D(FireModule):
    """
    Implements the Fire module discussed in https://arxiv.org/abs/1602.07360.
    Accepts volumetric data, i.e. dimensions (batch size, width, height, depth, features).
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
        super().__init__(
            layers.Conv3D,
            activation=activation,
            bias_initializer=bias_initializer,
            e1_filters=e1_filters,
            e3_filters=e3_filters,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=kernel_size,
            merge=merge,
            padding=padding,
            s1_filters=s1_filters,
            **kwargs,
        )


register_custom_objects([Fire1D, Fire2D, Fire3D, FireModule])

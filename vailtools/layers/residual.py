from tensorflow.keras import layers

from ..utils import register_custom_objects
from . import squeeze_excite


class ResidualBlock(layers.Layer):
    """
    Implements the simple residual block discussed in:
        https://arxiv.org/abs/1512.03385

    Squeeze and Excitation module can be enabled:
        https://arxiv.org/abs/1709.01507
    """

    def __init__(
        self,
        spatial_layer,
        channel_layer,
        activation="relu",
        bias_initializer="zeros",
        filters=16,
        kernel_initializer="glorot_uniform",
        kernel_size=3,
        merge=layers.Add,
        padding="same",
        residual_projection=True,
        se_reduce_factor=4,
        squeeze_and_excite=False,
        **kwargs,
    ):
        """
        Args:
            activation: (str or Callable)
                Name or instance of a keras activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            filters: (int)
                Number of filters per convolution.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            kernel_size: (tuple[int] or int)
                Convolution filter dimensions.
            merge: (keras.layers.Layer)
                Layer used to merge branches of computation.
            padding: (str)
                Convolution padding strategy.
            residual_projection: (bool)
                Toggles the use a linear projection on the residual connection.
            se_reduce_factor: (int)
                Reduction ratio passed to SqueezeExcite2D.
            squeeze_and_excite: (bool)
                Toggles the use of a SE block between the convolutions and residual merge.
            **kwargs:
        """
        super().__init__(**kwargs)
        self.spatial_layer = spatial_layer
        self.channel_layer = channel_layer

        self.activation = activation
        self.bias_initializer = bias_initializer
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.padding = padding
        self.residual_projection = residual_projection
        self.se_reduce_factor = se_reduce_factor
        self.squeeze_and_excite = squeeze_and_excite

        self.spatial_layer_1 = self.spatial_layer(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.spatial_layer_2 = self.spatial_layer(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )

        if self.squeeze_and_excite:
            self.channel_layer_1 = self.se(
                reduce_factor=self.se_reduce_factor, width=self.filters,
            )

        if self.residual_projection:
            self.projection = self.spatial_layer(
                filters=self.filters,
                kernel_size=1,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
            )
        self.merge = merge()

    def call(self, inputs, **kwargs):
        pred = self.spatial_layer_1(inputs)
        pred = self.spatial_layer_2(pred)
        if self.squeeze_and_excite:
            pred = self.channel_layer_1(pred)
        if self.residual_projection:
            inputs = self.projection(inputs)
        return self.merge([inputs, pred])


class Residual1D(ResidualBlock):
    """
    Implements the simple residual block discussed in:
        https://arxiv.org/abs/1512.03385

    Squeeze and Excitation module can be enabled:
        https://arxiv.org/abs/1709.01507
    """

    def __init__(
        self,
        activation="relu",
        bias_initializer="zeros",
        filters=16,
        kernel_initializer="glorot_uniform",
        kernel_size=3,
        merge=layers.Add,
        padding="same",
        residual_projection=True,
        se_reduce_factor=4,
        squeeze_and_excite=False,
        **kwargs,
    ):
        """
        Args:
            activation: (str or Callable)
                Name or instance of a keras activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            filters: (int)
                Number of filters per convolution.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            kernel_size: (tuple[int] or int)
                Convolution filter dimensions.
            merge: (keras.layers.Layer)
                Layer used to merge branches of computation.
            padding: (str)
                Convolution padding strategy.
            residual_projection: (bool)
                Toggles the use a linear projection on the residual connection.
            se_reduce_factor: (int)
                Reduction ratio passed to SqueezeExcite2D.
            squeeze_and_excite: (bool)
                Toggles the use of a SE block between the convolutions and residual merge.
            **kwargs:
        """
        super().__init__(
            layers.Conv1D,
            squeeze_excite.SqueezeExcite1D,
            activation=activation,
            bias_initializer=bias_initializer,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=kernel_size,
            merge=merge,
            padding=padding,
            residual_projection=residual_projection,
            se_reduce_factor=se_reduce_factor,
            squeeze_and_excite=squeeze_and_excite,
            **kwargs,
        )


class Residual2D(ResidualBlock):
    """
    Implements the simple residual block discussed in:
        https://arxiv.org/abs/1512.03385

    Squeeze and Excitation module can be enabled:
        https://arxiv.org/abs/1709.01507
    """

    def __init__(
        self,
        activation="relu",
        bias_initializer="zeros",
        filters=16,
        kernel_initializer="glorot_uniform",
        kernel_size=3,
        merge=layers.Add,
        padding="same",
        residual_projection=True,
        se_reduce_factor=4,
        squeeze_and_excite=False,
        **kwargs,
    ):
        """
        Args:
            activation: (str or Callable)
                Name or instance of a keras activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            filters: (int)
                Number of filters per convolution.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            kernel_size: (tuple[int] or int)
                Convolution filter dimensions.
            merge: (keras.layers.Layer)
                Layer used to merge branches of computation.
            padding: (str)
                Convolution padding strategy.
            residual_projection: (bool)
                Toggles the use a linear projection on the residual connection.
            se_reduce_factor: (int)
                Reduction ratio passed to SqueezeExcite2D.
            squeeze_and_excite: (bool)
                Toggles the use of a SE block between the convolutions and residual merge.
            **kwargs:
        """
        super().__init__(
            layers.Conv2D,
            squeeze_excite.SqueezeExcite2D,
            activation=activation,
            bias_initializer=bias_initializer,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=kernel_size,
            merge=merge,
            padding=padding,
            residual_projection=residual_projection,
            se_reduce_factor=se_reduce_factor,
            squeeze_and_excite=squeeze_and_excite,
            **kwargs,
        )


class Residual3D(ResidualBlock):
    """
    Implements the simple residual block discussed in:
        https://arxiv.org/abs/1512.03385

    Squeeze and Excitation module can be enabled:
        https://arxiv.org/abs/1709.01507
    """

    def __init__(
        self,
        activation="relu",
        bias_initializer="zeros",
        filters=16,
        kernel_initializer="glorot_uniform",
        kernel_size=3,
        merge=layers.Add,
        padding="same",
        residual_projection=True,
        se_reduce_factor=4,
        squeeze_and_excite=False,
        **kwargs,
    ):
        """
        Args:
            activation: (str or Callable)
                Name or instance of a keras activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            filters: (int)
                Number of filters per convolution.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            kernel_size: (tuple[int] or int)
                Convolution filter dimensions.
            merge: (keras.layers.Layer)
                Layer used to merge branches of computation.
            padding: (str)
                Convolution padding strategy.
            residual_projection: (bool)
                Toggles the use a linear projection on the residual connection.
            se_reduce_factor: (int)
                Reduction ratio passed to SqueezeExcite2D.
            squeeze_and_excite: (bool)
                Toggles the use of a SE block between the convolutions and residual merge.
            **kwargs:
        """
        super().__init__(
            layers.Conv3D,
            squeeze_excite.SqueezeExcite3D,
            activation=activation,
            bias_initializer=bias_initializer,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=kernel_size,
            merge=merge,
            padding=padding,
            residual_projection=residual_projection,
            se_reduce_factor=se_reduce_factor,
            squeeze_and_excite=squeeze_and_excite,
            **kwargs,
        )


class ResidualBottleneckBlock(layers.Layer):
    """
    Implements the residual bottleneck block discussed in:
        https://arxiv.org/abs/1512.03385

    Squeeze and Excitation module can be enabled:
        https://arxiv.org/abs/1709.01507
    """

    def __init__(
        self,
        spatial_layer,
        channel_layer,
        activation="relu",
        bias_initializer="zeros",
        filters=16,
        kernel_initializer="glorot_uniform",
        kernel_size=3,
        merge=layers.Add,
        neck_filters=None,
        padding="same",
        residual_projection=True,
        se_reduce_factor=4,
        squeeze_and_excite=False,
        **kwargs,
    ):
        """
        Args:
            activation: (str or Callable)
                Name or instance of a keras activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            filters: (int)
                Number of filters per convolution following the bottleneck.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            kernel_size: (tuple[int] or int)
                Convolution filter dimensions.
            merge: (keras.layers.Layer)
                Layer used to merge branches of computation.
            neck_filters: (int)
                Number of filters per convolution in the bottleneck.
            padding: (str)
                Convolution padding strategy.
            residual_projection: (bool)
                Toggles the use a linear projection on the residual connection.
            se_reduce_factor: (int)
                Reduction ratio passed to SqueezeExcite2D.
            squeeze_and_excite: (bool)
                Toggles the use of a SE block between the convolutions and residual merge.
            **kwargs:
        """
        super().__init__(**kwargs)
        self.spatial_layer = spatial_layer
        self.channel_layer = channel_layer

        self.activation = activation
        self.bias_initializer = bias_initializer
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.neck_filters = neck_filters or max(filters // 4, 1)
        self.padding = padding
        self.residual_projection = residual_projection
        self.se_reduce_factor = se_reduce_factor
        self.squeeze_and_excite = squeeze_and_excite

        self.spatial_layer_1 = self.spatial_layer(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.neck_filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=1,
        )

        self.spatial_layer_2 = self.spatial_layer(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.neck_filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.spatial_layer_3 = self.spatial_layer(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=1,
        )

        if self.squeeze_and_excite:
            self.channel_layer_1 = self.channel_layer(
                reduce_factor=self.excite_factor, width=self.filters,
            )

        if self.residual_projection:
            self.projection = self.spatial_layer(
                bias_initializer=self.bias_initializer,
                filters=self.filters,
                kernel_initializer=self.kernel_initializer,
                kernel_size=1,
            )
        self.merge = merge()

    def call(self, inputs, **kwargs):
        pred = self.spatial_layer_1(inputs)
        pred = self.spatial_layer_2(pred)
        pred = self.spatial_layer_3(pred)
        if self.squeeze_and_excite:
            pred = self.channel_layer_1(pred)
        if self.residual_projection:
            inputs = self.projection(inputs)
        return self.merge([inputs, pred])


class ResidualBottleneck1D(ResidualBottleneckBlock):
    """
    Implements the residual bottleneck block discussed in:
        https://arxiv.org/abs/1512.03385

    Squeeze and Excitation module can be enabled:
        https://arxiv.org/abs/1709.01507
    """

    def __init__(
        self,
        activation="relu",
        bias_initializer="zeros",
        filters=16,
        kernel_initializer="glorot_uniform",
        kernel_size=3,
        merge=layers.Add,
        neck_filters=None,
        padding="same",
        residual_projection=True,
        se_reduce_factor=4,
        squeeze_and_excite=False,
        **kwargs,
    ):
        """
        Args:
            activation: (str or Callable)
                Name or instance of a keras activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            filters: (int)
                Number of filters per convolution following the bottleneck.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            kernel_size: (tuple[int] or int)
                Convolution filter dimensions.
            merge: (keras.layers.Layer)
                Layer used to merge branches of computation.
            neck_filters: (int)
                Number of filters per convolution in the bottleneck.
            padding: (str)
                Convolution padding strategy.
            residual_projection: (bool)
                Toggles the use a linear projection on the residual connection.
            se_reduce_factor: (int)
                Reduction ratio passed to SqueezeExcite2D.
            squeeze_and_excite: (bool)
                Toggles the use of a SE block between the convolutions and residual merge.
            **kwargs:
        """
        super().__init__(
            layers.Conv1D,
            squeeze_excite.SqueezeExcite1D,
            activation=activation,
            bias_initializer=bias_initializer,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=kernel_size,
            merge=merge,
            neck_filters=neck_filters,
            padding=padding,
            residual_projection=residual_projection,
            se_reduce_factor=se_reduce_factor,
            squeeze_and_excite=squeeze_and_excite,
            **kwargs,
        )


class ResidualBottleneck2D(ResidualBottleneckBlock):
    """
    Implements the residual bottleneck block discussed in:
        https://arxiv.org/abs/1512.03385

    Squeeze and Excitation module can be enabled:
        https://arxiv.org/abs/1709.01507
    """

    def __init__(
        self,
        activation="relu",
        bias_initializer="zeros",
        filters=16,
        kernel_initializer="glorot_uniform",
        kernel_size=3,
        merge=layers.Add,
        neck_filters=None,
        padding="same",
        residual_projection=True,
        se_reduce_factor=4,
        squeeze_and_excite=False,
        **kwargs,
    ):
        """
        Args:
            activation: (str or Callable)
                Name or instance of a keras activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            filters: (int)
                Number of filters per convolution following the bottleneck.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            kernel_size: (tuple[int] or int)
                Convolution filter dimensions.
            merge: (keras.layers.Layer)
                Layer used to merge branches of computation.
            neck_filters: (int)
                Number of filters per convolution in the bottleneck.
            padding: (str)
                Convolution padding strategy.
            residual_projection: (bool)
                Toggles the use a linear projection on the residual connection.
            se_reduce_factor: (int)
                Reduction ratio passed to SqueezeExcite2D.
            squeeze_and_excite: (bool)
                Toggles the use of a SE block between the convolutions and residual merge.
            **kwargs:
        """
        super().__init__(
            layers.Conv2D,
            squeeze_excite.SqueezeExcite2D,
            activation=activation,
            bias_initializer=bias_initializer,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=kernel_size,
            merge=merge,
            neck_filters=neck_filters,
            padding=padding,
            residual_projection=residual_projection,
            se_reduce_factor=se_reduce_factor,
            squeeze_and_excite=squeeze_and_excite,
            **kwargs,
        )


class ResidualBottleneck3D(ResidualBottleneckBlock):
    """
    Implements the residual bottleneck block discussed in:
        https://arxiv.org/abs/1512.03385

    Squeeze and Excitation module can be enabled:
        https://arxiv.org/abs/1709.01507
    """

    def __init__(
        self,
        activation="relu",
        bias_initializer="zeros",
        filters=16,
        kernel_initializer="glorot_uniform",
        kernel_size=3,
        merge=layers.Add,
        neck_filters=None,
        padding="same",
        residual_projection=True,
        se_reduce_factor=4,
        squeeze_and_excite=False,
        **kwargs,
    ):
        """
        Args:
            activation: (str or Callable)
                Name or instance of a keras activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            filters: (int)
                Number of filters per convolution following the bottleneck.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            kernel_size: (tuple[int] or int)
                Convolution filter dimensions.
            merge: (keras.layers.Layer)
                Layer used to merge branches of computation.
            neck_filters: (int)
                Number of filters per convolution in the bottleneck.
            padding: (str)
                Convolution padding strategy.
            residual_projection: (bool)
                Toggles the use a linear projection on the residual connection.
            se_reduce_factor: (int)
                Reduction ratio passed to SqueezeExcite2D.
            squeeze_and_excite: (bool)
                Toggles the use of a SE block between the convolutions and residual merge.
            **kwargs:
        """
        super().__init__(
            layers.Conv3D,
            squeeze_excite.SqueezeExcite3D,
            activation=activation,
            bias_initializer=bias_initializer,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=kernel_size,
            merge=merge,
            neck_filters=neck_filters,
            padding=padding,
            residual_projection=residual_projection,
            se_reduce_factor=se_reduce_factor,
            squeeze_and_excite=squeeze_and_excite,
            **kwargs,
        )


register_custom_objects(
    [
        Residual1D,
        Residual2D,
        Residual3D,
        ResidualBlock,
        ResidualBottleneck1D,
        ResidualBottleneck2D,
        ResidualBottleneck3D,
        ResidualBottleneckBlock,
    ]
)

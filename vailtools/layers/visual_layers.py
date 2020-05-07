import numpy as np
from tensorflow.keras import layers

from ..utils import register_custom_objects


class DenseBlock(layers.Layer):
    """
    Implements the densely connected convolution block discussed in:
        https://arxiv.org/abs/1608.06993
    """

    def __init__(
        self,
        activation="relu",
        bias_initializer="zeros",
        depth=2,
        filters=16,
        kernel_initializer="glorot_uniform",
        kernel_size=(3, 3),
        merge=layers.Concatenate,
        padding="same",
        **kwargs,
    ):
        """
        Args:
            activation: (str or Callable)
                Name or instance of a keras activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            depth: (int)
                Number of convolutions used in block construction.
            filters: (int)
                Number of filters per convolution.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            kernel_size: (tuple[int] or int)
                Dimensions of the convolution filters.
            merge: (keras.layers.Layer)
                Layer used to merge branches of computation.
            padding: (str)
                Convolution padding strategy.
        """
        super().__init__(**kwargs)
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.depth = max(depth, 1)
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.merge = merge()
        self.padding = padding

        self.layers = [
            layers.Conv2D(
                activation=self.activation,
                bias_initializer=self.bias_initializer,
                filters=self.filters,
                kernel_initializer=self.kernel_initializer,
                kernel_size=self.kernel_size,
                padding=self.padding,
            )
            for _ in range(self.depth)
        ]

    def call(self, inputs, **kwargs):
        output = inputs
        for layer in self.layers:
            output = self.merge([output, layer(output)])
        return output


class DilationBlock(layers.Layer):
    """
    Implements a block of exponentially dilated convolutions.
    Inspired by architectures using dilated convolutions such as:
        https://arxiv.org/abs/1511.07122
        https://arxiv.org/abs/1710.02224
        https://arxiv.org/abs/1802.10062
    """

    def __init__(
        self,
        activation="selu",
        bias_initializer="zeros",
        dilations=None,
        filters=16,
        kernel_initializer="glorot_uniform",
        kernel_size=(3, 3),
        merge=layers.Add,
        padding="same",
        skip_connection=False,
        **kwargs,
    ):
        """
        Args:
            activation: (str or Callable)
                Name or instance of a keras activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            dilations: (Iterable[int])
                Convolution dilation rate.
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
        """
        super().__init__(**kwargs)
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.dilations = dilations or tuple(2 ** x for x in range(4))
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.merge = merge()
        self.padding = padding
        self.skip_connection = skip_connection
        self.layers = [
            layers.Conv2D(
                activation=self.activation,
                bias_initializer=self.bias_initializer,
                dilation_rate=d,
                filters=self.filters,
                kernel_initializer=self.kernel_initializer,
                kernel_size=self.kernel_size,
                padding=self.padding,
            )
            for d in self.dilations
        ]

    def call(self, inputs, **kwargs):
        preds = [layer(inputs) for layer in self.layers]
        if self.skip_connection:
            preds = [inputs] + preds
        return self.merge(preds)


class FractalBlock(layers.Layer):
    """
    Implements the fractal convolution block described in:
        https://arxiv.org/abs/1605.07648
    """

    def __init__(
        self,
        activation="relu",
        bias_initializer="zeros",
        depth=4,
        filters=16,
        kernel_initializer="glorot_uniform",
        kernel_size=(3, 3),
        merge=layers.Concatenate,
        padding="same",
        **kwargs,
    ):
        """
        Args:
            activation: (str or Callable)
                Name or instance of a keras activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            depth: (int)
                Fractal block rank.
                Rank 1 is a residual block.
                Rank 2 is a residual block of residual blocks.
                See Figure 1 of the paper for more details.
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
        """
        super().__init__(**kwargs)
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.depth = max(depth, 1)
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.merge = merge()
        self.padding = padding
        self.layers = [
            layers.Conv2D(
                activation=self.activation,
                bias_initializer=self.bias_initializer,
                filters=self.filters,
                kernel_initializer=self.kernel_initializer,
                kernel_size=self.kernel_size,
                padding=self.padding,
            )
            for _ in range(self.get_layer_count(self.depth))
        ]
        self.merges = [merge() for _ in range(self.get_merge_count(self.depth))]

    def get_layer_count(self, depth):
        if depth == 1:
            return 3
        else:
            return 2 * self.get_layer_count(depth - 1) + 1

    def get_merge_count(self, depth):
        if depth == 1:
            return 1
        else:
            return 2 * self.get_merge_count(depth - 1) + 1

    def call(self, inputs, layers=None, merges=None, **kwargs):
        if (layers is None) and (merges is None):
            layers = self.layers
            merges = self.merges

        if (len(layers) == 3) and (len(merges) == 1):
            branch_1 = layers[0](inputs)
            branch_1 = layers[1](branch_1)
            branch_2 = layers[2](inputs)
            return merges[0]([branch_1, branch_2])
        else:
            layer_split = (len(layers) - 1) // 2
            merge_split = (len(merges) - 1) // 2
            branch_1 = self.call(
                inputs, layers=layers[:layer_split], merges=merges[:merge_split],
            )
            branch_1 = self.call(
                branch_1, layers=layers[layer_split:-1], merges=merges[merge_split:-1],
            )
            branch_2 = layers[-1](inputs)
            return merges[-1]([branch_1, branch_2])


class ResidualBlock(layers.Layer):
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
        kernel_size=(3, 3),
        merge=layers.Concatenate,
        padding="same",
        residual_projection=False,
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
                Reduction ratio passed to SqueezeExciteBlock.
            squeeze_and_excite: (bool)
                Toggles the use of a SE block between the convolutions and residual merge.
            **kwargs:
        """
        super().__init__(**kwargs)
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.padding = padding
        self.residual_projection = residual_projection
        self.se_reduce_factor = se_reduce_factor
        self.squeeze_and_excite = squeeze_and_excite

        self.conv_1 = layers.Conv2D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.conv_2 = layers.Conv2D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )

        if self.squeeze_and_excite:
            self.se_block = SqueezeExciteBlock(
                reduce_factor=self.se_reduce_factor, width=self.filters,
            )

        if self.residual_projection:
            self.projection = layers.Conv2D(
                filters=self.filters,
                kernel_size=(1, 1),
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
            )
        self.merge = merge()

    def call(self, inputs, **kwargs):
        pred = self.conv_1(inputs)
        pred = self.conv_2(pred)
        if self.squeeze_and_excite:
            pred = self.se_block(pred)
        if self.residual_projection:
            inputs = self.projection(inputs)
        return self.merge([inputs, pred])


class ResidualBottleneckBlock(layers.Layer):
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
        kernel_size=(3, 3),
        merge=layers.Concatenate,
        neck_filters=None,
        padding="same",
        residual_projection=False,
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
                Reduction ratio passed to SqueezeExciteBlock.
            squeeze_and_excite: (bool)
                Toggles the use of a SE block between the convolutions and residual merge.
            **kwargs:
        """
        super().__init__(**kwargs)
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

        self.compress_conv = layers.Conv2D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.neck_filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=(1, 1),
        )

        self.bottleneck_conv = layers.Conv2D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.neck_filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.expand_conv = layers.Conv2D(
            activation=self.activation,
            filters=self.filters,
            kernel_size=(1, 1),
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )

        if self.squeeze_and_excite:
            self.se_block = SqueezeExciteBlock(
                reduce_factor=self.excite_factor, width=self.filters,
            )

        if self.residual_projection:
            self.projection = layers.Conv2D(
                filters=self.filters,
                kernel_size=(1, 1),
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
            )
        self.merge = merge()

    def call(self, inputs, **kwargs):
        pred = self.compress_conv(inputs)
        pred = self.bottleneck_conv(pred)
        pred = self.expand_conv(pred)
        if self.squeeze_and_excite:
            pred = self.se_block(pred)
        if self.residual_projection:
            inputs = self.projection(inputs)
        return self.merge([inputs, pred])


class SparseBlock(layers.Layer):
    """
    Implements the sparsely connected convolution block described in:
        https://arxiv.org/abs/1801.05895
    """

    def __init__(
        self,
        activation="relu",
        bias_initializer="zeros",
        depth=4,
        filters=16,
        kernel_initializer="glorot_uniform",
        kernel_size=(3, 3),
        merge=layers.Concatenate,
        padding="same",
        **kwargs,
    ):
        """
        Args:
            activation: (str or Callable)
                Name or instance of a keras activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            depth: (int)
                Number of convolutions used in block construction.
            filters: (int)
                Number of filters per convolution.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            kernel_size: (tuple[int] or int)
                Dimensions of the convolution filters.
            merge: (keras.layers.Layer)
                Layer used to merge branches of computation.
            padding: (str)
                Convolution padding strategy.
        """
        super().__init__(**kwargs)
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.depth = max(depth, 1)
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.merge = merge()
        self.padding = padding
        self.layers = [
            layers.Conv2D(
                activation=self.activation,
                bias_initializer=self.bias_initializer,
                filters=self.filters,
                kernel_initializer=self.kernel_initializer,
                kernel_size=self.kernel_size,
                padding=self.padding,
            )
            for _ in range(self.depth)
        ]

    def call(self, inputs, **kwargs):
        pred = inputs
        inputs = [inputs]
        for i in range(self.depth):
            inds = [-(2 ** j) for j in range(1 + int(np.log2(i + 1)))]
            if i >= 2:
                pred = self.merge([inputs[ind] for ind in inds])
            pred = self.layers[i](pred)
            inputs.append(pred)
        return pred


class SqueezeExciteBlock(layers.Layer):
    """
    Implements the Squeeze and Excitation block discussed in:
        https://arxiv.org/abs/1709.01507
    """

    def __init__(
        self,
        activation="relu",
        bias_initializer="zeros",
        kernel_initializer="glorot_uniform",
        merge=layers.Multiply,
        reduce_factor=4,
        width=16,
        **kwargs,
    ):
        """
        Args:
            activation: (str or Callable)
                Name or instance of a keras activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            depth: (int)
                Number of convolutions used in block construction.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            merge: (keras.layers.Layer)
                Layer used to merge branches of computation.
            reduce_factor: (int)
                Determines the number of neurons on the smaller excite layer.
            width: (int)
                Number of neurons in the larger excite layer.
        """
        super().__init__(**kwargs)
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.excite_factor = reduce_factor
        self.width = width
        self.kernel_initializer = kernel_initializer

        self.squeeze = layers.GlobalAveragePooling2D()
        self.excite_1 = layers.Dense(
            self.width // self.excite_factor,
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            kernel_initializer=self.kernel_initializer,
        )
        self.excite_2 = layers.Dense(
            self.width,
            activation="sigmoid",
            bias_initializer=self.bias_initializer,
            kernel_initializer=self.kernel_initializer,
        )
        self.reshape = layers.Reshape((1, 1, self.width))
        self.merge = merge()

    def call(self, inputs, **kwargs):
        pred = self.squeeze(inputs)
        pred = self.excite_1(pred)
        pred = self.excite_2(pred)
        pred = self.reshape(pred)
        return self.merge([inputs, pred])


register_custom_objects(
    [
        DenseBlock,
        DilationBlock,
        FractalBlock,
        ResidualBlock,
        ResidualBottleneckBlock,
        SparseBlock,
        SqueezeExciteBlock,
    ]
)

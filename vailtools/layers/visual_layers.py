import numpy as np
from tensorflow.keras import layers

from ..utils import register_custom_objects


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
            excite_factor=4,
            filters=16,
            kernel_initializer="glorot_uniform",
            kernel_size=(3, 3),
            merge=layers.Concatenate,
            padding="same",
            residual_projection=False,
            squeeze_and_excite=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.excite_factor = int(excite_factor)
        self.filters = int(filters)
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.padding = padding
        self.residual_projection = residual_projection
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
            self.squeeze = layers.GlobalAveragePooling2D()
            self.excite_1 = layers.Dense(
                self.filters // self.excite_factor,
                activation=self.activation,
                bias_initializer=self.bias_initializer,
                kernel_initializer=self.kernel_initializer,
            )
            self.excite_2 = layers.Dense(
                self.filters,
                activation="sigmoid",
                bias_initializer=self.bias_initializer,
                kernel_initializer=self.kernel_initializer,
            )
            self.reshape = layers.Reshape((1, 1, self.filters))
            self.scale = layers.Lambda(lambda x, y: x * y)

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
            se_pred = self.squeeze(pred)
            se_pred = self.excite_1(se_pred)
            se_pred = self.excite_2(se_pred)
            se_pred = self.reshape(se_pred)
            pred = self.scale(pred, se_pred)

        if self.residual_projection:
            inputs = self.projection(inputs)
        return self.merge([inputs, pred])


class ResidualBottleneckBlock(layers.Layer):
    """
    Implements the residual bottleneck block discussed in:
        https://arxiv.org/abs/1512.03385
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
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.neck_filters = neck_filters or max(filters // 4, 1)
        self.padding = padding
        self.residual_projection = residual_projection

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
        if self.residual_projection:
            inputs = self.projection(inputs)
        return self.merge([inputs, pred])


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
                Name of a keras activation function or an instance of a keras/Tensorflow activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            depth: (int)
                Number of convolutions used in block construction.
            filters: (int)
                Number of filters used in each convolution.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            kernel_size: (tuple[int] or int)
                Dimensions of the convolution filters.
            merge: (keras.layers.Layer)
                Keras layer that merges the input and output branches of a residual block.
            padding: (str)
                Padding strategy applied during convolution operations.
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
            pred = self.merge([inputs[ind] for ind in inds])
            pred = self.layers[i](pred)
            inputs.append(pred)
        return pred


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
                Name of a keras activation function or an instance of a keras/Tensorflow activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            depth: (int)
                Number of convolutions used in block construction.
            filters: (int)
                Number of filters used in each convolution.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            kernel_size: (tuple[int] or int)
                Dimensions of the convolution filters.
            merge: (keras.layers.Layer)
                Keras layer that merges the input and output branches of a residual block.
            padding: (str)
                Padding strategy applied during convolution operations.
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
                Name of a keras activation function or an instance of a keras/Tensorflow activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            dilations: (Iterable[int])
                Dilation rate used in each convolution.
            filters: (int)
                Number of filters used in each convolution.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            kernel_size: (tuple[int] or int)
                Dimensions of the convolution filters.
            merge: (keras.layers.Layer)
                Keras layer that merges the input and output branches of a residual block.
            padding: (str)
                Padding strategy applied during convolution operations.
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


class FireBlock(layers.Layer):
    """
    Implements the Fire module discussed in:
        https://arxiv.org/abs/1602.07360
    """

    def __init__(
            self,
            activation="relu",
            bias_initializer="zeros",
            e1_filters=4,
            e3_filters=4,
            kernel_initializer="glorot_uniform",
            kernel_size=(3, 3),
            merge=layers.Concatenate,
            padding="same",
            s1_filters=3,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.e1_filters = e1_filters
        self.e3_filters = e3_filters
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.padding = padding
        self.s1_filters = s1_filters

        self.squeeze = layers.Conv2D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.s1_filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.excitation_1 = layers.Conv2D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.e1_filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.excitation_3 = layers.Conv2D(
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
        pred = self.merge([
            self.excitation_1(pred),
            self.excitation_2(pred),
        ])
        return pred


register_custom_objects([
    DenseBlock,
    DilationBlock,
    FireBlock,
    FractalBlock,
    ResidualBlock,
    ResidualBottleneckBlock,
])

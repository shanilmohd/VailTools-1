from keras import layers
from keras.utils.generic_utils import get_custom_objects
import numpy as np


class ResidualBlock(layers.Layer):
    """
    Implements the simple residual block discussed in:
        https://arxiv.org/abs/1512.03385
    """

    def __init__(
            self,
            activation='relu',
            bias_initializer='zeros',
            filters=16,
            kernel_initializer='glorot_uniform',
            kernel_size=(3, 3),
            merge=layers.Concatenate,
            padding='same',
            residual_projection=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.padding = padding
        self.residual_projection = residual_projection

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
        if self.residual_projection:
            self.projection = layers.Conv2D(
                filters=self.filters,
                kernel_size=(1, 1),
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
            )
        self.final_merge = merge()

    def call(self, inputs, **kwargs):
        pred = self.conv_1(inputs)
        pred = self.conv_2(pred)
        if self.residual_projection:
            inputs = self.projection(inputs)
        return self.final_merge([inputs, pred])

    def compute_output_shape(self, input_shape):
        shape_1 = self.conv_1.compute_output_shape(input_shape)
        shape_2 = self.conv_2.compute_output_shape(shape_1)
        if self.residual_projection:
            input_shape = self.projection.compute_output_shape(input_shape)
        return self.final_merge.compute_output_shape([input_shape, shape_2])


class ResidualBottleneckBlock(layers.Layer):
    """
    Implements the residual bottleneck block discussed in:
        https://arxiv.org/abs/1512.03385
    """

    def __init__(
            self,
            activation='relu',
            bias_initializer='zeros',
            filters=16,
            kernel_initializer='glorot_uniform',
            kernel_size=(3, 3),
            merge=layers.Concatenate,
            neck_filters=None,
            padding='same',
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
        self.final_merge = merge()

    def call(self, inputs, **kwargs):
        pred = self.compress_conv(inputs)
        pred = self.bottleneck_conv(pred)
        pred = self.expand_conv(pred)
        if self.residual_projection:
            inputs = self.projection(inputs)
        return self.final_merge([inputs, pred])

    def compute_output_shape(self, input_shape):
        shape_1 = self.compress_conv.compute_output_shape(input_shape)
        shape_2 = self.bottleneck_conv.compute_output_shape(shape_1)
        shape_3 = self.expand_conv.compute_output_shape(shape_2)
        if self.residual_projection:
            input_shape = self.projection.compute_output_shape(input_shape)
        return self.final_merge.compute_output_shape([input_shape, shape_3])


class DenseBlock(layers.Layer):
    """
    Implements the densely connected convolution block discussed in:
        https://arxiv.org/abs/1608.06993
    """

    def __init__(
            self,
            activation='relu',
            bias_initializer='zeros',
            depth=2,
            filters=16,
            kernel_initializer='glorot_uniform',
            kernel_size=(3, 3),
            merge=layers.Concatenate,
            padding='same',
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

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for layer in self.layers:
            output_shape = self.merge.compute_output_shape([
                output_shape,
                layer.compute_output_shape(output_shape),
            ])
        return output_shape


class SparseBlock(layers.Layer):
    """
    Implements the sparsely connected convolution block described in:
        https://arxiv.org/abs/1801.05895
    """
    def __init__(
            self,
            activation='relu',
            bias_initializer='zeros',
            depth=4,
            filters=16,
            kernel_initializer='glorot_uniform',
            kernel_size=(3, 3),
            merge=layers.Concatenate,
            padding='same',
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

    def compute_output_shape(self, input_shape):
        pred_shape = input_shape
        input_shapes = [input_shape]
        for i in range(self.depth):
            inds = [-(2 ** j) for j in range(1 + int(np.log2(i + 1)))]
            pred_shape = self.merge.compute_output_shape([input_shapes[ind] for ind in inds])
            pred_shape = self.layers[i].compute_output_shape(pred_shape)
            input_shapes.append(pred_shape)
        return pred_shape


class FractalBlock(layers.Layer):
    """
    Implements the fractal convolution block described in:
        https://arxiv.org/abs/1605.07648
    """
    def __init__(
            self,
            activation='relu',
            bias_initializer='zeros',
            depth=4,
            filters=16,
            kernel_initializer='glorot_uniform',
            kernel_size=(3, 3),
            merge=layers.Concatenate,
            padding='same',
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
        self.merges = [
            merge()
            for _ in range(self.get_merge_count(self.depth))
        ]

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
                inputs,
                layers=layers[:layer_split],
                merges=merges[:merge_split],
            )
            branch_1 = self.call(
                branch_1,
                layers=layers[layer_split:-1],
                merges=merges[merge_split:-1],
            )
            branch_2 = layers[-1](inputs)
            return merges[-1]([branch_1, branch_2])

    def compute_output_shape(self, input_shape, layers=None, merges=None):
        if not layers and not merges:
            layers = self.layers
            merges = self.merges

        if (len(layers) == 3) and (len(merges) == 1):
            branch_1_shape = layers[0].compute_output_shape(input_shape)
            branch_1_shape = layers[1].compute_output_shape(branch_1_shape)
            branch_2_shape = layers[2].compute_output_shape(input_shape)
            return merges[0].compute_output_shape([branch_1_shape, branch_2_shape])
        else:
            layer_split = (len(layers) - 1) // 2
            merge_split = (len(merges) - 1) // 2
            branch_1_shape = self.compute_output_shape(
                input_shape,
                layers=layers[:layer_split],
                merges=merges[:merge_split],
            )
            branch_1_shape = self.compute_output_shape(
                branch_1_shape,
                layers=layers[layer_split:-1],
                merges=merges[merge_split:-1],
            )
            branch_2_shape = layers[-1].compute_output_shape(input_shape)
            return merges[-1].compute_output_shape([branch_1_shape, branch_2_shape])


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
            activation='selu',
            bias_initializer='zeros',
            dilations=None,
            filters=16,
            kernel_initializer='glorot_uniform',
            kernel_size=(3, 3),
            merge=layers.Add,
            padding='same',
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

    def compute_output_shape(self, input_shape):
        pred_shapes = [
            layer.compute_output_shape(input_shape)
            for layer in self.layers
        ]
        if self.skip_connection:
            pred_shapes = [input_shape] + pred_shapes
        return self.merge.compute_output_shape(pred_shapes)


# Todo: May want to add some validation to ensure that builtin Keras objects are
#  not overwritten.
get_custom_objects().update({
    x.__name__: x
    for x in [ResidualBlock, ResidualBottleneckBlock, DenseBlock, FractalBlock, DilationBlock]
})

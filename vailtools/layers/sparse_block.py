import numpy as np
from tensorflow.keras import layers

from ..utils import register_custom_objects


class SparseBlock(layers.Layer):
    """
    Implements the sparsely connected convolution block described in:
        https://arxiv.org/abs/1801.05895

    Can be specialized to operate on different ranked tensors by passing an
    appropriate primary layer.
    """

    def __init__(
        self,
        primary_layer,
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
            primary_layer(
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


class Sparse1D(SparseBlock):
    """
    Implements the sparsely connected convolution block described in:
        https://arxiv.org/abs/1801.05895

    Accepts sequence data, i.e. dims of (batch size, steps, features).
    """

    def __init__(
        self,
        activation="relu",
        bias_initializer="zeros",
        depth=4,
        filters=16,
        kernel_initializer="glorot_uniform",
        kernel_size=3,
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
        super().__init__(
            layers.Conv1D,
            activation=activation,
            bias_initializer=bias_initializer,
            depth=depth,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=kernel_size,
            merge=merge,
            padding=padding,
            **kwargs,
        )


class Sparse2D(SparseBlock):
    """
    Implements the sparsely connected convolution block described in:
        https://arxiv.org/abs/1801.05895

    Accepts image data, i.e. dims of (batch size, height, width, features).
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
        super().__init__(
            layers.Conv2D,
            activation=activation,
            bias_initializer=bias_initializer,
            depth=depth,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=kernel_size,
            merge=merge,
            padding=padding,
            **kwargs,
        )


class Sparse3D(SparseBlock):
    """
    Implements the sparsely connected convolution block described in:
        https://arxiv.org/abs/1801.05895

    Accepts volumetric data, i.e. dims of (batch size, height, width, depth, features).
    """

    def __init__(
        self,
        activation="relu",
        bias_initializer="zeros",
        depth=4,
        filters=16,
        kernel_initializer="glorot_uniform",
        kernel_size=(3, 3, 3),
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
        super().__init__(
            layers.Conv3D,
            activation=activation,
            bias_initializer=bias_initializer,
            depth=depth,
            filters=filters,
            kernel_initializer=kernel_initializer,
            kernel_size=kernel_size,
            merge=merge,
            padding=padding,
            **kwargs,
        )


register_custom_objects([Sparse1D, Sparse2D, Sparse3D, SparseBlock])

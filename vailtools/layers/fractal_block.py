from tensorflow.keras import layers

from ..utils import register_custom_objects


class FractalBlock(layers.Layer):
    """
    Implements the fractal convolution block described in:
        https://arxiv.org/abs/1605.07648
    """

    def __init__(
        self,
        primary_layer,
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
            primary_layer: (keras.layers.Layer)
                One of {Conv1D, Conv2D, Conv3D}.
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
        self.merge = merge
        self.padding = padding

        self.primary_layer = primary_layer
        self.layers = [
            self.primary_layer(
                activation=self.activation,
                bias_initializer=self.bias_initializer,
                filters=self.filters,
                kernel_initializer=self.kernel_initializer,
                kernel_size=self.kernel_size,
                padding=self.padding,
            )
            for _ in range(self.get_layer_count(self.depth))
        ]
        self.merges = [self.merge() for _ in range(self.get_merge_count(self.depth))]

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

    def get_config(self):
        base = super().get_config()
        config = {
            "primary_layer": self.primary_layer,
            "activation": self.activation,
            "bias_initializer": self.bias_initializer,
            "depth": self.depth,
            "filters": self.filters,
            "kernel_initializer": self.kernel_initializer,
            "kernel_size": self.kernel_size,
            "merge": self.merge,
            "padding": self.padding,
        }
        return {**base, **config}


class Fractal1D(FractalBlock):
    """
    Implements the fractal convolution block described in:
        https://arxiv.org/abs/1605.07648

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


class Fractal2D(FractalBlock):
    """
    Implements the fractal convolution block described in:
        https://arxiv.org/abs/1605.07648

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


class Fractal3D(FractalBlock):
    """
    Implements the fractal convolution block described in:
        https://arxiv.org/abs/1605.07648

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


register_custom_objects([Fractal1D, Fractal2D, Fractal3D, FractalBlock])

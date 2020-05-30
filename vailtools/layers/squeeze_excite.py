from tensorflow.keras import layers

from ..utils import register_custom_objects


class SqueezeExcite1D(layers.Layer):
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

        self.squeeze = layers.GlobalAveragePooling1D()
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
        self.reshape = layers.Reshape((1, self.width))
        self.merge = merge()

    def call(self, inputs, **kwargs):
        pred = self.squeeze(inputs)
        pred = self.excite_1(pred)
        pred = self.excite_2(pred)
        pred = self.reshape(pred)
        return self.merge([inputs, pred])


class SqueezeExcite2D(layers.Layer):
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


class SqueezeExcite3D(layers.Layer):
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

        self.squeeze = layers.GlobalAveragePooling3D()
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
        self.reshape = layers.Reshape((1, 1, 1, self.width))
        self.merge = merge()

    def call(self, inputs, **kwargs):
        pred = self.squeeze(inputs)
        pred = self.excite_1(pred)
        pred = self.excite_2(pred)
        pred = self.reshape(pred)
        return self.merge([inputs, pred])


register_custom_objects([SqueezeExcite1D, SqueezeExcite2D, SqueezeExcite3D])

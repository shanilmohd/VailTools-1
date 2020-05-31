from tensorflow.keras import layers

from ..utils import register_custom_objects


class SelectiveKernel1D(layers.Layer):
    """
    Implements the selective kernel convolution described in:
        https://arxiv.org/abs/1903.06586
    """

    def __init__(
        self,
        activation="relu",
        bias_initializer="zeros",
        depth=4,
        filters=32,
        kernel_initializer="glorot_uniform",
        kernel_size=(3, 3),
        merge=layers.Concatenate,
        padding="same",
        reduction_rate=16,
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
        self.reduction_rate = reduction_rate

        self.conv_3 = layers.Conv1D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=3,
            padding=self.padding,
        )
        self.conv_5 = layers.Conv1D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=3,
            dilation_rate=2,
            padding=self.padding,
        )
        self.pool = layers.GlobalAveragePooling1D()
        self.fc_1 = layers.Dense(
            max(self.filters // self.reduction_rate, 32), activation=self.activation
        )
        self.fc_2 = layers.Dense(self.filters, activation="softmax")
        self.fc_3 = layers.Dense(self.filters, activation="softmax")
        self.attention_reshape = layers.Reshape((1, self.filters))

    def call(self, inputs, **kwargs):
        split_1 = self.conv_3(inputs)
        split_2 = self.conv_5(inputs)
        features = self.fc_1(self.pool(layers.add([split_1, split_2])))
        attention_1 = self.attention_reshape(self.fc_2(features))
        attention_2 = self.attention_reshape(self.fc_3(features))
        fuse_1 = layers.multiply([split_1, attention_1])
        fuse_2 = layers.multiply([split_2, attention_2])
        return layers.add([fuse_1, fuse_2])


class SelectiveKernel2D(layers.Layer):
    """
    Implements the selective kernel convolution described in:
        https://arxiv.org/abs/1903.06586
    """

    def __init__(
        self,
        activation="relu",
        bias_initializer="zeros",
        depth=4,
        filters=32,
        kernel_initializer="glorot_uniform",
        kernel_size=(3, 3),
        merge=layers.Concatenate,
        padding="same",
        reduction_rate=16,
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
        self.reduction_rate = reduction_rate

        self.conv_3 = layers.DepthwiseConv2D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            kernel_initializer=self.kernel_initializer,
            kernel_size=3,
            padding=self.padding,
        )
        self.conv_5 = layers.DepthwiseConv2D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            kernel_initializer=self.kernel_initializer,
            kernel_size=3,
            dilation_rate=2,
            padding=self.padding,
        )
        self.pool = layers.GlobalAveragePooling2D()
        self.fc_1 = layers.Dense(
            max(self.filters // self.reduction_rate, 32), activation=self.activation
        )
        self.fc_2 = layers.Dense(self.filters, activation="softmax")
        self.fc_3 = layers.Dense(self.filters, activation="softmax")
        self.attention_reshape = layers.Reshape((1, 1, self.filters))

    def call(self, inputs, **kwargs):
        split_1 = self.conv_3(inputs)
        split_2 = self.conv_5(inputs)
        features = self.fc_1(self.pool(layers.add([split_1, split_2])))
        attention_1 = self.attention_reshape(self.fc_2(features))
        attention_2 = self.attention_reshape(self.fc_3(features))
        fuse_1 = layers.multiply([split_1, attention_1])
        fuse_2 = layers.multiply([split_2, attention_2])
        return layers.add([fuse_1, fuse_2])


class SelectiveKernel3D(layers.Layer):
    """
    Implements the selective kernel convolution described in:
        https://arxiv.org/abs/1903.06586
    """

    def __init__(
        self,
        activation="relu",
        bias_initializer="zeros",
        depth=4,
        filters=32,
        kernel_initializer="glorot_uniform",
        kernel_size=(3, 3),
        merge=layers.Concatenate,
        padding="same",
        reduction_rate=16,
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
        self.reduction_rate = reduction_rate

        self.conv_3 = layers.Conv3D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=3,
            padding=self.padding,
        )
        self.conv_5 = layers.Conv3D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=3,
            dilation_rate=2,
            padding=self.padding,
        )
        self.pool = layers.GlobalAveragePooling3D()
        self.fc_1 = layers.Dense(
            max(self.filters // self.reduction_rate, 32), activation=self.activation
        )
        self.fc_2 = layers.Dense(self.filters, activation="softmax")
        self.fc_3 = layers.Dense(self.filters, activation="softmax")
        self.attention_reshape = layers.Reshape((1, 1, 1, self.filters))

    def call(self, inputs, **kwargs):
        split_1 = self.conv_3(inputs)
        split_2 = self.conv_5(inputs)
        features = self.fc_1(self.pool(layers.add([split_1, split_2])))
        attention_1 = self.attention_reshape(self.fc_2(features))
        attention_2 = self.attention_reshape(self.fc_3(features))
        fuse_1 = layers.multiply([split_1, attention_1])
        fuse_2 = layers.multiply([split_2, attention_2])
        return layers.add([fuse_1, fuse_2])


register_custom_objects([SelectiveKernel1D, SelectiveKernel2D, SelectiveKernel3D])

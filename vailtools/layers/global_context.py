"""
Implementations of the Global Context Module from https://arxiv.org/abs/1904.11492
for sequence data, image data, and volumetric data.

TODO: Ensure that the dot products are being used correctly.
"""

from tensorflow.keras import backend as K
from tensorflow.keras import layers

from ..utils import register_custom_objects


class GlobalContext1D(layers.Layer):
    """
    Accepts sequence data, i.e. dimensions (batch size, time, features).

    References:
        https://arxiv.org/abs/1904.11492
    """

    def __init__(
            self,
            activation="relu",
            bias_initializer="zeros",
            filters=16,
            kernel_initializer="glorot_uniform",
            merge=layers.Add,
            project_inputs=False,
            reduction_factor=4,
            **kwargs,
    ):
        """
        Args:
            activation: (str or Callable)
                Name or instance of a keras activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            filters: (None or int)
                Number of filters used in the expand convolutions.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            merge: (keras.layers.Layer)
                Layer used to merge branches of computation.
            padding: (str)
                Convolution padding strategy.
            project_inputs: (bool)
                Applies a linear projection to the inputs.
                Required when filters does not equal the channel dimension of the inputs.
            reduction_factor: (int)
                Determines the number of filters in the first transform convolution.
            **kwargs:
        """
        super().__init__(**kwargs)
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.project_inputs = project_inputs
        self.reduction_factor = reduction_factor

        self.context = layers.Conv1D(
            activation="softmax",
            bias_initializer=self.bias_initializer,
            filters=1,
            kernel_initializer=self.kernel_initializer,
            kernel_size=1,
        )
        self.context_combine = layers.Dot(axes=1)
        self.transform_1 = layers.Conv1D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.filters // self.reduction_factor,
            kernel_initializer=self.kernel_initializer,
            kernel_size=1,
        )
        self.layer_norm = layers.LayerNormalization()
        self.transform_activation = layers.Activation(self.activation)
        self.transform_2 = layers.Conv1D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=1,
        )

        if self.project_inputs:
            self.projection = layers.Conv1D(
                filters=self.filters,
                kernel_size=1,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
            )

        self.merge = merge()

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.filters)

    def call(self, inputs, **kwargs):
        if self.project_inputs:
            inputs = self.projection(inputs)

        # Context modelling
        pred = self.context(inputs)
        pred = self.context_combine([pred, inputs])

        # Transform
        pred = self.transform_1(pred)
        pred = self.layer_norm(pred)
        pred = self.transform_activation(pred)
        pred = self.transform_2(pred)

        return self.merge([inputs, pred])


class GlobalContext2D(layers.Layer):
    """
    Accepts image data, i.e. dimensions (batch size, width, height, features)

    References:
        https://arxiv.org/abs/1904.11492
    """

    def __init__(
            self,
            activation="relu",
            bias_initializer="zeros",
            filters=16,
            kernel_initializer="glorot_uniform",
            merge=layers.Add,
            project_inputs=False,
            reduction_factor=4,
            **kwargs,
    ):
        """
        Args:
            activation: (str or Callable)
                Name or instance of a keras activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            filters: (None or int)
                Number of filters used in the expand convolutions.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            merge: (keras.layers.Layer)
                Layer used to merge branches of computation.
            padding: (str)
                Convolution padding strategy.
            project_inputs: (bool)
                Applies a linear projection to the inputs.
                Required when filters does not equal the channel dimension of the inputs.
            reduction_factor: (int)
                Determines the number of filters in the first transform convolution.
            **kwargs:
        """
        super().__init__(**kwargs)
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.project_inputs = project_inputs
        self.reduction_factor = reduction_factor

        self.context = layers.Conv2D(
            activation="softmax",
            bias_initializer=self.bias_initializer,
            filters=1,
            kernel_initializer=self.kernel_initializer,
            kernel_size=(1, 1),
        )
        self.context_reshape_1 = None
        self.context_reshape_2 = None
        self.context_combine = layers.Dot(axes=2)
        self.transform_1 = layers.Conv2D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.filters // self.reduction_factor,
            kernel_initializer=self.kernel_initializer,
            kernel_size=(1, 1),
        )
        self.layer_norm = layers.LayerNormalization()
        self.transform_activation = layers.Activation(self.activation)
        self.transform_2 = layers.Conv2D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=(1, 1),
        )

        if self.project_inputs:
            self.projection = layers.Conv2D(
                filters=self.filters,
                kernel_size=(1, 1),
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
            )

        self.merge = merge()

    def build(self, input_shape):
        if self.project_inputs:
            self.context_reshape_1 = layers.Reshape((1, -1, self.filters))
        else:
            self.context_reshape_1 = layers.Reshape((1, -1, input_shape[-1]))
        self.context_reshape_2 = layers.Reshape((1, -1, 1))

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.filters)

    def call(self, inputs, **kwargs):
        if self.project_inputs:
            inputs = self.projection(inputs)

        # Context modelling
        pred = self.context(inputs)
        pred = self.context_combine([self.context_reshape_2(pred), self.context_reshape_1(inputs)])
        pred = K.squeeze(pred, axis=-2)

        # Transform
        pred = self.transform_1(pred)
        pred = self.layer_norm(pred)
        pred = self.transform_activation(pred)
        pred = self.transform_2(pred)

        return self.merge([inputs, pred])


class GlobalContext3D(layers.Layer):
    """
    Accepts volumetric data, i.e. dimensions (batch size, width, height, depth, features)

    References:
        https://arxiv.org/abs/1904.11492
    """

    def __init__(
            self,
            activation="relu",
            bias_initializer="zeros",
            filters=16,
            kernel_initializer="glorot_uniform",
            merge=layers.Add,
            project_inputs=False,
            reduction_factor=4,
            **kwargs,
    ):
        """
        Args:
            activation: (str or Callable)
                Name or instance of a keras activation function.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            filters: (None or int)
                Number of filters used in the expand convolutions.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            merge: (keras.layers.Layer)
                Layer used to merge branches of computation.
            padding: (str)
                Convolution padding strategy.
            project_inputs: (bool)
                Applies a linear projection to the inputs.
                Required when filters does not equal the channel dimension of the inputs.
            reduction_factor: (int)
                Determines the number of filters in the first transform convolution.
            **kwargs:
        """
        super().__init__(**kwargs)
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.project_inputs = project_inputs
        self.reduction_factor = reduction_factor

        self.context = layers.Conv3D(
            activation="softmax",
            bias_initializer=self.bias_initializer,
            filters=1,
            kernel_initializer=self.kernel_initializer,
            kernel_size=(1, 1, 1),
        )
        self.context_reshape_1 = None
        self.context_reshape_2 = None
        self.context_combine = layers.Dot(axes=3)
        self.transform_1 = layers.Conv3D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.filters // self.reduction_factor,
            kernel_initializer=self.kernel_initializer,
            kernel_size=(1, 1, 1),
        )
        self.layer_norm = layers.LayerNormalization()
        self.transform_activation = layers.Activation(self.activation)
        self.transform_2 = layers.Conv3D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=(1, 1, 1),
        )

        if self.project_inputs:
            self.projection = layers.Conv3D(
                filters=self.filters,
                kernel_size=(1, 1, 1),
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
            )

        self.merge = merge()

    def build(self, input_shape):
        if self.project_inputs:
            self.context_reshape_1 = layers.Reshape((1, 1, -1, self.filters))
        else:
            self.context_reshape_1 = layers.Reshape((1, 1, -1, input_shape[-1]))
        self.context_reshape_2 = layers.Reshape((1, 1, -1, 1))

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.filters)

    def call(self, inputs, **kwargs):
        if self.project_inputs:
            inputs = self.projection(inputs)

        # Context modelling
        pred = self.context(inputs)
        pred = self.context_combine([self.context_reshape_2(pred), self.context_reshape_1(inputs)])
        pred = K.squeeze(K.squeeze(pred, axis=-2), axis=-2)

        # Transform
        pred = self.transform_1(pred)
        pred = self.layer_norm(pred)
        pred = self.transform_activation(pred)
        pred = self.transform_2(pred)

        return self.merge([inputs, pred])


register_custom_objects([GlobalContext1D, GlobalContext2D, GlobalContext3D])

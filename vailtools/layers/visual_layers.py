from keras import layers
from keras.utils.generic_utils import get_custom_objects


class ResidualBlock(layers.Layer):
    """
    Implements the simple residual block discussed in:
        https://arxiv.org/abs/1512.03385
    """
    def __init__(
            self,
            activation='selu',
            bias_initializer='zeros',
            filters=16,
            kernel_initializer='glorot_uniform',
            kernel_size=(3, 3),
            padding='same',
            residual_projection=False,
            **kwargs,
    ):
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.padding = padding
        self.residual_projection = residual_projection

        self.conv_1 = None
        self.conv_2 = None
        self.projection = None

        super().__init__(**kwargs)

    def build(self, input_shape):
        self.conv_1 = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            activation=self.activation,
        )
        self.conv_1.build(input_shape)
        self._trainable_weights.extend(self.conv_1.trainable_weights)

        self.conv_2 = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        self.conv_2.build(self.conv_1.compute_output_shape(input_shape))
        self._trainable_weights.extend(self.conv_2.trainable_weights)

        if self.residual_projection:
            self.projection = layers.Conv2D(
                filters=self.filters,
                kernel_size=(1, 1),
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
            )
            self.projection.build(input_shape)
            self._trainable_weights.extend(self.projection.trainable_weights)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        pred = self.conv_1(inputs)
        pred = self.conv_2(pred)
        if self.residual_projection:
            inputs = self.projection(inputs)
        return layers.concatenate([inputs, pred])

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)

        if self.residual_projection:
            output_shape[-1] = self.filters * 2
        else:
            output_shape[-1] += self.filters

        return tuple(output_shape)


class ResidualBottleneckBlock(layers.Layer):
    """
    Implements the residual bottleneck block discussed in:
        https://arxiv.org/abs/1512.03385
    """

    def __init__(
            self,
            activation='selu',
            bias_initializer='zeros',
            filters=16,
            kernel_initializer='glorot_uniform',
            kernel_size=(3, 3),
            neck_filters=None,
            padding='same',
            residual_projection=False,
            **kwargs,
    ):
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.neck_filters = neck_filters or max(filters // 4, 1)
        self.padding = padding
        self.residual_projection = residual_projection

        self.compress_conv = None
        self.bottleneck_conv = None
        self.expand_conv = None
        self.projection = None

        super().__init__(**kwargs)

    def build(self, input_shape):
        self.compress_conv = layers.Conv2D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.neck_filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=(1, 1),
        )
        self.compress_conv.build(input_shape)
        self._trainable_weights.extend(self.compress_conv.trainable_weights)
        output_shape = self.compress_conv.compute_output_shape(input_shape)

        self.bottleneck_conv = layers.Conv2D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            filters=self.neck_filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.bottleneck_conv.build(output_shape)
        self._trainable_weights.extend(self.bottleneck_conv.trainable_weights)
        output_shape = self.bottleneck_conv.compute_output_shape(output_shape)

        self.expand_conv = layers.Conv2D(
            activation=self.activation,
            filters=self.filters,
            kernel_size=(1, 1),
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        self.expand_conv.build(output_shape)
        self._trainable_weights.extend(self.expand_conv.trainable_weights)

        if self.residual_projection:
            self.projection = layers.Conv2D(
                filters=self.filters,
                kernel_size=(1, 1),
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
            )
            self.projection.build(input_shape)
            self._trainable_weights.extend(self.projection.trainable_weights)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        pred = self.compress_conv(inputs)
        pred = self.bottleneck_conv(pred)
        pred = self.expand_conv(pred)
        if self.residual_projection:
            inputs = self.projection(inputs)
        return layers.concatenate([inputs, pred])

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)

        if self.residual_projection:
            output_shape[-1] = self.filters * 2
        else:
            output_shape[-1] += self.filters

        return tuple(output_shape)


# Todo: May want to add some validation to ensure that builtin Keras objects are
#  not overwritten.
get_custom_objects().update({
    x.__name__: x
    for x in [ResidualBlock, ResidualBottleneckBlock]
})

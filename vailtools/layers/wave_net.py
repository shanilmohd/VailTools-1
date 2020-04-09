from tensorflow.keras import layers

from ..utils import register_custom_objects


class WaveNetBlock(layers.Layer):
    """
    Implements the basic building block of the WaveNet architecture:
        https://arxiv.org/abs/1609.03499
    """

    def __init__(
        self,
        activation="tanh",
        bias_initializer="zeros",
        dilation_rate=1,
        filters=16,
        gate_activation="sigmoid",
        gate_merge=layers.Multiply,
        kernel_initializer="glorot_uniform",
        kernel_size=3,
        padding="causal",
        project=True,
        skip_merge=layers.Add,
        **kwargs,
    ):
        """
        Args:
            activation: (str or Callable)
                Name of a keras activation function or an instance of a keras/Tensorflow activation function.
                Applied to the non-gate branch of a gated activation unit.
            bias_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            dilation_rate: (int)
                Dilation rate used in convolutions.
            filters: (int)
                Number of filters used in convolutions.
            kernel_initializer: (str or Callable)
                Name or instance of a keras.initializers.Initializer.
            padding: (str)
                Padding scheme used in convolutions. Options are {'valid', 'same', 'causal'}.
            project: (bool)
                Toggles the use of a linear projection on the residual branch.
                This alleviates channel dimension issues when filters is not equal
                to the number of input channels.
            gate_activation: (str or Callable)
                Name of a keras activation function or an instance of a keras/Tensorflow activation function.
                Applied to the gate branch of a gated activation unit
            kernel_size: (tuple[int] or int)
                Dimensions of the convolution filters.
            residual_merge: (keras.layers.Layer)
                Keras layer that merges the input and output branches of a residual block.
        """
        super().__init__(**kwargs)
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.dilation_rate = dilation_rate
        self.filters = filters
        self.gate_activation = gate_activation
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.project = project

        self.value_branch = layers.Conv1D(
            activation=self.activation,
            bias_initializer=self.bias_initializer,
            dilation_rate=self.dilation_rate,
            filters=self.filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=self.kernel_size,
            padding=padding,
        )
        self.gate_branch = layers.Conv1D(
            activation=self.gate_activation,
            bias_initializer=self.bias_initializer,
            dilation_rate=self.dilation_rate,
            filters=self.filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=self.kernel_size,
            padding=padding,
        )
        self.skip_out = layers.Conv1D(
            bias_initializer=self.bias_initializer,
            filters=self.filters,
            kernel_initializer=self.kernel_initializer,
            kernel_size=1,
        )
        if self.project:
            self.project_layer = layers.Conv1D(
                bias_initializer=self.bias_initializer,
                filters=self.filters,
                kernel_initializer=self.kernel_initializer,
                kernel_size=1,
            )

        self.gate_merge = gate_merge()
        self.skip_merge = skip_merge()

    def call(self, inputs, **kwargs):
        value = self.value_branch(inputs)
        gate = self.gate_branch(inputs)
        gated_value = self.gate_merge([value, gate])
        skip_out = self.skip_out(gated_value)

        if self.project:
            inputs = self.project_layer(inputs)

        return skip_out, self.skip_merge([inputs, skip_out])


register_custom_objects([WaveNetBlock])
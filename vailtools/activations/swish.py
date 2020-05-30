from tensorflow import keras
from tensorflow.keras import backend as K

from ..utils import register_custom_objects

if getattr(keras.activations, "swish", None) is None:

    def swish(x):
        return x * keras.activations.sigmoid(x)


else:
    swish = keras.activations.swish


def mish(x):
    """
    A non-trainable implementation of the Mish activation function.

    References:
        - [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681)
    """
    return x * keras.activations.tanh(keras.activations.softplus(x))


class Swish(keras.layers.Layer):
    """
    A trainable implementation of the Swish activation function.

    References:
        - [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
        - [Sigmoid-weighted linear units for neural network function
           approximation in reinforcement learning](https://arxiv.org/abs/1702.03118)
    """

    def __init__(self, beta=1.0, trainable=True, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.beta = K.variable(beta, dtype=K.floatx(), name="beta")
        self.trainable = trainable
        if self.trainable:
            self._trainable_weights.append(self.beta)

    def call(self, inputs, **kwargs):
        return inputs * keras.activations.sigmoid(self.beta * inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "beta": self.get_weights()[0] if self.trainable else self.beta,
            "trainable": self.trainable,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class Mish(keras.layers.Layer):
    """
    A trainable implementation of the Mish activation function.

    References:
        - [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681)
    """

    def __init__(self, beta=1.0, trainable=True, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.beta = K.variable(beta, dtype=K.floatx(), name="beta")
        self.trainable = trainable
        if self.trainable:
            self._trainable_weights.append(self.beta)

    def call(self, inputs, **kwargs):
        return inputs * keras.activations.tanh(
            self.beta * keras.activations.softplus(inputs)
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "beta": self.get_weights()[0] if self.trainable else self.beta,
            "trainable": self.trainable,
        }
        base_config = super().get_config()
        return {**base_config, **config}


register_custom_objects([mish, Mish, swish, Swish])

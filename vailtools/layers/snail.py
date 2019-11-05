"""
Implements the layers used to construct Simple Neural Attentive Meta-Learner (SNAiL)
architectures, which are intended to be used on meta-learning and reinforcement
learning tasks.
"""

import keras.backend as K
import numpy as np
from keras.layers import Dense, Conv1D, Layer, Softmax
from keras.layers.merge import concatenate, multiply
from keras.utils.generic_utils import get_custom_objects


class SnailAttentionBlock(Layer):
    """
    Implements the Attention Block used to build SNAIL architectures:
        https://arxiv.org/abs/1707.03141

    Borrowed from:
        https://github.com/philipperemy/keras-snail-attention/blob/master/attention.py
    """
    def __init__(self, key_size, value_size, **kwargs):
        self.key_size = key_size
        self.value_size = value_size
        self.sqrt_k = np.sqrt(key_size)
        self.keys_fc = None
        self.queries_fc = None
        self.values_fc = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        # https://stackoverflow.com/questions/54194724/how-to-use-keras-layers-in-custom-keras-layer
        self.keys_fc = Dense(self.key_size)
        self.keys_fc.build(input_shape)
        self._trainable_weights.extend(self.keys_fc.trainable_weights)

        self.queries_fc = Dense(self.key_size)
        self.queries_fc.build(input_shape)
        self._trainable_weights.extend(self.queries_fc.trainable_weights)

        self.values_fc = Dense(self.value_size)
        self.values_fc.build(input_shape)
        self._trainable_weights.extend(self.values_fc.trainable_weights)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # check that the implementation matches exactly py torch.
        keys = self.keys_fc(inputs)
        queries = self.queries_fc(inputs)
        values = self.values_fc(inputs)
        logits = K.batch_dot(queries, K.permute_dimensions(keys, (0, 2, 1)))
        mask = K.ones_like(logits) * np.triu((-np.inf) * np.ones(logits.shape.as_list()[1:]), k=1)
        logits = mask + logits
        probs = Softmax(axis=-1)(logits / self.sqrt_k)
        read = K.batch_dot(probs, values)
        output = K.concatenate([inputs, read], axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] += self.value_size
        return tuple(output_shape)


class SnailDenseBlock(Layer):
    """
    Implements the Dense Block used to build SNAIL architectures:
        https://arxiv.org/abs/1707.03141
    """
    def __init__(self, filters, dilation_rate, **kwargs):
        self.filters = filters
        self.dilation_rate = dilation_rate
        self.value_branch = None
        self.gate_branch = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.value_branch = Conv1D(
            filters=self.filters,
            kernel_size=2,
            dilation_rate=self.dilation_rate,
            padding='causal',
            activation='tanh',
        )
        self.gate_branch = Conv1D(
            filters=self.filters,
            kernel_size=2,
            dilation_rate=self.dilation_rate,
            padding='causal',
            activation='sigmoid',
        )

        self.value_branch.build(input_shape)
        self._trainable_weights.extend(self.value_branch.trainable_weights)
        self.gate_branch.build(input_shape)
        self._trainable_weights.extend(self.gate_branch.trainable_weights)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        activations = multiply([self.value_branch(inputs), self.gate_branch(inputs)])
        return concatenate([activations, inputs])

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] += self.filters
        return tuple(output_shape)


class SnailTCBlock(Layer):
    """
    Implements the Temporal Convolution Block used to build SNAIL architectures:
        https://arxiv.org/abs/1707.03141
    """
    def __init__(self, sequence_length, filters, **kwargs):
        self.sequence_length = sequence_length
        self.filters = filters
        self.layers = []
        layer_count = int(np.ceil(np.log2(self.sequence_length)))
        self.layer_count = max(layer_count, 1)
        super().__init__(**kwargs)

    def build(self, input_shape):
        output_shape = input_shape
        for i in range(self.layer_count):
            layer = SnailDenseBlock(filters=self.filters, dilation_rate=2**i)
            layer.build(output_shape)
            output_shape = layer.compute_output_shape(output_shape)
            self._trainable_weights.extend(layer.trainable_weights)
            self.layers.append(layer)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        pred = inputs
        for layer in self.layers:
            pred = layer(pred)
        return pred

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] += self.filters * self.layer_count
        return tuple(output_shape)


# Register custom Keras objects
# Should prevent the end user from needing to manually declare custom objects
# when saving and loading models made by or using VaiLTools
# Todo: May want to add some validation to ensure that builtin Keras objects are
#  not overwritten.
get_custom_objects().update({
    x.__name__: x
    for x in [SnailAttentionBlock, SnailDenseBlock, SnailTCBlock]
})

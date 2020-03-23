"""
Implements the layers used to construct Simple Neural Attentive Meta-Learner (SNAiL)
architectures, which are intended to be used on meta-learning and reinforcement
learning tasks.
"""

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.utils import get_custom_objects


class SnailAttentionBlock(layers.Layer):
    """
    Implements the Attention Block used to build SNAIL architectures:
        https://arxiv.org/abs/1707.03141

    Borrowed from:
        https://github.com/philipperemy/keras-snail-attention/blob/master/attention.py
    """

    def __init__(self, key_size, value_size, **kwargs):
        super().__init__(**kwargs)
        self.key_size = key_size
        self.value_size = value_size
        self.sqrt_k = np.sqrt(key_size)
        self.keys_fc = layers.Dense(self.key_size)
        self.queries_fc = layers.Dense(self.key_size)
        self.values_fc = layers.Dense(self.value_size)

    def call(self, inputs, **kwargs):
        # check that the implementation matches exactly py torch.
        keys = self.keys_fc(inputs)
        queries = self.queries_fc(inputs)
        values = self.values_fc(inputs)
        logits = K.batch_dot(queries, K.permute_dimensions(keys, (0, 2, 1)))
        mask = K.ones_like(logits) * np.triu(
            (-np.inf) * np.ones(logits.shape.as_list()[1:]), k=1
        )
        logits = mask + logits
        probs = layers.Softmax(axis=-1)(logits / self.sqrt_k)
        read = K.batch_dot(probs, values)
        output = K.concatenate([inputs, read], axis=-1)
        return output


class SnailDenseBlock(layers.Layer):
    """
    Implements the Dense Block used to build SNAIL architectures:
        https://arxiv.org/abs/1707.03141
    """

    def __init__(
        self, filters, dilation_rate, gate_merge=layers.Concatenate, **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.dilation_rate = dilation_rate
        self.gate_merge = gate_merge()
        self.value_branch = layers.Conv1D(
            filters=self.filters,
            kernel_size=2,
            dilation_rate=self.dilation_rate,
            padding="causal",
            activation="tanh",
        )

        self.gate_branch = layers.Conv1D(
            filters=self.filters,
            kernel_size=2,
            dilation_rate=self.dilation_rate,
            padding="causal",
            activation="sigmoid",
        )

    def call(self, inputs, **kwargs):
        activations = layers.Multiply()(
            [self.value_branch(inputs), self.gate_branch(inputs)]
        )
        return self.gate_merge([activations, inputs])


class SnailTCBlock(layers.Layer):
    """
    Implements the Temporal Convolution Block used to build SNAIL architectures:
        https://arxiv.org/abs/1707.03141
    """

    def __init__(self, sequence_length, filters, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.filters = filters
        layer_count = int(np.ceil(np.log2(self.sequence_length)))
        self.layer_count = max(layer_count, 1)
        self.layers = [
            SnailDenseBlock(filters=self.filters, dilation_rate=2 ** i)
            for i in range(self.layer_count)
        ]

    def call(self, inputs, **kwargs):
        pred = inputs
        for layer in self.layers:
            pred = layer(pred)
        return pred


# Register custom Keras objects
# Should prevent the end user from needing to manually declare custom objects
# when saving and loading models made by or using VaiLTools
# Todo: May want to ensure that builtin objects are not overwritten.
get_custom_objects().update(
    {x.__name__: x for x in [SnailAttentionBlock, SnailDenseBlock, SnailTCBlock]}
)

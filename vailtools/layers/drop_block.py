"""
Borrowed from https://github.com/CyberZHG/keras-drop-block on 2020/04/08.
Converted from keras to tensorflow.keras.

Todo: Loading models with DropBlock layers in them is not functional until TF 2.2.0
    due to a bug in the keras.models.load_model function.
    Reference: https://github.com/tensorflow/tensorflow/issues/37339
"""

from tensorflow import keras
from tensorflow.keras import backend as K

from ..utils import register_custom_objects


class DropBlock1D(keras.layers.Layer):
    """
    References:
        https://arxiv.org/pdf/1810.12890.pdf
    """

    def __init__(
        self, rate, block_size=None, sync_channels=False, dynamic=True, **kwargs
    ):
        """
        Args:
            rate: Probability of dropping the original feature.
            block_size: Size for each mask block.
            sync_channels: Whether to use the same dropout for all channels.
            **kwargs: Arguments for parent class.
        """
        super().__init__(dynamic=dynamic, **kwargs)
        self.block_size = block_size
        self.rate = rate
        self.sync_channels = sync_channels
        self.supports_masking = True

    def get_config(self):
        config = {
            "block_size": self.block_size,
            "rate": self.rate,
            "sync_channels": self.sync_channels,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        """
        Args:
          inputs: 
          mask: (Default value = None)

        Returns:
        """
        return mask

    def _get_gamma(self, feature_dim):
        """
        Get the number of activation units to drop

        Args:
          feature_dim: 

        Returns:
        """
        feature_dim = K.cast(feature_dim, K.floatx())
        block_size = K.constant(self.block_size, dtype=K.floatx())
        return (self.rate / block_size) * (
            feature_dim / (feature_dim - block_size + 1.0)
        )

    def _compute_valid_seed_region(self, seq_length):
        """
        Args:
          seq_length: 

        Returns:
        """
        positions = K.arange(seq_length)
        half_block_size = self.block_size // 2
        valid_seed_region = K.switch(
            K.all(
                K.stack(
                    [
                        positions >= half_block_size,
                        positions < seq_length - half_block_size,
                    ],
                    axis=-1,
                ),
                axis=-1,
            ),
            K.ones((seq_length,)),
            K.zeros((seq_length,)),
        )
        return K.expand_dims(K.expand_dims(valid_seed_region, axis=0), axis=-1)

    def _compute_drop_mask(self, shape):
        """
        Args:
          shape: 

        Returns:
        """
        seq_length = shape[1]
        mask = K.random_binomial(shape, p=self._get_gamma(seq_length))
        mask *= self._compute_valid_seed_region(seq_length)
        mask = keras.layers.MaxPool1D(
            pool_size=self.block_size, padding="same", strides=1,
        )(mask)
        return 1.0 - mask

    def call(self, inputs, training=None):
        """
        Args:
          inputs: 
          training: (Default value = None)

        Returns:
        """
        if self.block_size is None:
            spatial_dims = K.int_shape(inputs)[1:-1]
            self.block_size = int(sum(spatial_dims) / len(spatial_dims) / 4)

        def dropped_inputs():
            """ """
            data_format = K.image_data_format()
            outputs = inputs
            if data_format == "channels_first":
                outputs = K.permute_dimensions(outputs, [0, 2, 1])
            shape = K.shape(outputs)
            if self.sync_channels:
                mask = self._compute_drop_mask([shape[0], shape[1], 1])
            else:
                mask = self._compute_drop_mask(shape)
            outputs = (
                outputs * mask * (K.cast(K.prod(shape), dtype=K.floatx()) / K.sum(mask))
            )
            if data_format == "channels_first":
                outputs = K.permute_dimensions(outputs, [0, 2, 1])
            return outputs

        return K.in_train_phase(dropped_inputs, inputs, training=training)


class DropBlock2D(keras.layers.Layer):
    """
    References:
        https://arxiv.org/pdf/1810.12890.pdf
    """

    def __init__(
        self, rate, block_size=None, sync_channels=False, dynamic=True, **kwargs,
    ):
        """
        Args:
            rate: Probability of dropping the original feature.
            block_size: Size for each mask block.
            sync_channels: Whether to use the same dropout for all channels.
            **kwargs: Arguments for parent class.
        """
        super().__init__(dynamic=dynamic, **kwargs)
        self.block_size = block_size
        self.rate = rate
        self.sync_channels = sync_channels
        self.supports_masking = True

    def get_config(self):
        config = {
            "block_size": self.block_size,
            "rate": self.rate,
            "sync_channels": self.sync_channels,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_mask(self, inputs, mask=None):
        """
        Args:
          inputs: 
          mask: (Default value = None)

        Returns:
        """
        return mask

    def compute_output_shape(self, input_shape):
        """
        Args:
          input_shape: 

        Returns:
        """
        return input_shape

    def _get_gamma(self, height, width):
        """
        Get the number of activation units to drop

        Args:
          height: 
          width: 

        Returns:
        """
        height, width = K.cast(height, K.floatx()), K.cast(width, K.floatx())
        block_size = K.constant(self.block_size, dtype=K.floatx())
        return (self.rate / (block_size ** 2)) * (
            height * width / ((height - block_size + 1.0) * (width - block_size + 1.0))
        )

    def _compute_valid_seed_region(self, height, width):
        """
        Args:
          height: 
          width: 

        Returns:
        """
        positions = K.concatenate(
            [
                K.expand_dims(
                    K.tile(K.expand_dims(K.arange(height), axis=1), [1, width]), axis=-1
                ),
                K.expand_dims(
                    K.tile(K.expand_dims(K.arange(width), axis=0), [height, 1]), axis=-1
                ),
            ],
            axis=-1,
        )
        half_block_size = self.block_size // 2
        valid_seed_region = K.switch(
            K.all(
                K.stack(
                    [
                        positions[:, :, 0] >= half_block_size,
                        positions[:, :, 1] >= half_block_size,
                        positions[:, :, 0] < height - half_block_size,
                        positions[:, :, 1] < width - half_block_size,
                    ],
                    axis=-1,
                ),
                axis=-1,
            ),
            K.ones((height, width)),
            K.zeros((height, width)),
        )
        return K.expand_dims(K.expand_dims(valid_seed_region, axis=0), axis=-1)

    def _compute_drop_mask(self, shape):
        """
        Args:
          shape: 

        Returns:
        """
        height, width = shape[1], shape[2]
        mask = K.random_binomial(shape, p=self._get_gamma(height, width))
        mask *= self._compute_valid_seed_region(height, width)
        mask = keras.layers.MaxPool2D(
            pool_size=(self.block_size, self.block_size), padding="same", strides=1,
        )(mask)
        return 1.0 - mask

    def call(self, inputs, training=None):
        """
        Args:
          inputs: 
          training: (Default value = None)

        Returns:
        """
        if self.block_size is None:
            spatial_dims = K.int_shape(inputs)[1:-1]
            self.block_size = int(sum(spatial_dims) / len(spatial_dims) / 4)

        def dropped_inputs():
            data_format = K.image_data_format()
            outputs = inputs
            if data_format == "channels_first":
                outputs = K.permute_dimensions(outputs, [0, 2, 3, 1])
            shape = K.shape(outputs)
            if self.sync_channels:
                mask = self._compute_drop_mask([shape[0], shape[1], shape[2], 1])
            else:
                mask = self._compute_drop_mask(shape)
            outputs = (
                outputs * mask * (K.cast(K.prod(shape), dtype=K.floatx()) / K.sum(mask))
            )
            if data_format == "channels_first":
                outputs = K.permute_dimensions(outputs, [0, 3, 1, 2])
            return outputs

        return K.in_train_phase(dropped_inputs, inputs, training=training)


register_custom_objects([DropBlock1D, DropBlock2D])

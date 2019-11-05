from keras import layers
from keras.models import Model

from ..network_blocks import residual_block


def res_net(
        activation='selu',
        bias_initializer='zeros',
        blocks_per_layer=2,
        dense_neurons=512,
        depth=2,
        drop_prob=0.25,
        filters=64,
        final_activation='softmax',
        input_shape=(None, None, None),
        kernel_initializer='glorot_uniform',
        noise_std=0.,
        num_classes=10,
        residual_merge=layers.add,
):
    inputs = layers.Input(shape=input_shape)
    pred = layers.GaussianNoise(stddev=noise_std)(inputs)

    for _ in range(depth):
        for i in range(blocks_per_layer):
            pred = residual_block(
                pred,
                activation=activation,
                bias_initializer=bias_initializer,
                filters=filters,
                kernel_initializer=kernel_initializer,
                merge=residual_merge,
                project=not bool(i),
            )
        pred = layers.add([layers.MaxPool2D()(pred), layers.AvgPool2D()(pred)])
        filters *= 2

    pred = layers.Conv2D(
        bias_initializer=bias_initializer,
        filters=filters,
        kernel_initializer=kernel_initializer,
        kernel_size=1,
    )(pred)
    pred = layers.GlobalAveragePooling2D()(pred)

    pred = layers.Dense(dense_neurons)(pred)
    pred = layers.BatchNormalization()(pred)
    pred = layers.Activation(activation)(pred)
    pred = layers.Dropout(drop_prob)(pred)

    pred = layers.Dense(dense_neurons)(pred)
    pred = layers.BatchNormalization()(pred)
    pred = layers.Activation(activation)(pred)
    pred = layers.Dropout(drop_prob)(pred)

    pred = layers.Dense(num_classes)(pred)
    pred = layers.BatchNormalization()(pred)
    pred = layers.Activation(final_activation)(pred)
    return Model(inputs=inputs, outputs=pred)

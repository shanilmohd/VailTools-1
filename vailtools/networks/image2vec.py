from keras.layers import Activation, add, AvgPool2D, BatchNormalization, Conv2D, Dense, Dropout, MaxPool2D, Input, \
    GaussianNoise, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD

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
        loss='categorical_crossentropy',
        metrics=None,
        noise_std=0.,
        num_classes=10,
        optimizer=None,
        residual_merge=add,
):
    if optimizer is None:
        optimizer = SGD(momentum=0.9)

    input_ = Input(shape=input_shape)
    pred = GaussianNoise(stddev=noise_std)(input_)

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
        pred = add([MaxPool2D()(pred), AvgPool2D()(pred)])
        filters *= 2

    pred = Conv2D(
        bias_initializer=bias_initializer,
        filters=filters,
        kernel_initializer=kernel_initializer,
        kernel_size=1,
    )(pred)
    pred = GlobalAveragePooling2D()(pred)

    pred = Dense(dense_neurons)(pred)
    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)
    pred = Dropout(drop_prob)(pred)

    pred = Dense(dense_neurons)(pred)
    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)
    pred = Dropout(drop_prob)(pred)

    pred = Dense(num_classes)(pred)
    pred = BatchNormalization()(pred)
    pred = Activation(final_activation)(pred)

    model = Model(inputs=input_, outputs=pred)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

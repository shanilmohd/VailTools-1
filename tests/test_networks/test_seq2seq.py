import numpy as np

from vailtools.networks import snail_control, snail_mdp, snail_visual, wave_net


def test_wave_net(
    batch_size=4, batches=1, max_features=32, maxlen=32,
):
    model = wave_net(
        embedding_input_dim=max_features,
        final_activation="sigmoid",
        input_shape=(maxlen,),
        flatten_output=True,
        output_channels=1,
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    x_train = np.random.randint(0, max_features, (batches * batch_size, maxlen))
    y_train = np.random.randint(0, 2, size=(batches * batch_size, 1))

    model.fit(x_train, y_train, batch_size=batch_size)


def test_snail_control(
    batch_size=4, batches=1, max_features=32, maxlen=32,
):
    model = snail_control(
        embedding_input_dim=max_features,
        final_activation="sigmoid",
        input_shape=(maxlen,),
        output_size=1,
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    x_train = np.random.randint(0, max_features, (batches * batch_size, maxlen))
    y_train = np.random.randint(0, 2, size=(batches * batch_size, 1))

    model.fit(x_train, y_train, batch_size=batch_size)


def test_snail_mdp(
    batch_size=4, batches=1, max_features=32, maxlen=32,
):
    model = snail_mdp(
        embedding_input_dim=max_features,
        final_activation="sigmoid",
        input_shape=(maxlen,),
        output_size=1,
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    x_train = np.random.randint(0, max_features, (batches * batch_size, maxlen))
    y_train = np.random.randint(0, 2, size=(batches * batch_size, 1))

    model.fit(x_train, y_train, batch_size=batch_size)


def test_snail_visual(
    batch_size=4, batches=1, max_features=32, maxlen=32,
):
    model = snail_visual(
        embedding_input_dim=max_features,
        final_activation="sigmoid",
        input_shape=(maxlen,),
        output_size=1,
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    x_train = np.random.randint(0, max_features, (batches * batch_size, maxlen))
    y_train = np.random.randint(0, 2, size=(batches * batch_size, 1))

    model.fit(x_train, y_train, batch_size=batch_size)

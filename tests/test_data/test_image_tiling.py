import numpy as np
from tensorflow.keras.datasets import mnist

from vailtools.data import image_tiling


def test_image_to_tiles():
    (train_x, _), _ = mnist.load_data()
    train_x = train_x[..., np.newaxis]
    tiles = image_tiling.image_to_tiles(
        train_x[0], window_shape=np.array((14, 14)), step=None,
    )
    assert tiles.shape == (4, 14, 14, 1)


def test_tiles_to_image():
    (train_x, _), _ = mnist.load_data()
    train_x = train_x[..., np.newaxis]
    tiles = image_tiling.image_to_tiles(
        train_x[0], window_shape=np.array((14, 14)), step=None,
    )
    recon = image_tiling.tiles_to_image(
        tiles,
        image_shape=np.array((28, 28)),
        window_shape=np.array((14, 14)),
        step=None,
    )

    assert np.abs(train_x[0] - recon).sum() == 0
    assert train_x[0].shape == recon.shape


def test_tiles_per_image_usual():
    tiles = image_tiling.tiles_per_image(
        image_shape=np.array((64, 64)), window_shape=np.array((2, 2)), step=None,
    )

    assert tiles == 32 * 32


def test_tiles_per_image_uneven():
    tiles = image_tiling.tiles_per_image(
        image_shape=np.array((64, 63)), window_shape=np.array((2, 2)), step=None,
    )

    assert tiles == 32 * 32

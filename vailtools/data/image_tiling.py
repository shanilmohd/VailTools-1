"""
Provides functions that compute tile indices, cut an image into a stack of tiles
(tiling), and reconstruct an image from tiles (un-tiling).

Assumes images are 3D, i.e. (height, width, channels), and that tiling should be
performed over the spatial dimensions, i.e. (height, width).

The tiling implemented below always results in tiles that have the specified
spatial dimensions and cover the entire image. The default behavior is to minimize
the overlap between the tiles, i.e. step is identical to window_shape whenever possible,
though some overlap may occur near the edges of an image.
"""
import numpy as np


def image_to_tiles(
        image,
        window_shape=np.array((512, 512)),
        step=None,
):
    """
    Cuts an image into a stack of tiles where each tile has spatial dimensions,
    i.e. height and width, defined by window shape. The spacing of the tiles is
    controlled by step, which defaults to zero tile overlap. Images which are not
    cleanly covered by the provided configuration may result in the final column
    and/or row of tiles featuring a small overlap.

    Args:
        image: (numpy.ndarray) Image with shape (H, W, C)
        window_shape: (tuple[int, int])
        step: (tuple[int, int] or None)

    Returns: (numpy.ndarray)
        Stack of tiles with dimensions (N, Win_H, Win_W, C)
    """
    image_shape = image.shape[:-1]
    if step is None:
        step = window_shape.copy()
    if np.any(image_shape < window_shape):
        raise ValueError(f'All image dimensions, {image_shape}, must be greater'
                         f' than all window dimensions, {window_shape}!')
    slices_gen = generate_tile_slices(
        image_shape=image_shape,
        window_shape=window_shape,
        step=step,
    )
    tiles = [image[slices] for slices in slices_gen]
    return np.stack(tiles)


def tiles_to_image(
        tiles,
        image_shape=np.array((904, 1224)),
        window_shape=np.array((512, 512)),
        step=None,
        dtype=float
):
    """
    Reconstructs an image from a stack of tiles.

    Args:
        tiles: (numpy.ndarray) Stack of tiles with shape (N, Win_H, Win_W, C)
        image_shape: (tuple[int, int, int]) Image with shape (H, W, C)
        window_shape: (tuple[int, int])
        step: (tuple[int, int] or None)
        dtype: Any valid numpy.ndarray dtype

    Returns: (numpy.ndarray)
        Image with dimensions equal to image_shape, populated from tiles
    """
    if step is None:
        step = window_shape.copy()
    if np.any(image_shape < window_shape):
        raise ValueError(f'All image dimensions, {image_shape}, must be greater'
                         f' than all window dimensions, {window_shape}!')
    slices_list = list(generate_tile_slices(
        image_shape=image_shape,
        window_shape=window_shape,
        step=step,
    ))
    if len(slices_list) != len(tiles):
        raise ValueError(f'Mismatch between tiles {len(tiles)} and generated '
                         f'slices {len(slices_list)}')

    image = np.zeros(image_shape, dtype=dtype)
    counts = np.zeros(image_shape, dtype=float)
    for tile, slices in zip(tiles, slices_list):
        image[slices] += tile
        counts[slices] += 1.
    return image / counts


def generate_tile_slices(
        image_shape=np.array((904, 1224)),
        window_shape=np.array((512, 512)),
        step=None,
):
    """
    Computes and yields slices that correspond to tiles of shape window_shape,
    spaced by step, located in an image with dimensions image_shape.

    Args:
        image_shape: (tuple[int, int, int])
        window_shape: (tuple[int, int])
        step: (tuple[int, int] or None)

    Yields: (tuple[slice, slice, slice])
        Slices corresponding to a single 3D tile
    """
    if step is None:
        step = window_shape.copy()
    rows, cols = image_shape // step
    odd_row, odd_col = image_shape % step
    for row in range(rows):
        for col in range(cols):
            yield (
                slice(row * step[0], row * step[0] + window_shape[0]),
                slice(col * step[1], col * step[1] + window_shape[1]),
                slice(None),
            )
        if odd_col:
            yield (
                slice(row * step[0], row * step[0] + window_shape[0]),
                slice(-window_shape[1], None),
                slice(None),
            )

    if odd_row:
        for col in range(cols):
            yield (
                slice(-window_shape[0], None),
                slice(col * step[1], col * step[1] + window_shape[1]),
                slice(None),
            )
        if odd_col:
            yield (
                slice(-window_shape[0], None),
                slice(-window_shape[1], None),
                slice(None),
            )


def tiles_per_image(
        image_shape=np.array((904, 1224)),
        window_shape=np.array((512, 512)),
        step=None,
):
    """
    Args:
        image_shape: (numpy.ndarray) Spatial dimensions of an image
        window_shape: (numpy.ndarray) Spatial dimensions of a window
        step: (numpy.ndarray) Step size to be used in each spatial dimension

    Returns: (int)
        Number of tiles that will be generated
    """
    if step is None:
        step = window_shape.copy()
    if np.any(image_shape < window_shape):
        raise ValueError(f'All image dimensions, {image_shape}, must be greater'
                         f' than all window dimensions, {window_shape}!')
    counts = np.ceil(image_shape / step)
    return int(counts.prod())

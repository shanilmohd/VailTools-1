"""
Manages the construction of random dataset splits (i.e. training, validation, testing).
Supports serialization and deserialization of splits using .npy files and provides a convenience
function for applying splits to arbitrary numpy arrays.
"""

from collections import namedtuple
from pathlib import Path

import numpy as np


SplitIndices = namedtuple('SplitIndices', ['train', 'val', 'test'])


def generate_splits(samples, train_frac=None, val_frac=None, test_frac=None):
    if train_frac is None:
        train_frac = 1. - (val_frac + test_frac)
    elif val_frac is None:
        val_frac = 1. - (train_frac + test_frac)
    elif test_frac is None:
        test_frac = 1. - (train_frac + val_frac)
    else:
        total = train_frac + val_frac + test_frac
        train_frac /= total
        val_frac /= total
        test_frac /= total

    if train_frac + val_frac + test_frac != 1.:
        raise ValueError(
            'Split fractions should sum to 1, but {} + {} + {} = {}!'.format(
                train_frac, val_frac, test_frac, train_frac + val_frac + test_frac))

    train_count = int(np.ceil(samples * train_frac))
    val_count = int(np.floor(samples * val_frac))
    inds = np.random.permutation(samples)
    return SplitIndices(
        train=inds[:train_count],
        val=inds[train_count:train_count + val_count],
        test=inds[train_count + val_count:],
    )


def load_splits(file):
    path = Path(file)
    assert path.is_file() and (path.suffix == '.npz')
    return SplitIndices(
        **np.load(file)
    )


DataSplit = namedtuple('DataSplit', ['x', 'y'])


def split_data(x, y, splits=None, save_splits=False, save_path='.', save_name=None):
    if splits is None:
        splits = generate_splits(len(x), train_frac=.6, val_frac=.2, test_frac=.2)

    if save_splits:
        if save_name is None:
            save_name = 'splits'
        np.savez_compressed(f'{save_path}/{save_name}.npz', **splits._asdict())
    return (
        DataSplit(x[splits.train], y[splits.train]),
        DataSplit(x[splits.val], y[splits.val]),
        DataSplit(x[splits.test], y[splits.test]),
    )
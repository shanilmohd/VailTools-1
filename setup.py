#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
from pathlib import Path

import setuptools


# Package meta-data.
NAME = 'vailtools'
DESCRIPTION = 'Components, tools, and utilities for building, training, and testing artificial neural networks.'
URL = 'https://gitlab.com/vail-uvm/vail-tools'
EMAIL = 'vail.csds@gmail.com'
AUTHOR = 'Vermont Artificial Intelligence Laboratory'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = 0.1

# Required packages
REQUIRED = [
    'keras', 'tensorflow', 'numpy'
]

# Optional packages
EXTRAS = {
}

# Path to package top-level
here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

# Ensure that Git submodules are correctly initialized
p = Path(f'{here}/vailtools/losses/LovaszSoftmax/__init__.py')
if not p.is_file():
    p.write_text('from .tensorflow.lovasz_losses_tf import lovasz_hinge, lovasz_softmax\n')

setuptools.setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=setuptools.find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
)

# VaiL Tools
Components, tools, and utilities for building, training, and testing artificial neural networks in Python.

The primary goal of this package is to provide reference implementations of tools, tricks,
and architectures, allowing researchers to quickly and easily test a variety of configurations.
In support of this, VaiL Tools prioritizes modularity, usability, and clarity.

Please open an issue or contact us if you encounter any problems with the package or have a
feature request.

Vail Tools is built on top of TensorFlow, and primarily targets TensoFlow.Keras.


## Contents
| Name       | Description                                                                             |
| ---        | ---                                                                                     |
| callbacks  | Specialized learning rate schedulers and other callbacks used by keras.models.Model.fit |
| data       | Data management utilities such as image tiling and data split generation.               |
| evaluation | Utilities applied at test-time, such as test-time data augmentation.                    |
| layers     | Building blocks for neural networks such as residual blocks, dense blocks, etc.         |
| losses     | Loss functions used to train neural network models.                                     |
| metrics    | Metrics useful for monitoring training performance.                                     |
| networks   | Fully constructed and compiled Keras models.                                            |


## Installation
The prefered installation method is via `pip`:
```bash
pip install git+https://gitlab.com/vail-uvm/vailtools.git
```
This should get things running quickly and easily, though the normal complications with
getting GPU-accelerated TensorFlow are still present.

The easiest way to get things working with GPU-acceleration is to use [Anaconda](https://www.anaconda.com/),
which provides a convenient `tensorflow-gpu` package.

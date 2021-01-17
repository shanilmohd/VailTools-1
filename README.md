# VaiL Tools
Components, tools, and utilities for building, training, and testing artificial
neural networks in Python.
VaiL Tools is built on top of TensorFlow, and targets TensorFlow.Keras and the
TensorFlow 2.0 API.

The primary goal of this package is to provide reference implementations of tools,
tricks, and architectures, allowing researchers to quickly and easily test a
variety of configurations.
In support of this, VaiL Tools prioritizes modularity, usability, and clarity.

Please open an issue or contact us if you encounter any problems or have a
feature request.

## Installation
The preferred installation method is via `pip`:
```bash
pip install git+https://gitlab.com/vail-uvm/vailtools.git
```
This should get things running quickly and easily, though the normal complications with
getting GPU-accelerated TensorFlow are still present.

The easiest way to get things working with GPU-acceleration is to use [Anaconda](https://www.anaconda.com/),
which provides a convenient `tensorflow-gpu` package.

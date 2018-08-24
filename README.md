# VaiL Tools
Components, tools, and utilities for building, training, and testing artificial neural networks.

## Supported Frameworks
Vail Tools primarily targets Keras with the Tensorflow backend, though support for other framework 
configurations may be expanded in the future.

## Installation
Please clone the repository then install it locally using the following commands:
```bash
git clone git@gitlab.com:vail-uvm/vailtools.git
cd VailTools
pip install -e .
```
Or if you do not have an SSH key setup for GitLab, then use:
```bash
git clone https://gitlab.com/vail-uvm/vailtools.git
cd VailTools
pip install -e .
```

## Contents
| Name           | Description                                                                                      |
| ---            | ---                                                                                              |
| callbacks      | Specialized learning rate schedulers and other callbacks used by keras.models.Model.fit          |
| data           | Data management utilities such as image tiling and data split generation.                        |
| evaluation     | Utilities applied at test-time, such as test-time data augmentation.                             |
| losses         | Loss functions implemented in Keras, Tensorflow, and Numpy.                                      |
| metrics        | Metrics useful for monitoring training performance, implemented in Keras, Tensorflow, and Numpy. |
| network_blocks | Building blocks for neural networks such as residual blocks, dense blocks, etc.                  |
| networks       | Fully constructed and compiled Keras models.                                                     |

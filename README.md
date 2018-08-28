# VaiL Tools
Components, tools, and utilities for building, training, and testing artificial neural networks in Python.

## Supported Frameworks
Vail Tools primarily targets Keras with the Tensorflow backend, though support for other framework 
configurations may be expanded in the future.

## Requirements
VaiL Tools has been tested with the following configuration, though it may function with a wider range of versions for certain packages.

| Name       | Version   |
| ---        | ---       |
| Python     | \>=3.6.0  |
| Numpy      | \>=1.14.5 |
| Tensorflow | \>=1.8    |
| Keras      | \>=2.0.0  |


## Installation
Please clone the repository then install it locally using the following commands:
```bash
git clone git@gitlab.com:vail-uvm/vailtools.git
cd vailtools
git submodule init
git submodule update
pip install -e .
```
If you do not have an SSH key setup for GitLab, then use:
```bash
git clone https://gitlab.com/vail-uvm/vailtools.git
cd vailtools
git submodule init
git submodule update
pip install -e .
```

__Note:__ Installing VaiL Tools with the above commands will result in the installation of 
Numpy, Keras, and/or Tensorflow, if any of these packages are not already installed.
Due to issues that may be encountered when installing these packages,
users may wish to install and test them prior to using VaiL Tools.
[Anaconda](https://www.anaconda.com/) provides a convenient solution to this,
users are encouraged to follow the Anaconda setup instructions provided in
[this blog post](https://johnhringiv.com/installing_tensorflow.php). 

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

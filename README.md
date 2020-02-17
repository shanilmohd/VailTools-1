# VaiL Tools
Components, tools, and utilities for building, training, and testing artificial neural networks in Python.

The primary goal of this package is to provide reference implementations of tools, tricks,
and architectures, allowing researchers to quickly and easily test a variety of configurations.
In support of this, VaiL Tools prioritizes modularity, usability, and clarity.

Please open an issue or contact us if you encounter any problems with the package or have a
feature request.

## Supported Frameworks
Vail Tools primarily targets Keras with the Tensorflow backend, though support for other framework 
configurations may be expanded in the future.

## Contents
| Name       | Description                                                                             |
| ---        | ---                                                                                     |
| callbacks  | Specialized learning rate schedulers and other callbacks used by keras.models.Model.fit |
| data       | Data management utilities such as image tiling and data split generation.               |
| evaluation | Utilities applied at test-time, such as test-time data augmentation.                    |
| layers     | Building blocks for neural networks such as residual blocks, dense blocks, etc.         |
| losses     | Loss functions implemented in Keras, Tensorflow, and Numpy.                             |
| metrics    | Metrics useful for monitoring training performance.                                     |
| networks   | Fully constructed and compiled Keras models.                                            |

## Requirements
VaiL Tools has been tested with the following configuration, though it may function with a wider range of versions for certain packages.

| Name       | Version   |
| ---        | ---       |
| Python     | \>=3.6.0  |
| Numpy      | \>=1.14.5 |
| Tensorflow | \>=1.8    |
| Keras      | \>=2.0.0  |


## Installation
Clone the repository then install it locally using the following commands:
```bash
git clone git@gitlab.com:vail-uvm/vailtools.git
cd vailtools
bash install.sh
```
If you do not have SSH setup for GitLab, then use:
```bash
git clone https://gitlab.com/vail-uvm/vailtools.git
cd vailtools
bash install.sh
```

__Note:__ Installing VaiL Tools with the above commands will result in the installation of 
Numpy, Keras, and/or Tensorflow, if any of these packages are not already installed.
Due to issues that may be encountered when installing these packages,
users may wish to install and test them prior to using VaiL Tools.
[Anaconda](https://www.anaconda.com/) provides a convenient solution to this,
users are encouraged to follow the Anaconda setup instructions provided in
[this blog post](https://johnhringiv.com/installing_tensorflow.php).

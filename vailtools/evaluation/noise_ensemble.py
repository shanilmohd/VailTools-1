"""
Provides tools that may be applied to Keras models at test/evaluation time.
"""

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def noise_ensemble(model, x, batch_size=32, noise_std=0.01, count=30, verbose=False):
    """
    Implements test-time data augmentation using Gaussian noise for arbitrary Keras
    models that may be applied post-training without any additional model configuration.

    Note that ImageDataGenerator yields partial batches, so more batches may
    required to obtain the desired number of predictions than expected.

    Tested with noise_std of 0.001 and 0.1 and observed corresponding increase in
    prediction std. dev., indicating that noise is being applied correctly.

    Results are inconclusive so far as to whether this provides an improvement
    in prediction quality, though the addition of uncertainty information via
    the standard deviation can be helpful.
    The noise level used has a potentially large effect on the results and selecting
    the correct noise level should be considered carefully.

    Exhausts 16GB of CPU memory very quickly as the size of the test set increases.
    Should work on a streaming version that works on a generator and works one sample
    at a time.

    Args:
        model: (keras.models.Model)
        x: (numpy.ndarray)
        batch_size: (int)
        noise_std: (float)
        count: (int)
        verbose: (bool)

    Returns: (tuple[numpy.ndarray])
        Mean and standard deviation over count predictions for each sample in x.
    """
    gen = ImageDataGenerator(preprocessing_function=get_noise_func(noise_std))
    pred = model.predict_generator(
        gen.flow(x, batch_size=batch_size, shuffle=False),
        steps=count * int(np.ceil(len(x) / batch_size)),
        verbose=verbose,
    )
    # Truncate the number of predicted slices if necessary
    if len(pred) % (len(x) * count):
        pred = pred[: -(len(pred) % (len(x) * count))]

    # Split the stack of predictions into one chunk per iteration
    pred = np.stack(np.split(pred, count), axis=-1)
    return np.mean(pred, axis=-1), np.std(pred, axis=-1)


def get_noise_func(noise_std):
    """
    Creates a function that takes a numpy.ndarray and returns that array after
    applying an additive Gaussian noise with zero mean and the provided standard
    deviation.

    Args:
        noise_std: (float) Standard deviation of the Gaussian noise.

    Returns: (Callable[[numpy.ndarray], numpy.ndarray])
    """

    def noise_func(img):
        return img + np.random.normal(0, noise_std, img.shape)

    return noise_func

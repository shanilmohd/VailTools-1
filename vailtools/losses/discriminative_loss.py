"""
A Tensorflow implementation of the loss function described in 'Semantic Instance
Segmentation with a Discriminative Loss Function'.

See https://arxiv.org/abs/1708.02551 for details.
"""


import tensorflow as tf


def discriminative_loss(y_true, y_pred):
    """
    Computes the discriminative loss for a batch of images.

    Args:
        y_true: (tensorflow.Tensor) True instance labels, shape=(B, N, N, 1)
        y_pred: (tensorflow.Tensor) Embedded pixels, shape=(B, N, N, E)

    Returns: (tensorflow.Tensor)
        Discriminative loss for each sample, shape=(B, 1).
    """
    return tf.map_fn(
        sample_loss,
        (y_true, y_pred),
        dtype=tf.float32,
        parallel_iterations=10,
    )


def sample_loss(
    vals,
    alpha=1.,
    beta=1.,
    gamma=0.0001,
    delta_v=0.5,
    delta_d=3.0,
):
    """
    Computes the discriminative loss for a single image.

    Args:
        vals: (tuple)
        alpha: (float) Variance term weight.
        beta: (float) Distance term weight.
        gamma: (float) Regularization term weight.
        delta_v: (float) Variance term hinge length.
        delta_d: (float) Distance term hinge length.

    Returns: (tensorflow.scalar)
        Discriminative loss for a single sample.
    """
    y_true, y_pred = vals
    centroids = compute_centroids(y_true, y_pred)
    return (
        alpha * var_loss(y_true, y_pred, centroids, delta_v) +
        beta * dist_loss(centroids, delta_d) +
        gamma * reg_loss(centroids)
    )


def var_loss(y_true, y_pred, centroids, delta):
    """
    Computes the intra-cluster pull term for the discriminative loss function.

    Args:
        y_true: (tensorflow.Tensor) True instance labels, shape=(N, N, 1)
        y_pred: (tensorflow.Tensor) Embedded pixels, shape=(N, N, E)
        centroids: (tensorflow.Tensor) Mean of C E-dimensional clusters, shape=(C, E)
        delta: (tensorflow.scalar) Hinge length.

    Returns: (tensorflow.scalar)
        Variance term for the discriminative loss function.
    """
    vals, idx = tf.unique(tf.reshape(y_true, [tf.size(y_true)]))
    y_true = tf.squeeze(y_true)
    # Specify dimensions, since tf.boolean_mask requires that they be known statically
    y_true.set_shape([None, None])

    def loop_body(i, loss):
        instance = tf.boolean_mask(y_pred, tf.equal(y_true, vals[i]))
        loss_ = safe_norm(instance - centroids[i], axis=-1)
        loss_ = tf.square(tf.nn.relu(loss_ - delta))
        return [i + 1, loss.write(i, tf.reduce_mean(loss_))]

    loop_vars = [tf.constant(0), tf.TensorArray(dtype=tf.float32, size=tf.size(vals))]
    _, losses = tf.while_loop(
        lambda i, loss_: tf.less(i, tf.size(vals)),
        loop_body,
        loop_vars,
        parallel_iterations=100,
    )
    return tf.reduce_mean(losses.stack())


def dist_loss(centroids, delta):
    """
    Computes the inter-cluster push term for the discriminative loss function.

    Credit to:
        https://stackoverflow.com/questions/43534057/evaluate-all-pair-combinations-of-rows-of-two-tensors-in-tensorflow

    Args:
        centroids: (tensorflow.Tensor) Mean of C E-dimensional clusters, shape=(C, E)
        delta: (tensorflow.scalar) Hinge length.

    Returns: (tensorflow.scalar)
        Distance term for the discriminative loss function.
    """
    diff = tf.expand_dims(centroids, 0) - tf.expand_dims(centroids, 1)
    diff = tf.reshape(diff, [-1, tf.shape(centroids)[-1]])
    masked = tf.boolean_mask(diff, tf.reduce_sum(diff, axis=-1) > 0)
    return tf.reduce_mean(tf.square(tf.nn.relu(2. * delta - safe_norm(masked, axis=-1))))


def reg_loss(centroids):
    """
    Computes the regularization term for the discriminative loss function.

    Args:
        centroids: (tensorflow.Tensor) Mean of C E-dimensional clusters, shape=(C, E)

    Returns: (tensorflow.scalar)
        Regularization term for the discriminative loss function.
    """
    return tf.reduce_mean(safe_norm(centroids, axis=-1))


def compute_centroids(y_true, y_pred):
    """
    Args:
        y_true: (tensorflow.Tensor) True instance labels, shape=(N, N, 1)
        y_pred: (tensorflow.Tensor) Embedded pixels, shape=(N, N, E)

    Returns: (tensorflow.Tensor)
        Mean of each cluster, shape=(C, E).
    """
    # unique_with_counts expects a vector (1-D Tensor)
    y_true = tf.reshape(y_true, [tf.size(y_true)])
    y, idx, counts = tf.unique_with_counts(y_true)

    # unsorted_segment_sum expects shape(idx) = shape(y_pred)[:N] for some N
    idx = tf.reshape(idx, tf.shape(y_pred)[:2])
    return tf.unsorted_segment_sum(y_pred, idx, tf.size(y)) / tf.expand_dims(tf.to_float(counts), axis=-1)


def safe_norm(x, ord_=2, axis=None, keep_dims=False, eps=1e-7, name='safe_norm'):
    """
    Computes the ord-norm of the input.
    Used in place of tensorflow.norm in order to avoid gradient instabilities.

    Credit to https://stackoverflow.com/questions/46359221/tensorflow-loss-function-zeroes-out-after-first-epoch

    Args:
        x: (tensorflow.Tensor) Expected to have at least two dimensions (Batch size, sample dimension).
        ord_: (numeric) Order of the norm.
        axis: (int) axis over which the norm will be calculated.
        keep_dims: (bool) Keep reduced dimensions or drop reduced dimensions.
        eps: (float) Small value to stabilize computation.
        name: (str) Name assigned to the operation in the computation graph.

    Returns: (tensorflow.Tensor)
        ord-norm of the input, identical dimensions except for axis which now has a size of 1.
    """
    inner = tf.reduce_sum(tf.pow(x, ord_), axis=axis, keep_dims=keep_dims)
    normed = tf.pow(inner + eps, 1. / ord_)
    return tf.identity(normed, name=name)

"""knet definition
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim


def knet_layer(pairwise_features, n_pair_features,
               n_kernels=1, hlayer_size=20, softmax_kernel=True):

    n_objects = tf.shape(pairwise_features)[0]

    pairwise_features_reshaped = tf.reshape(
        pairwise_features, [
            1, n_objects, n_objects, n_pair_features])

    conv1 = slim.layers.conv2d(
        pairwise_features_reshaped,
        hlayer_size,
        [1, 1],
        activation_fn=tf.nn.relu)

    conv2 = slim.layers.conv2d(
        conv1,
        n_kernels,
        [1, 1],
        activation_fn=None)

    conv2_t = tf.transpose(conv2)

    if softmax_kernel:
        kernels = tf.nn.softmax(
            tf.reshape(
                conv2_t, [
                    n_kernels, n_objects, n_objects]))
    else:
        kernels = tf.reshape(conv2, [n_kernels, n_objects, n_objects])

    return kernels


def apply_kernel(kernels, object_features, n_kernels, n_object_features):
    """
    Parameters
    ----------
    kernels : Tensor containing kernel matrix to apply
    object_features : Tensor containing object features
    n_kernels : number of kernels
    n_object_features : number of object features

    Returns
    -------
    transformed_features_flat : Tensor containing features after kernel application
    """

    n_objects = tf.shape(object_features)[0]

    object_features_reshaped = tf.reshape(object_features, [1, n_objects, -1])

    object_features_broadcasted = tf.tile(
        object_features_reshaped, [n_kernels, 1, 1])

    transformed_features = tf.batch_matmul(
        kernels, object_features_broadcasted)

    transformed_features_t = tf.transpose(transformed_features, perm=[1, 0, 2])
    transformed_features_flat = tf.reshape(
        transformed_features_t, [
            n_objects, n_kernels * n_object_features])

    return transformed_features_flat

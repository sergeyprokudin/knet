"""knet definition
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim

def knet_layer(object_features, pairwise_features, n_kernels=1, n_neurons=20):
    """Basic knet layer

    Parameters :


    """
    inference = object_features

    kernel_1 = slim.layers.conv2d(
        pairwise_features,
        n_kernels * n_neurons,
        [1, 1],
        scope='conv_3',
        # weights_initializer=tf.truncated_normal_initializer(stddev=0.0001),
        activation_fn=tf.nn.relu)

    kernel_2 = slim.layers.conv2d(
        kernel_1,
        n_kernels,
        [1, 1],
        scope='conv_4',
        # weights_initializer=tf.truncated_normal_initializer(stddev=0.0001),
        activation_fn=None)

    kernel_2_reshape = tf.reshape(kernel_2, [n_kernels, -1, -1])

    inference = tf.mul(kernel_2_reshape, object_features)

    return inference

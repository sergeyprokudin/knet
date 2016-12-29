"""knet definition
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def batch_split(data, batch_size):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def prelu(data, name_or_scope=None):
    with tf.variable_scope(
            name_or_scope,
            default_name='prelu',
            values=[data]):
        alphas = tf.get_variable(shape=data.get_shape().as_list()[-1:],
                                 initializer=tf.constant_initializer(0.01),
                                 name="alphas")

        return tf.nn.relu(data) + tf.mul(alphas, (data - tf.abs(data))) * 0.5

def knet_layer(object_features, pairwise_features, n_objects, n_pair_features, n_object_features, n_kernels=1, hlayer_size=20):
    """Basic knet layer
    """
    knet_ops = {}

    knet_ops['object_features'] = object_features

    pairwise_features_reshaped = tf.reshape(pairwise_features, [1, n_objects, n_objects, n_pair_features])

    knet_ops['pairwise_features'] = pairwise_features
    knet_ops['pairwise_features_reshaped'] = pairwise_features_reshaped

    W_conv1 = weight_variable([1, 1, n_pair_features, hlayer_size])
    b_conv1 = bias_variable([hlayer_size])
    h_conv1 = conv2d(pairwise_features_reshaped, W_conv1) + b_conv1
    h_conv1_relu = tf.nn.relu(h_conv1)
    knet_ops['W_conv1'] = W_conv1
    knet_ops['h_conv1'] = h_conv1
    knet_ops['h_conv1_relu'] = h_conv1_relu

    W_conv2 = weight_variable([1, 1, hlayer_size, n_kernels])
    b_conv2 = bias_variable([n_kernels])
    h_conv2 =  conv2d(h_conv1_relu, W_conv2) + b_conv2
    knet_ops['W_conv2'] = W_conv2
    knet_ops['h_conv2'] = h_conv2

    h_conv2_t = tf.transpose(h_conv2)
    kernels = tf.reshape(h_conv2_t,[n_kernels, n_objects, n_objects])

    knet_ops['kernels'] = kernels

    object_features_reshaped = tf.reshape(object_features, [1,n_objects,-1])
    knet_ops['object_features_reshaped'] = object_features_reshaped
    object_features_broadcasted = tf.tile(object_features_reshaped, [n_kernels,1,1])
    knet_ops['object_features_broadcasted'] = object_features_broadcasted

    transformed_features = tf.batch_matmul(kernels , object_features_broadcasted)

    transformed_features_t = tf.transpose(transformed_features, perm=[1,0,2])
    transformed_features_flat = tf.reshape(transformed_features_t, [n_objects, n_kernels*n_object_features])

    knet_ops['transformed_features'] = transformed_features
    knet_ops['transformed_features_flat'] = transformed_features_flat

    return transformed_features_flat, knet_ops

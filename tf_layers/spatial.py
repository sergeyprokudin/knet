"""Collection of tf ops for performing operaion on spatial features
"""

import  tensorflow as tf


def construct_pairwise_features_tf(features,
                                   name_or_scope=None):
    """Construct pairwise features matrix

    Parameters
    -------
    features - Tensor of shape [n_objects, n_features]

    Returns
    --------
    Tensor of shape [1, n_objects, n_objects, n_features*2] (ready for knet convolution)
    """
    with tf.variable_scope(name_or_scope,
                           default_name='pairwise_features',
                           values=[features]):
        # We only support flat features.
        features.get_shape().assert_has_rank(2)
        feature_shape = tf.shape(features)
        n_hypotheses = feature_shape[0]
        n_features = features.get_shape().as_list()[1]

        pair_1 = tf.tile(features, [1, n_hypotheses])
        pair_1 = tf.reshape(pair_1, [-1, n_features])

        pair_2 = tf.tile(features, [n_hypotheses, 1])
        pairwise_features = tf.concat(1, [pair_1, pair_2])

        return tf.reshape(
            pairwise_features,
            [1, n_hypotheses, n_hypotheses, n_features * 2])

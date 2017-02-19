"""Collection of tf ops for performing operaion on spatial features
"""

import tensorflow as tf


def construct_pairwise_features_tf(features_1,
                                   features_2=None,
                                   name_or_scope=None):
    """Construct pairwise features matrix

    Parameters
    -------
    features_1 - Tensor of shape [n1_objects, n_features]
    features_2 - second Tensor of shape [n2_objects, n_features].
                If None, construct pairwise terms from first matrix only
    Returns
    --------
    Tensor of shape [1, n1_objects, n2_objects, n_features*2] (ready for knet convolution)
    """
    with tf.variable_scope(name_or_scope,
                           default_name='pairwise_features',
                           values=[features_1]):
        if features_2 is None:
            features_2 = features_1
        # We only support flat features.
        features_1.get_shape().assert_has_rank(2)
        t1_shape = tf.shape(features_1)
        t2_shape = tf.shape(features_2)
        n1_hypotheses = t1_shape[0]
        n2_hypotheses = t2_shape[0]
        n_features = features_1.get_shape().as_list()[1]

        pair_1 = tf.tile(features_1, [1, n2_hypotheses])
        pair_1 = tf.reshape(pair_1, [-1, n_features])

        pair_2 = tf.tile(features_2, [n1_hypotheses, 1])
        pairwise_features = tf.concat(1, [pair_1, pair_2])

        return tf.reshape(
            pairwise_features,
            [n1_hypotheses, n2_hypotheses, n_features * 2])


def compute_overlap(zeros, pos_1, dim_1, pos_2, dim_2, name_or_scope=None):
    with tf.variable_scope(name_or_scope,
                           default_name='compute_overlap',
                           values=[pos_1, dim_1, pos_2, dim_2]):
        # We assert that all shapes are fine.
        zeros.get_shape().assert_is_compatible_with(pos_1.get_shape())
        zeros.get_shape().assert_is_compatible_with(dim_1.get_shape())
        zeros.get_shape().assert_is_compatible_with(pos_2.get_shape())
        zeros.get_shape().assert_is_compatible_with(dim_2.get_shape())

        pos_1_end = tf.add(pos_1, dim_1, name='pos_1_end')
        pos_2_end = tf.add(pos_2, dim_2, name='pos_2_end')

        max_pos_start = tf.maximum(pos_1, pos_2, name='max_pos_start')
        min_pos_end = tf.minimum(pos_1_end, pos_2_end, name='min_pos_ned')
        return tf.maximum(zeros, min_pos_end - max_pos_start, name='overlap')


def compute_intersection(x_overlap, y_overlap, name_or_scope=None):
    with tf.variable_scope(name_or_scope,
                           default_name='compute_intersection',
                           values=[x_overlap, y_overlap]):
        return tf.mul(x_overlap, y_overlap)


def compute_union(width_1, height_1, width_2, height_2,
                  intersection, name_or_scope=None):
    with tf.variable_scope(name_or_scope,
                           default_name='compute_union',
                           values=[width_1,
                                   height_1,
                                   width_2,
                                   height_2,
                                   intersection]):
        rect_1 = tf.mul(width_1, height_1, name='rectangle_1')
        rect_2 = tf.mul(width_2, height_2, name='rectangle_2')
        total = tf.add(rect_1, rect_2, name='rectangle_1_2')
        return tf.sub(total, intersection, name='union')


def compute_pairwise_spatial_features_iou_tf(
        pairwise_spatial_features, name_or_scope=None):
    """

    Parameters
    ----------
    pairwise_spatial_features - Tensor of format [n1_hypotheses, n2_hypotheses, 8]
                                where last dimension contains information about coordinates of a pair of
                                boxes in format [x11, y11, x12, y12, x21, y21, x22, y22]
    name_or_scope

    Returns
    -------
    Tensor of format [n1_hypotheses, n2_hypotheses, 1] containing information on intersection
    over union for each bbox pair
    """
    with tf.variable_scope(
            name_or_scope,
            default_name='construct_pairwise_spatial_features_iou',
            values=[pairwise_spatial_features]):
        # We only support flat features and thus have to have a rank of 3
        # [num_hypotheses, num_hypotheses, feature_size].
        pairwise_spatial_features.get_shape().assert_has_rank(3)

        # Assert ops cannot be executed on gpus
        # assert_ops = []
        # pairwise_spatial_features_shape = tf.shape(pairwise_spatial_features)
        # assert_ops.append(
        #     tf.Assert(tf.equal(pairwise_spatial_features_shape[0],
        #                        pairwise_spatial_features_shape[1]),
        #               [pairwise_spatial_features_shape]))

        # with tf.control_dependencies(assert_ops):
        x1 = pairwise_spatial_features[:, :, 0]
        y1 = pairwise_spatial_features[:, :, 1]
        w1 = pairwise_spatial_features[:, :, 2] - pairwise_spatial_features[:, :, 0]
        h1 = pairwise_spatial_features[:, :, 3] - pairwise_spatial_features[:, :, 1]
        x2 = pairwise_spatial_features[:, :, 4]
        y2 = pairwise_spatial_features[:, :, 5]
        w2 = pairwise_spatial_features[:, :, 6] - pairwise_spatial_features[:, :, 4]
        h2 = pairwise_spatial_features[:, :, 7] - pairwise_spatial_features[:, :, 5]

        zeros = tf.zeros(tf.shape(pairwise_spatial_features)[0:2])

        x_overlap = compute_overlap(zeros, x1, w1, x2, w2,
                                    name_or_scope='x_overlap')
        y_overlap = compute_overlap(zeros, y1, h1, y2, h2,
                                    name_or_scope='y_overlap')

        intersection = compute_intersection(x_overlap, y_overlap)
        union = compute_union(w1, h1, w2, h2, intersection)
        # Assert ops cannot be executed on gpus.
        # we make sure that union > 0.
        # with tf.control_dependencies([tf.assert_positive(union)]):
        # Each kernel has to be of the shape
        # [n_hypotheses, n_hypotheses, feature].
        return tf.expand_dims(tf.div(intersection, union, name='iou'), 2)

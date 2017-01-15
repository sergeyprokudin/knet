"""Collection of different general purpose layers
"""

import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def prelu(data, name_or_scope=None):
    with tf.variable_scope(
            name_or_scope,
            default_name='prelu',
            values=[data]):
        alphas = tf.get_variable(shape=data.get_shape().as_list()[-1:],
                                 initializer=tf.constant_initializer(0.01),
                                 name="alphas")

        return tf.nn.relu(data) + tf.mul(alphas, (data - tf.abs(data))) * 0.5


def data_subselection_hard_negative_tf(targets, loss, name_or_scope=None):
    with tf.variable_scope(name_or_scope,
                           default_name='data_subselection_mask',
                           values=[targets, loss]):

        loss_reduced = tf.squeeze(tf.reduce_max(loss, [1]))
        targets_reduced = tf.squeeze(tf.reduce_max(targets, [1]))

        n_hypotheses = tf.shape(loss_reduced)[0]

        with tf.device('cpu'):
            _, ordering = tf.nn.top_k(loss_reduced, k=n_hypotheses)
            ordering = tf.to_int64(ordering)

            loss_reduced = tf.gather(loss_reduced, ordering)
            targets_reduced = tf.gather(targets_reduced, ordering)

            indices_positive = tf.squeeze(tf.where(
                tf.greater(targets_reduced, 0)))

            indices_backup = tf.squeeze(tf.where(
                tf.equal(targets_reduced, 0)))

            indices_negative = tf.squeeze(tf.where(
                tf.logical_and(tf.greater(loss_reduced, 0),
                               tf.equal(targets_reduced, 0))))

            rank_positive = tf.rank(indices_positive)
            rank_negative = tf.rank(indices_negative)

            def _empty_pos():
                # We return some index just any.
                return (tf.gather(ordering, tf.constant([0], dtype=tf.int64)),
                        tf.constant(1, dtype=tf.int32))

            def _pos():
                return (tf.gather(ordering, indices_positive),
                        tf.shape(indices_positive)[0])

            final_positive, size_positive = tf.cond(
                tf.equal(rank_positive, 0),
                _empty_pos,
                _pos)

            def _empty_neg():
                # We return just some index.
                return indices_backup

            def _neg():
                size_negative = tf.shape(indices_negative)[0]
                size_max = tf.minimum(
                    tf.maximum(
                        size_positive,
                        100),
                    size_negative)
                return tf.gather(ordering,
                                 indices_negative[:size_max])

            final_negative = tf.cond(tf.equal(rank_negative, 0),
                                     _empty_neg,
                                     _neg)

            return tf.concat(0, [final_positive, final_negative])

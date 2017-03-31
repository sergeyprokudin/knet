import tensorflow as tf


def mask_argmax_row(data, name_or_scope=None):
    with tf.variable_scope(name_or_scope,
                           default_name='mask_argmax_row',
                           values=[data]):
        n_data = tf.shape(data)[1]

        # Now we can pick the argmax for every column.
        data_argmax_row = tf.argmax(data, 1)

        # We create a one hot vector for every column.
        data_mask = tf.one_hot(data_argmax_row, depth=n_data)

        return tf.multiply(data, data_mask)


def construct_ground_truth_per_label_tf(dt_gt_iou,
                                        gt_labels,
                                        label,
                                        iou_threshold=0.5,
                                        name_or_scope=None):
    with tf.device('cpu'):
        with tf.variable_scope(
                name_or_scope,
                default_name='construct_ground_truth_per_label',
                values=[dt_gt_iou, gt_labels]):
            dt_gt_iou.get_shape().assert_has_rank(2)
            gt_labels.get_shape().assert_has_rank(1)

            # The gt_coords are in the format
            # [n_gt_hypotheses, (x, y, w, h, label)]
            # The dt_gt_iou are in the format
            # [n_hypotheses, (iou_gt_0, ... , iou_gt_n)]

            # The indices are on the labels which allow us to select the
            # right subselection of the dt_gt_iou.
            gt_hypotheses_indices = tf.where(tf.equal(gt_labels, label))

            # We transpose to subselect the gt_hypotheses
            dt_gt_iou_transpose = tf.transpose(dt_gt_iou)
            dt_gt_iou_transpose_subset = tf.squeeze(
                tf.gather(dt_gt_iou_transpose, gt_hypotheses_indices), [1])

            # We want to operate on the hypotheses again, so we transpose back.
            dt_gt_iou_subset = tf.transpose(dt_gt_iou_transpose_subset)

            n_gt_hypotheses = tf.shape(dt_gt_iou_subset)[1]

            def _empty():
                return dt_gt_iou_subset

            def _not_empty():
                zeros = tf.zeros_like(dt_gt_iou_subset)

                labels_ground_truth = tf.where(
                    dt_gt_iou_subset < iou_threshold,
                    zeros,
                    dt_gt_iou_subset)

                return mask_argmax_row(labels_ground_truth)

        return tf.cond(tf.equal(n_gt_hypotheses, 0), _empty, _not_empty)


def construct_independent_labels(dt_gt_iou,
                                 gt_labels,
                                 label,
                                 iou_threshold=0.5,
                                 name_or_scope=None):
    with tf.device('cpu'):
        with tf.variable_scope(
                name_or_scope,
                default_name='construct_ground_truth_per_label',
                values=[dt_gt_iou, gt_labels]):
            dt_gt_iou.get_shape().assert_has_rank(2)
            gt_labels.get_shape().assert_has_rank(1)

            # The gt_coords are in the format
            # [n_gt_hypotheses, (x, y, w, h, label)]
            # The dt_gt_iou are in the format
            # [n_hypotheses, (iou_gt_0, ... , iou_gt_n)]

            # The indices are on the labels which allow us to select the
            # right subselection of the dt_gt_iou.
            gt_hypotheses_indices = tf.where(tf.equal(gt_labels, label))

            # We transpose to subselect the gt_hypotheses
            dt_gt_iou_transpose = tf.transpose(dt_gt_iou)
            dt_gt_iou_transpose_subset = tf.squeeze(
                tf.gather(dt_gt_iou_transpose, gt_hypotheses_indices), [1])

            # We want to operate on the hypotheses again, so we transpose back.
            dt_gt_iou_subset = tf.transpose(dt_gt_iou_transpose_subset)

            dt_gt_shape = tf.shape(dt_gt_iou_subset)

            n_dt_hypotheses = dt_gt_shape[0]
            n_gt_hypotheses = dt_gt_shape[1]

            def _empty():
                return tf.zeros([n_dt_hypotheses])

            def _not_empty():
                return tf.to_float(tf.reduce_max(dt_gt_iou_subset, axis=1) > iou_threshold)

        return tf.cond(tf.equal(n_gt_hypotheses, 0), _empty, _not_empty)

def binarize(tensor, condition):
    return tf.where(
        condition,
        tf.ones_like(tensor),
        tf.zeros_like(tensor))


def compute_match_gt_perfect_per_label_tf(labels, matching_gt_mask,
                                          name_or_scope=None):
    """Return 1.0 for the best hypothesis and ground truth match, else 0.0.
    """
    with tf.variable_scope(name_or_scope,
                           default_name='compute_match_gt_perfect_per_label',
                           values=[labels]):

        n_gt_hypotheses = tf.shape(labels)[1]
        n_hypotheses = tf.shape(labels)[0]

        def _not_empty():
            labels_argmax = tf.argmax(labels, axis=0)
            labels_gt_matrix = tf.one_hot(labels_argmax, depth=n_hypotheses)
            labels_gt_matrix = tf.multiply(labels_gt_matrix, tf.expand_dims(matching_gt_mask,  axis=1))
            # This is a vector of the length of the hypotheses.
            return tf.reduce_sum(labels_gt_matrix, [0])

        def _empty():
            return tf.zeros((n_hypotheses,), dtype=tf.float32)

        return tf.cond(tf.equal(n_gt_hypotheses, 0), _empty, _not_empty)


def compute_match_gt_net_per_label_tf(predictions,
                                      labels,
                                      label,
                                      name_or_scope=None):
    """Return 1.0, best predicted hypothesis and ground truth match, else 0.0.

    If the network is already predicting something useful for a
    potential ground truth hypothesis we encourage to perform better for
    this hypothesis. If there is no positive prediction covering a
    ground truth hypothesis we encourage the best possible match under
    intersection over union.
    """
    with tf.variable_scope(name_or_scope,
                           default_name='compute_match_gt_net_per_label',
                           values=[predictions,
                                   labels]):

        n_gt_hypotheses = tf.shape(labels)[1]
        n_hypotheses = tf.shape(labels)[0]

        def _not_empty():
            labels_masked_bin = tf.to_float(tf.greater(labels, 0.0))
            # we need to remove columns that are not matching any hypothesis
            matching_gt_mask = tf.to_float(tf.not_equal(tf.reduce_sum(labels_masked_bin, 0), 0))
            labels_masked_ordered = tf.transpose(
                tf.multiply(tf.transpose(labels_masked_bin),
                       predictions[:, label]))
            return compute_match_gt_perfect_per_label_tf(labels_masked_ordered, matching_gt_mask)

        def _empty():
            return tf.zeros((n_hypotheses,), dtype=tf.float32)

        return tf.cond(tf.equal(n_gt_hypotheses, 0), _empty, _not_empty)

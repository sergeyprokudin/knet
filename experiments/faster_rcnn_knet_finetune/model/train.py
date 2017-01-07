"""Training routine for knet trained on top of FasterRCNN inference

    data_dir - directory containing train data :
                dt_coords.pkl - detections' bounding boxes coordinates in format [frame_id, x_min, y_min, width, height]
                dt_features.pkl - detections' bounding boxes features
                gt_coords.pkl - ground truth bounding boxes coordinates in format [frame_id, x_min, y_min, width, height, class_id]

"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import joblib
import multiprocessing
from functools import partial
import os
import sys
import shutil
import ntpath

import gflags
import logging
from google.apputils import app

from tools import bbox_utils, nms, metrics
from tf_layers import knet, spatial

gflags.DEFINE_string('data_dir', None, 'directory containing train data')
gflags.DEFINE_string(
    'log_dir',
    None,
    'directory to save logs and trained models')
gflags.DEFINE_integer(
    'data_loader_num_threads',
    5,
    'Number of threads used during data loading and preprocessing')
gflags.DEFINE_integer(
    'n_kernels',
    8,
    'Number of kernels to used in knet layer')
gflags.DEFINE_float(
    'best_iou_thres',
    0.5,
    'Number of threads used during data loading and preprocessing')
gflags.DEFINE_boolean(
    'logging_to_stdout',
    False,
    'Whether to write logs to stdout or to logfile')
gflags.DEFINE_integer('n_epochs', 100000, 'Number of training epochs')
gflags.DEFINE_integer('pos_weight', 1000, 'Weight of positive sample')

gflags.DEFINE_integer('knet_hlayer_size', 100, 'Size of knet hidden layers')

gflags.DEFINE_float('optimizer_step', 0.001, 'Learning step for optimizer')
gflags.DEFINE_boolean(
    'start_from_scratch',
    True,
    'Whether to load checkpoint (if it exists) or completely retrain the model')
gflags.DEFINE_boolean(
    'softmax_loss',
    False,
    'Whether to use softmax inference (this will make classes mutually exclusive)')
gflags.DEFINE_boolean(
    'softmax_kernel',
    True,
    'Whether to use softmax inference (this will make classes mutually exclusive)')
gflags.DEFINE_float('nms_thres', 0.8, 'NMS threshold')
gflags.DEFINE_integer(
    'n_eval_frames',
    100,
    'Number of frames to use for intermediate evaluation')
gflags.DEFINE_integer(
    'eval_step',
    5000,
    'Evaluate model after each eval_step')


FLAGS = gflags.FLAGS

DT_COORDS = 'dt_coords'
GT_COORDS = 'gt_coords'
GT_LABELS = 'gt_labels'
DT_LABELS = 'dt_labels'
DT_FEATURES = 'dt_features'
DT_INFERENCE = 'dt_inference'
DT_GT_IOU = 'dt_gt_iou'
DT_DT_IOU = 'dt_dt_iou'
N_DT_COORDS = 4
N_DT_FEATURES = 21
N_OBJECTS = 300
N_CLASSES = 21


def get_frame_data(fid, data):
    frame_data = {}
    fid_dt_ix = data[DT_COORDS][:, 0] == fid
    frame_data[DT_COORDS] = data[DT_COORDS][fid_dt_ix, 1:]
    frame_data[DT_FEATURES] = data[DT_FEATURES][fid_dt_ix]
    fid_gt_ix = data[GT_COORDS][:, 0] == fid
    frame_data[GT_COORDS] = data[GT_COORDS][fid_gt_ix, 1:5]
    frame_data[GT_LABELS] = data[GT_COORDS][fid_gt_ix, 5]
    frame_data[DT_GT_IOU] = bbox_utils.compute_sets_iou(
        frame_data[DT_COORDS], frame_data[GT_COORDS])
    #frame_data[DT_DT_IOU] = bbox_utils.compute_sets_iou(frame_data[DT_COORDS], frame_data[DT_COORDS])
    frame_data[DT_LABELS] = np.zeros([N_OBJECTS, N_CLASSES])
    for class_id in range(0, N_CLASSES):
        class_gt_boxes = frame_data[GT_LABELS] == class_id
        class_dt_gt = frame_data[DT_GT_IOU][:, class_gt_boxes]
        if (class_dt_gt.shape[1] != 0):
            frame_data[DT_LABELS][:, class_id] = np.max(
                bbox_utils.compute_best_iou(class_dt_gt), axis=1)
    return frame_data


def split_by_frames(data):
    unique_fids = np.unique(
        np.hstack([data[DT_COORDS][:, 0], data[GT_COORDS][:, 0]])).astype(int)
    pool = multiprocessing.Pool(FLAGS.data_loader_num_threads)
    get_frame_data_partial = partial(get_frame_data, data=data)
    frames_data_train = dict(
        zip(unique_fids, pool.map(get_frame_data_partial, unique_fids)))
    return frames_data_train


def preprocess_data(data_dir):
    data = {}
    data[DT_COORDS] = joblib.load(os.path.join(data_dir, 'dt_coords.pkl'))
    data[DT_FEATURES] = joblib.load(os.path.join(data_dir, 'dt_features.pkl'))
    data[GT_COORDS] = joblib.load(os.path.join(data_dir, 'gt_coords.pkl'))
    frames_data_train = split_by_frames(data)
    return frames_data_train


def load_data(data_dir):
    frames_data_cache_file = os.path.join(data_dir, 'frames_data.pkl')
    if (os.path.exists(frames_data_cache_file)):
        logging.info('loading frame bbox data info from cash..')
        frames_data = joblib.load(frames_data_cache_file)
    else:
        logging.info(
            'computing frame bbox data (IoU, labels, etc) - this could take some time..')
        frames_data = preprocess_data(data_dir)
        joblib.dump(frames_data, frames_data_cache_file)
    return frames_data


def set_logging(to_stdout=True, log_file=None):
    if (to_stdout):
        logging.basicConfig(
            format='%(asctime)s : %(message)s',
            level=logging.INFO,
            stream=sys.stdout)
    else:
        logging.basicConfig(
            format='%(asctime)s : %(message)s',
            level=logging.INFO,
            filename=log_file)
        print("logs could be found at %s" % log_file)
    return


def shuffle_samples(n_frames):
    return np.random.choice(n_frames, n_frames, replace=False)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def softmax(logits):
    n_classes = logits.shape[1]
    return np.exp(logits) / np.tile(np.sum(np.exp(logits),
                                           axis=1).reshape(-1, 1), [1, n_classes])


def print_debug_info(sess, input_ops, loss_op, knet_ops,
                     inference_op, frame_data, exp_log_dir):
    """ Print current values of kernel, inference, number of positive elements, etc.
    """
    feed_dict = {input_ops[DT_COORDS]: frame_data[DT_COORDS],
                 input_ops[DT_FEATURES]: frame_data[DT_FEATURES],
                 input_ops[DT_LABELS]: frame_data[DT_LABELS]}
    inference_orig = frame_data[DT_FEATURES]
    inference, loss, knet_data = sess.run(
        [inference_op, loss_op, knet_ops], feed_dict=feed_dict)
    logging.info("loss : %f" % loss)
    # #logging.info("initial scores for pos values : %s"%frame_data[DT_FEATURES][np.where(frame_data[DT_LABELS][0:N_OBJECTS]>0)])
    logging.info("non-neg kernel elements : %d" %
                 np.sum(knet_data['kernels'] > 0))
    logging.info("initial scores for matching bboxes : %s" %
                 inference_orig[np.where(frame_data[DT_LABELS] > 0)])
    logging.info("new knet scores for matching bboxes : %s" %
                 inference[np.where(frame_data[DT_LABELS] > 0)])
    num_gt = int(np.sum(frame_data[DT_LABELS]))
    num_pos_inf_orig = int(np.sum(inference_orig[:, 1:] > 0.0))
    num_pos_inf = int(np.sum(inference[:, 1:] > 0.5))
    logging.info(
        "frame num_gt : %d , num_pos_inf_orig : %d, num_pos_inf : %d" %
        (num_gt, num_pos_inf_orig, num_pos_inf))

    #import ipdb; ipdb.set_trace()

    import matplotlib; matplotlib.use('Agg')

    matplotlib.pyplot.imshow(softmax(inference_orig))
    matplotlib.pyplot.savefig(os.path.join(exp_log_dir, 'detections_orig.png'))
    matplotlib.pyplot.imshow(inference)
    matplotlib.pyplot.savefig(os.path.join(exp_log_dir, 'detections.png'))
    matplotlib.pyplot.imshow(knet_data['kernels'][0,:,:])
    matplotlib.pyplot.savefig(os.path.join(exp_log_dir, 'kernel.png'))

    return


def eval_model(sess, inference_op, input_ops, iou_op, frames_data,
               summary_writer, global_step, out_dir, full_eval=False, n_eval_frames=100):

    if (not os.path.exists(out_dir)):
        os.makedirs(out_dir)

    dt_gt_match_orig = []
    dt_gt_match_new = []
    dt_gt_match_orig_nms = []
    dt_gt_match_new_nms = []

    inference_orig_all = []
    inference_new_all = []
    coords = []
    gt_labels_all = []

    n_total_frames = len(frames_data.keys())

    if (full_eval):
        n_eval_frames = n_total_frames

    eval_data = {}

    # for fid in shuffle_samples(n_total_frames)[0:n_eval_frames]:
    for fid in range(0, n_eval_frames):

        eval_data[fid] = {}

        frame_data = frames_data[fid]

        gt_labels_all.append(frame_data[GT_LABELS].reshape(-1, 1))

        feed_dict = {input_ops[DT_COORDS]: frame_data[DT_COORDS][0:N_OBJECTS],
                     input_ops[DT_FEATURES]: frame_data[DT_FEATURES][0:N_OBJECTS],
                     input_ops[DT_LABELS]: frame_data[DT_LABELS][0:N_OBJECTS]}

        dt_scores = frame_data[DT_FEATURES]
        inference_orig = softmax(dt_scores)
        eval_data[fid]['dt_coords'] = frame_data[DT_COORDS]
        inference_orig_all.append(inference_orig)
        eval_data[fid]['inference_orig'] = inference_orig

        inference_new, dt_dt_iou = sess.run(
            [inference_op, iou_op], feed_dict=feed_dict)
        inference_new_all.append(inference_new)
        eval_data[fid]['inference_new'] = inference_new

        dt_gt_match_orig.append(
            metrics.match_dt_gt_all_classes(
                frame_data[DT_GT_IOU],
                frame_data[GT_LABELS],
                inference_orig))
        dt_gt_match_new.append(
            metrics.match_dt_gt_all_classes(
                frame_data[DT_GT_IOU],
                frame_data[GT_LABELS],
                inference_new))

        is_suppressed_orig = nms.nms_all_classes(
            dt_dt_iou, inference_orig, iou_thr=FLAGS.nms_thres)
        is_suppressed_new = nms.nms_all_classes(
            dt_dt_iou, inference_new, iou_thr=FLAGS.nms_thres)

        dt_gt_match_orig_nms.append(
            metrics.match_dt_gt_all_classes(
                frame_data[DT_GT_IOU],
                frame_data[GT_LABELS],
                inference_orig,
                dt_is_suppressed_info=is_suppressed_orig))
        dt_gt_match_new_nms.append(
            metrics.match_dt_gt_all_classes(
                frame_data[DT_GT_IOU],
                frame_data[GT_LABELS],
                inference_new,
                dt_is_suppressed_info=is_suppressed_new))

    gt_labels = np.vstack(gt_labels_all)
    inference_orig = np.vstack(inference_orig_all)
    inference_new = np.vstack(inference_new_all)

    dt_gt_match_orig = np.vstack(dt_gt_match_orig)
    dt_gt_match_new = np.vstack(dt_gt_match_new)
    dt_gt_match_orig_nms = np.vstack(dt_gt_match_orig_nms)
    dt_gt_match_new_nms = np.vstack(dt_gt_match_new_nms)

    if (full_eval):
        eval_data_file = os.path.join(
            out_dir, 'eval_data_step' + str(global_step) + '.pkl')
        joblib.dump(eval_data, eval_data_file)

    ap_orig, _ = metrics.average_precision_all_classes(
        dt_gt_match_orig, inference_orig, gt_labels)
    ap_orig_nms, _ = metrics.average_precision_all_classes(
        dt_gt_match_orig_nms, inference_orig, gt_labels)
    ap_new, _ = metrics.average_precision_all_classes(
        dt_gt_match_new, inference_new, gt_labels)
    ap_new_nms, _ = metrics.average_precision_all_classes(
        dt_gt_match_new_nms, inference_new, gt_labels)

    map_orig = np.nanmean(ap_orig)
    map_orig_nms = np.nanmean(ap_orig_nms)
    map_knet = np.nanmean(ap_new)
    map_knet_nms = np.nanmean(ap_new_nms)

    logging.info('mAP original inference : %f' % (map_orig))
    logging.info('mAP original inference (NMS) : %f' % (map_orig_nms))
    logging.info('mAP knet inference : %f' % (map_knet))
    logging.info('mAP knet inference (NMS) : %f' % (map_knet_nms))

    return map_knet

def write_scalar_summary(value, name, summary_writer, step_id):
    test_map_summ = tf.Summary(
        value=[
            tf.Summary.Value(
                tag=name,
                simple_value=value),
        ])
    summary_writer.add_summary(
        test_map_summ, global_step=step_id)
    return



def main(_):

    experiment_name = 'pw_' + str(FLAGS.pos_weight) + \
                     '_khls_' + str(FLAGS.knet_hlayer_size) + \
                     '_lr_' + str(FLAGS.optimizer_step) +\
                     '_sml_' + str(FLAGS.softmax_loss) +\
                     '_smk_' + str(FLAGS.softmax_kernel) +\
                     '_nk_' + str(FLAGS.n_kernels)

    exp_log_dir = os.path.join(FLAGS.log_dir, experiment_name)

    if (not os.path.exists(exp_log_dir)):
        os.makedirs(exp_log_dir)
    else:
        if (FLAGS.start_from_scratch):
            shutil.rmtree(exp_log_dir)
            os.makedirs(exp_log_dir)

    log_file = os.path.join(exp_log_dir, 'training.log')
    set_logging(FLAGS.logging_to_stdout, log_file)

    # loading data
    logging.info('loading data..')
    logging.info('train..')
    train_data_dir = os.path.join(FLAGS.data_dir, 'train')
    frames_data_train = load_data(train_data_dir)
    logging.info('test..')
    test_data_dir = os.path.join(FLAGS.data_dir, 'test')
    frames_data_test = load_data(test_data_dir)

    logging.info('defining the model..')
    n_frames = len(frames_data_train.keys())
    # model definition
    input_ops = {}
    input_ops[DT_COORDS] = tf.placeholder(
        tf.float32, shape=[
            N_OBJECTS, N_DT_COORDS], name=DT_COORDS)
    input_ops[DT_FEATURES] = tf.placeholder(
        tf.float32,
        shape=[
            N_OBJECTS,
            N_DT_FEATURES],
        name=DT_FEATURES)
    input_ops[DT_LABELS] = tf.placeholder(
        tf.float32, shape=[
            N_OBJECTS, N_CLASSES], name=DT_LABELS)

    pairwise_features_tf = spatial.construct_pairwise_features_tf(input_ops[
                                                                  DT_COORDS])
    iou_feature_tf = spatial.compute_pairwise_spatial_features_iou_tf(
        pairwise_features_tf)

    pairwise_scores_tf = spatial.construct_pairwise_features_tf(input_ops[
                                                                DT_FEATURES])
    spatial_features_tf = tf.concat(2, [iou_feature_tf, pairwise_scores_tf])
    n_spatial_features = N_DT_FEATURES * 2 + 1

    dt_new_features_tf, knet_ops_tf = knet.knet_layer(input_ops[DT_FEATURES], spatial_features_tf, n_kernels=FLAGS.n_kernels,
                                                      n_objects=N_OBJECTS, n_pair_features=n_spatial_features, n_object_features=N_DT_FEATURES,
                                                      softmax_kernel=FLAGS.softmax_kernel,
                                                      hlayer_size=FLAGS.knet_hlayer_size)

    W_fc1 = weight_variable([FLAGS.n_kernels * N_DT_FEATURES, FLAGS.knet_hlayer_size])
    b_fc1 = bias_variable([FLAGS.knet_hlayer_size])
    h_conv1 = tf.nn.relu(tf.matmul(dt_new_features_tf, W_fc1) + b_fc1)
    W_fc2 = weight_variable([FLAGS.knet_hlayer_size, N_CLASSES])
    b_fc2 = bias_variable([N_CLASSES])

    logits_tf = tf.matmul(h_conv1, W_fc2) + b_fc2

    if (FLAGS.softmax_loss):
        inference_tf = tf.nn.softmax(logits_tf)
        weights_tf = tf.add(tf.ones([N_OBJECTS, N_CLASSES]), tf.mul(input_ops[DT_LABELS], FLAGS.pos_weight-1))
        loss_tf = tf.nn.softmax_cross_entropy_with_logits(
            logits_tf, input_ops[DT_LABELS])
    else:
        inference_tf = tf.nn.sigmoid(logits_tf)
        loss_tf = tf.nn.weighted_cross_entropy_with_logits(
            logits_tf, input_ops[DT_LABELS], pos_weight=FLAGS.pos_weight)

    loss_final_tf = tf.reduce_mean(loss_tf)

    tf.summary.scalar('cross_entropy_loss', loss_final_tf)

    merged_summaries = tf.summary.merge_all()

    train_step = tf.train.AdamOptimizer(
        FLAGS.optimizer_step).minimize(loss_final_tf)

    with tf.Session() as sess:
        step_id = 0
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(
            max_to_keep=5,
            keep_checkpoint_every_n_hours=1.0)

        if (not FLAGS.start_from_scratch):
            ckpt_path = tf.train.latest_checkpoint(exp_log_dir)
            if (ckpt_path is not None):
                logging.info('model exists, restoring..')
                ckpt_name = ntpath.basename(ckpt_path)
                step_id = int(ckpt_name.split('-')[1])
                saver.restore(sess, ckpt_path)

        model_file = os.path.join(exp_log_dir, 'model')
        summary_writer = tf.summary.FileWriter(exp_log_dir, sess.graph)

        logging.info('training started..')

        for epoch_id in range(0, FLAGS.n_epochs):
            for fid in shuffle_samples(n_frames):
                frame_data = frames_data_train[fid]
                feed_dict = {input_ops[DT_COORDS]: frame_data[DT_COORDS][0:N_OBJECTS],
                             input_ops[DT_FEATURES]: frame_data[DT_FEATURES][0:N_OBJECTS],
                             input_ops[DT_LABELS]: frame_data[DT_LABELS][0:N_OBJECTS]}
                summary, _ = sess.run([merged_summaries, train_step],
                                      feed_dict=feed_dict)
                summary_writer.add_summary(summary, global_step=step_id)
                summary_writer.flush()
                step_id += 1
                if (step_id % 5000 == 0):
                    logging.info('current step : %d' % step_id)
                    print_debug_info(
                        sess,
                        input_ops,
                        loss_final_tf,
                        knet_ops_tf,
                        inference_tf,
                        frame_data,
                        exp_log_dir)
                    full_eval = False
                    if (step_id % 100000 == 0):
                        full_eval = True
                    # logging.info('evaluating on TRAIN..')
                    # train_out_dir = os.path.join(exp_log_dir, 'train')
                    # train_map = eval_model(sess, inference_tf, input_ops, iou_feature_tf,
                    #                        frames_data_train, summary_writer,
                    #                        global_step=step_id, n_eval_frames=FLAGS.n_eval_frames,
                    #                        out_dir=train_out_dir,
                    #                        full_eval=full_eval)
                    # if (full_eval) :
                    #     write_scalar_summary(train_map, 'train_map_full', summary_writer, step_id)
                    # else :
                    #     write_scalar_summary(train_map, 'train_map', summary_writer, step_id)
                    logging.info('evaluating on TEST..')
                    test_out_dir = os.path.join(exp_log_dir, 'test')
                    test_map = eval_model(sess, inference_tf, input_ops, iou_feature_tf,
                                          frames_data_test, summary_writer,
                                          global_step=step_id, n_eval_frames=FLAGS.n_eval_frames,
                                          out_dir=test_out_dir,
                                          full_eval=full_eval)
                    if (full_eval) :
                        write_scalar_summary(test_map, 'test_map_full', summary_writer, step_id)
                    else :
                        write_scalar_summary(test_map, 'test_map', summary_writer, step_id)
                    saver.save(sess, model_file, global_step=step_id)
    return

if __name__ == '__main__':
    gflags.mark_flag_as_required('data_dir')
    gflags.mark_flag_as_required('log_dir')
    app.run()

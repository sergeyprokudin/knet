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
import model as nnms
import eval

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
gflags.DEFINE_float('nms_thres', 0.5, 'NMS threshold')
gflags.DEFINE_integer(
    'n_eval_frames',
    1000,
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
DT_LABELS_BASIC = 'dt_labels_basic'
DT_FEATURES = 'dt_features'
DT_INFERENCE = 'dt_inference'
DT_GT_IOU = 'dt_gt_iou'
DT_DT_IOU = 'dt_dt_iou'
N_DT_COORDS = 4
N_DT_FEATURES = 21
N_OBJECTS = 20
N_CLASSES = 21


def get_frame_data(fid, data):
    frame_data = {}
    fid_dt_ix = data[DT_COORDS][:, 0] == fid
    frame_data[DT_COORDS] = data[DT_COORDS][fid_dt_ix, 1:][0:N_OBJECTS]
    frame_data[DT_FEATURES] = data[DT_FEATURES][fid_dt_ix][0:N_OBJECTS]
    fid_gt_ix = data[GT_COORDS][:, 0] == fid
    frame_data[GT_COORDS] = data[GT_COORDS][fid_gt_ix, 1:5]
    frame_data[GT_LABELS] = data[GT_COORDS][fid_gt_ix, 5]
    frame_data[DT_GT_IOU] = bbox_utils.compute_sets_iou(
        frame_data[DT_COORDS], frame_data[GT_COORDS])
    # frame_data[DT_DT_IOU] = bbox_utils.compute_sets_iou(frame_data[DT_COORDS], frame_data[DT_COORDS])
    frame_data[DT_LABELS] = np.zeros([N_OBJECTS, N_CLASSES])
    frame_data[DT_LABELS_BASIC] = np.zeros([N_OBJECTS, N_CLASSES])
    for class_id in range(0, N_CLASSES):
        class_gt_boxes = frame_data[GT_LABELS] == class_id
        class_dt_gt = frame_data[DT_GT_IOU][:, class_gt_boxes]
        if (class_dt_gt.shape[1] != 0):
            frame_data[DT_LABELS][:, class_id] = np.max(
                bbox_utils.compute_best_iou(class_dt_gt), axis=1)
            frame_data[DT_LABELS_BASIC][:, class_id][
                np.max(class_dt_gt, axis=1) > 0.5] = 1
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
    if os.path.exists(frames_data_cache_file):
        logging.info('loading frame bbox data info from cash..')
        frames_data = joblib.load(frames_data_cache_file)
    else:
        logging.info(
            'computing frame bbox data (IoU, labels, etc) - this could take some time..')
        frames_data = preprocess_data(data_dir)
        joblib.dump(frames_data, frames_data_cache_file)
    return frames_data


def set_logging(to_stdout=True, log_file=None):
    if to_stdout:
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

    if not os.path.exists(exp_log_dir):
        os.makedirs(exp_log_dir)
    else:
        if FLAGS.start_from_scratch:
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

    nnms_model = nnms.NeuralNMS(n_detections=N_OBJECTS,
                                n_dt_features=N_DT_FEATURES,
                                n_classes=N_CLASSES,
                                n_kernels=FLAGS.n_kernels,
                                pos_weight=FLAGS.pos_weight,
                                knet_hlayer_size=FLAGS.knet_hlayer_size,
                                optimizer_step=FLAGS.optimizer_step)

    merged_summaries = tf.merge_all_summaries()

    with tf.Session() as sess:
        step_id = 0
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(
            max_to_keep=5,
            keep_checkpoint_every_n_hours=1.0)

        if not FLAGS.start_from_scratch:
            ckpt_path = tf.train.latest_checkpoint(exp_log_dir)
            if ckpt_path is not None:
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
                feed_dict = {nnms_model.dt_coords: frame_data[DT_COORDS],
                             nnms_model.dt_features: frame_data[DT_FEATURES],
                             nnms_model.dt_labels: frame_data[DT_LABELS]}
                summary, _ = sess.run([merged_summaries, nnms_model.train_step],
                                      feed_dict=feed_dict)
                summary_writer.add_summary(summary, global_step=step_id)
                summary_writer.flush()

                step_id += 1
                if step_id % FLAGS.eval_step == 0:
                    eval.print_debug_info(sess=sess,
                                          nnms_model=nnms_model,
                                          frame_data=frame_data,
                                          outdir=exp_log_dir,
                                          fid=fid)
                    full_eval = False
                    if step_id % 100000 == 0:
                        full_eval = True
                    logging.info('evaluating on TEST..')
                    test_out_dir = os.path.join(exp_log_dir, 'test')
                    test_map = eval.eval_model(sess, nnms_model,
                                               frames_data_test,
                                               global_step=step_id, n_eval_frames=1000,
                                               out_dir=test_out_dir,
                                               full_eval=full_eval,
                                               nms_thres=FLAGS.nms_thres)
                    if full_eval:
                        write_scalar_summary(
                            test_map, 'test_map_full', summary_writer, step_id)
                    else:
                        write_scalar_summary(
                            test_map, 'test_map', summary_writer, step_id)
                    saver.save(sess, model_file, global_step=step_id)
    return

if __name__ == '__main__':
    gflags.mark_flag_as_required('data_dir')
    gflags.mark_flag_as_required('log_dir')
    app.run()

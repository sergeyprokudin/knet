"""Training routine for knet trained on top of FasterRCNN inference

    data_dir - directory containing train data :
                dt_coords.pkl - detections' bounding boxes coordinates in format [frame_id, x_min, y_min, width, height]
                dt_features.pkl - detections' bounding boxes features
                gt_coords.pkl - ground truth bounding boxes coordinates in format [frame_id, x_min, y_min, width, height, class_id]

"""

import binascii
import logging
import shutil
import subprocess
import sys
from functools import partial
from timeit import default_timer as timer

import eval
import gflags
import joblib
import ntpath
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import yaml
from google.apputils import app
from nms_network import model as nms_net
from tools import bbox_utils
from tools import experiment_config as expconf

gflags.DEFINE_string('data_dir', None, 'directory containing train data')
gflags.DEFINE_string('log_dir', None, 'directory to save logs and trained models')
gflags.DEFINE_string('config_path', None, 'config with main model params')

FLAGS = gflags.FLAGS

N_DT_COORDS = 4
N_FC_FEATURES_FULL = 4096
N_FC_FEATURES_SHORT = 100
N_CLASS_SCORES = 21
N_DT_FEATURES_FULL = N_CLASS_SCORES + N_FC_FEATURES_FULL
N_DT_FEATURES_SHORT = N_CLASS_SCORES + N_FC_FEATURES_SHORT
N_OBJECTS = 20
N_CLASSES = 21


def get_frame_data(fid, data, n_bboxes):
    frame_data = {}
    fid_dt_ix = data[nms_net.DT_COORDS][:, 0] == fid
    frame_data[nms_net.DT_COORDS] = data[nms_net.DT_COORDS][fid_dt_ix, 1:][0:n_bboxes]
    frame_data[nms_net.DT_FEATURES] = data[nms_net.DT_FEATURES][fid_dt_ix][0:n_bboxes]
    frame_data[nms_net.DT_SCORES] = data[nms_net.DT_SCORES][fid_dt_ix][0:n_bboxes]
    fid_gt_ix = data[nms_net.GT_COORDS][:, 0] == fid
    frame_data[nms_net.GT_COORDS] = data[nms_net.GT_COORDS][fid_gt_ix, 1:5]
    frame_data[nms_net.GT_LABELS] = data[nms_net.GT_COORDS][fid_gt_ix, 5]
    frame_data[nms_net.DT_GT_IOU] = bbox_utils.compute_sets_iou(
        frame_data[nms_net.DT_COORDS], frame_data[nms_net.GT_COORDS])
    # frame_data[nnms.DT_DT_IOU] = bbox_utils.compute_sets_iou(frame_data[nnms.DT_COORDS], frame_data[nnms.DT_COORDS])
    frame_data[nms_net.DT_LABELS] = np.zeros([n_bboxes, N_CLASSES])
    frame_data[nms_net.DT_LABELS_BASIC] = np.zeros([n_bboxes, N_CLASSES])
    for class_id in range(0, N_CLASSES):
        class_gt_boxes = frame_data[nms_net.GT_LABELS] == class_id
        class_dt_gt = frame_data[nms_net.DT_GT_IOU][:, class_gt_boxes]
        if class_dt_gt.shape[1] != 0:
            frame_data[nms_net.DT_LABELS][:, class_id] = np.max(
                bbox_utils.compute_best_iou(class_dt_gt), axis=1)
            frame_data[nms_net.DT_LABELS_BASIC][:, class_id][
                np.max(class_dt_gt, axis=1) > 0.5] = 1
    # logging.info('finished processing frame %d' % fid)
    return frame_data


def split_by_frames(data, n_bboxes):
    unique_fids = np.unique(np.hstack([data[nms_net.DT_COORDS][:, 0], data[nms_net.GT_COORDS][:, 0]])).astype(int)
    get_frame_data_partial = partial(get_frame_data, data=data, n_bboxes=n_bboxes)
    frames_data_train = dict(
        zip(unique_fids, map(get_frame_data_partial, unique_fids)))
    return frames_data_train


def preprocess_data(data_dir, n_bboxes, use_short_features=False):
    if use_short_features:
        dt_features_path = os.path.join(data_dir, 'dt_features_short.pkl')
    else :
        dt_features_path = os.path.join(data_dir, 'dt_features_full.pkl')
    data = {}
    data[nms_net.DT_COORDS] = joblib.load(os.path.join(data_dir, 'dt_coords.pkl'))
    data[nms_net.DT_SCORES] = joblib.load(os.path.join(data_dir, 'dt_scores.pkl'))
    data[nms_net.DT_FEATURES] = joblib.load(os.path.join(data_dir, dt_features_path))
    data[nms_net.GT_COORDS] = joblib.load(os.path.join(data_dir, 'gt_coords.pkl'))
    logging.info('finished loading data')
    frames_data_train = split_by_frames(data, n_bboxes)
    return frames_data_train


def load_data(data_dir, n_bboxes, use_short_features=False):
    if use_short_features:
        frames_data_cache_file = os.path.join(data_dir, 'frames_data_short_' + str(n_bboxes) + '.pkl')
    else:
        frames_data_cache_file = os.path.join(data_dir, 'frames_data_full_' + str(n_bboxes) + '.pkl')
    if os.path.exists(frames_data_cache_file):
        logging.info('loading frame bbox data info from cash..')
        frames_data = joblib.load(frames_data_cache_file)
    else:
        logging.info(
            'computing frame bbox data (IoU, labels, etc) - this could take some time..')
        frames_data = preprocess_data(data_dir, n_bboxes, use_short_features=use_short_features)
        joblib.dump(frames_data, frames_data_cache_file)
    return frames_data


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


def input_ops(n_detections,
              n_dt_features):

    input_dict = {}
    n_dt_coords = 4
    input_dict['dt_coords'] = tf.placeholder(
        tf.float32, shape=[
                None, n_dt_coords])

    input_dict['dt_features'] = tf.placeholder(tf.float32,
                                 shape=[
                                     None,
                                     n_dt_features])

    input_dict['dt_probs'] = tf.placeholder(tf.float32,
                                 shape=[
                                     None,
                                     21])

    input_dict['gt_coords'] = tf.placeholder(tf.float32, shape=[None, 4])

    input_dict['gt_labels'] = tf.placeholder(tf.float32, shape=None)

    input_dict['keep_prob'] = tf.placeholder(tf.float32)

    return input_dict


def shuffle_train_test(frames_data_train, frames_data_test):
    all_frames = frames_data_train.copy()
    n_frames_train = len(frames_data_train)
    for fid, data in frames_data_test.iteritems():
        all_frames[n_frames_train+fid] = data
    n_frames_all = len(all_frames)
    shuffled_fids = shuffle_samples(n_frames_all)
    # half = n_frames_all / 2
    train_fids = shuffled_fids[0:9000]
    test_fids = shuffled_fids[9000:]
    train = {}
    test = {}
    for i, fid in enumerate(train_fids):
        train[i] = all_frames[fid]
    for i, fid in enumerate(test_fids):
        test[i] = all_frames[fid]
    return train, test


def main(_):

    config = expconf.ExperimentConfig(data_dir=FLAGS.data_dir,
                                      root_log_dir=FLAGS.log_dir,
                                      config_path=FLAGS.config_path)

    config.save_results()

    logging.info('loading data..')
    logging.info('train..')
    frames_data_train = load_data(config.train_data_dir,
                                  n_bboxes=config.n_bboxes,
                                  use_short_features=config.use_reduced_fc_features)
    logging.info('test..')
    frames_data_test = load_data(config.test_data_dir,
                                 n_bboxes=config.n_bboxes,
                                 use_short_features=config.use_reduced_fc_features)

    if config.shuffle_train_test:
        frames_data_train, frames_data_test = shuffle_train_test(frames_data_train, frames_data_test)

    n_frames_train = len(frames_data_train.keys())
    n_frames_test = len(frames_data_test.keys())

    logging.info('building model graph..')

    in_ops = input_ops(config.n_bboxes, config.n_dt_features)

    nnms_model = nms_net.NMSNetwork(n_classes=N_CLASSES,
                                    input_ops=in_ops,
                                    loss_type='nms_loss',
                                    **config.nms_network_config)

    with tf.Session() as sess:
        step_id = 0

        sess.run(nnms_model.init_op)

        saver = tf.train.Saver(
            max_to_keep=5,
            keep_checkpoint_every_n_hours=1.0)

        if not config.start_from_scratch:
            ckpt_path = tf.train.latest_checkpoint(config.log_dir)
            if ckpt_path is not None:
                logging.info('model exists, restoring..')
                ckpt_name = ntpath.basename(ckpt_path)
                step_id = int(ckpt_name.split('-')[1])
                saver.restore(sess, ckpt_path)

        summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph)

        logging.info('training started..')
        for epoch_id in range(0, config.n_epochs):

            step_times = []

            for fid in shuffle_samples(n_frames_train):
                frame_data = frames_data_train[fid]
                feed_dict = {nnms_model.dt_coords: frame_data[nms_net.DT_COORDS],
                             nnms_model.dt_features: frame_data[nms_net.DT_FEATURES],
                             nnms_model.dt_probs: frame_data[nms_net.DT_FEATURES][:, 0:21],
                             nnms_model.gt_coords: frame_data[nms_net.GT_COORDS],
                             nnms_model.gt_labels: frame_data[nms_net.GT_LABELS],
                             nnms_model.keep_prob: config.keep_prob_train,
                             nnms_model.learning_rate: 0.0001}

                start_step = timer()

                summary, _ = sess.run([nnms_model.merged_summaries, nnms_model.train_step],
                                      feed_dict=feed_dict)
                end_step = timer()

                step_times.append(end_step-start_step)

                summary_writer.add_summary(summary, global_step=step_id)
                summary_writer.flush()

                step_id += 1

            if step_id % config.eval_step == 0:

                logging.info('step : %d, time : %s' % (step_id, str(np.mean(step_times))))

                fid = shuffle_samples(n_frames_test)[0]

                frame_data = frames_data_test[fid]

                eval.print_debug_info(sess=sess,
                                      nnms_model=nnms_model,
                                      frame_data=frame_data,
                                      outdir=config.log_dir,
                                      fid=fid)

                logging.info('evaluating on TRAIN..')
                train_out_dir = os.path.join(config.log_dir, 'train')
                train_map, train_map_nms = eval.eval_model(sess, nnms_model,
                                            frames_data_train,
                                            global_step=step_id,
                                            n_eval_frames=config.n_eval_frames,
                                            out_dir=train_out_dir,
                                            full_eval=config.full_eval,
                                            nms_thres=config.nms_thres)

                write_scalar_summary(train_map, 'train_map', summary_writer, step_id)

                logging.info('evaluating on TEST..')
                test_out_dir = os.path.join(config.log_dir, 'test')
                test_map, test_map_nms = eval.eval_model(sess, nnms_model,
                                           frames_data_test,
                                           global_step=step_id,
                                           n_eval_frames=config.n_eval_frames,
                                           out_dir=test_out_dir,
                                           full_eval=config.full_eval,
                                           nms_thres=config.nms_thres)

                write_scalar_summary(test_map, 'test_map', summary_writer, step_id)

                config.update_results(step_id,
                                      train_map,
                                      train_map_nms,
                                      test_map,
                                      test_map_nms,
                                      np.mean(step_times))

                config.save_results()

                saver.save(sess, config.model_file, global_step=step_id)
    return

if __name__ == '__main__':
    gflags.mark_flag_as_required('data_dir')
    gflags.mark_flag_as_required('log_dir')
    gflags.mark_flag_as_required('config_path')
    app.run()

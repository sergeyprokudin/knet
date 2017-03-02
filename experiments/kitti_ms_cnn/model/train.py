"""Training routine for knet trained on top of MS-CNN inference on KITTI object detection data

"""

import binascii
import logging
import shutil
import subprocess
import sys
from functools import partial
from timeit import default_timer as timer

import gflags
import ntpath
import numpy as np
import os
import tensorflow as tf
from google.apputils import app
from nms_network import model as nms_net
import eval
from data import get_frame_data
from tools import experiment_config as expconf

gflags.DEFINE_string('data_dir', None, 'directory containing train data')
gflags.DEFINE_string('root_log_dir', None, 'root directory to save logs')
gflags.DEFINE_string('config_path', None, 'path to experiment config')

FLAGS = gflags.FLAGS

CAR_CLASSES = {'Car': 0, 'Van': 1, 'Truck': 2, 'Tram': 3}


def shuffle_samples(n_frames):
    return np.random.choice(n_frames, n_frames, replace=False)


def input_ops(n_dt_features):

    input_dict = {}
    n_dt_coords = 4
    input_dict['dt_coords'] = tf.placeholder(
        tf.float32, shape=[
                None, n_dt_coords])

    input_dict['dt_features'] = tf.placeholder(tf.float32,
                                 shape=[
                                     None,
                                     n_dt_features])

    input_dict['gt_coords'] = tf.placeholder(tf.float32, shape=[None, 4])

    input_dict['gt_labels'] = tf.placeholder(tf.float32, shape=None)

    input_dict['keep_prob'] = tf.placeholder(tf.float32)

    return input_dict


def main(_):


    config = expconf.ExperimentConfig(data_dir=FLAGS.data_dir,
                                      root_log_dir=FLAGS.root_log_dir,
                                      config_path=FLAGS.config_path)

    labels_dir = os.path.join(FLAGS.data_dir, 'label_2')
    detections_dir = os.path.join(FLAGS.data_dir, 'detection_2')

    frames_ids = np.asarray([int(ntpath.basename(path).split('.')[0]) for path in os.listdir(labels_dir)])

    n_frames = len(frames_ids)

    half = n_frames/2

    shuffled_samples = shuffle_samples(n_frames)
    train_frames = frames_ids[shuffled_samples[0:half]]
    n_train_samples = len(train_frames)
    test_frames = frames_ids[shuffled_samples[half:]]
    n_test_samples = len(test_frames)

    logging.info('building model graph..')

    in_ops = input_ops(config.n_dt_features)

    nnms_model = nms_net.NMSNetwork(n_classes=1,
                                    input_ops=in_ops,
                                    **config.nms_network_config)

    logging.info('training started..')

    with tf.Session() as sess:

        sess.run(nnms_model.init_op)

        step_id = 0
        step_times = []

        for epoch_id in range(0, config.n_epochs):

            for fid in train_frames:

                start_step = timer()

                frame_data = get_frame_data(frame_id=fid,
                                            labels_dir=labels_dir,
                                            detections_dir=detections_dir,
                                            n_detections=config.n_bboxes,
                                            n_features=config.n_dt_features)

                feed_dict = {nnms_model.dt_coords: frame_data['dt_coords'],
                             nnms_model.dt_features: frame_data['dt_features'],
                             nnms_model.gt_coords: frame_data['gt_coords'],
                             nnms_model.gt_labels: frame_data['gt_labels'],
                             nnms_model.keep_prob: 1.0}

                _ = sess.run([nnms_model.train_step],
                             feed_dict=feed_dict)

                step_id += 1

                end_step = timer()
                step_times.append(end_step-start_step)

                if step_id % 1000 == 0:
                    logging.info('curr step : %d, mean time for step : %s' % (step_id, str(np.mean(step_times))))
                    step_times = []

                    train_map, test_map_nms = eval.eval_model(sess,
                                                              nnms_model,
                                                              detections_dir=detections_dir,
                                                              labels_dir=labels_dir,
                                                              eval_frames=train_frames[0:500],
                                                              n_bboxes=config.n_bboxes,
                                                              n_features=config.n_dt_features,
                                                              nms_thres=0.5)

                    test_map, test_map_nms = eval.eval_model(sess,
                                                             nnms_model,
                                                             detections_dir=detections_dir,
                                                             labels_dir=labels_dir,
                                                             eval_frames=test_frames[0:500],
                                                             n_bboxes=config.n_bboxes,
                                                             n_features=config.n_dt_features,
                                                             nms_thres=0.5)

    return

if __name__ == '__main__':
    gflags.mark_flag_as_required('data_dir')
    gflags.mark_flag_as_required('root_log_dir')
    gflags.mark_flag_as_required('config_path')
    app.run()

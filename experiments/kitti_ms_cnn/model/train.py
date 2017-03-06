"""Training routine for knet trained on top of MS-CNN inference on KITTI object detection data

"""

import logging
from timeit import default_timer as timer

import gflags
import ntpath
import numpy as np
import os
import tensorflow as tf
from google.apputils import app
from nms_network import model as nms_net
import eval_supp
from data import get_frame_data_fixed
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

    input_dict['nms_labels'] = tf.placeholder(tf.float32, shape=None)

    input_dict['keep_prob'] = tf.placeholder(tf.float32)

    return input_dict


def main(_):

    config = expconf.ExperimentConfig(data_dir=FLAGS.data_dir,
                                      root_log_dir=FLAGS.root_log_dir,
                                      config_path=FLAGS.config_path)

    logging.info("config info : %s" % config.config)

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

    in_ops = input_ops(config.n_dt_features+1)

    nnms_model = nms_net.NMSNetwork(n_classes=1,
                                    n_bboxes=config.n_bboxes,
                                    input_ops=in_ops,
                                    **config.nms_network_config)

    logging.info('training started..')

    with tf.Session() as sess:

        sess.run(nnms_model.init_op)

        step_id = 0
        step_times = []

        for epoch_id in range(0, 10):

            for fid in train_frames:

                start_step = timer()

                frame_data = get_frame_data_fixed(frame_id=fid,
                                            labels_dir=labels_dir,
                                            detections_dir=detections_dir,
                                            n_detections=config.n_bboxes,
                                            n_features=config.n_dt_features)

                feed_dict = {nnms_model.dt_coords: frame_data['dt_coords'],
                             nnms_model.dt_features: frame_data['dt_features'],
                             nnms_model.gt_coords: frame_data['gt_coords'],
                             nnms_model.gt_labels: frame_data['gt_labels'],
                             # nnms_model.nms_labels: frame_data['nms_labels'],
                             nnms_model.keep_prob: config.keep_prob_train}

                _ = sess.run([nnms_model.train_step],
                                                       feed_dict=feed_dict)

                # import ipdb; ipdb.set_trace()

                step_id += 1

                end_step = timer()
                step_times.append(end_step-start_step)

                if step_id % 1000 == 0:
                    logging.info('curr step : %d, mean time for step : %s' % (step_id, str(np.mean(step_times))))

                    train_losses = []
                    for tfid in train_frames[0:100]:

                        frame_data = get_frame_data_fixed(frame_id=tfid,
                                                    labels_dir=labels_dir,
                                                    detections_dir=detections_dir,
                                                    n_detections=config.n_bboxes,
                                                    n_features=config.n_dt_features)

                        feed_dict = {nnms_model.dt_coords: frame_data['dt_coords'],
                                     nnms_model.dt_features: frame_data['dt_features'],
                                     nnms_model.gt_coords: frame_data['gt_coords'],
                                     nnms_model.gt_labels: frame_data['gt_labels'],
                                     # nnms_model.nms_labels: frame_data['nms_labels'],
                                     nnms_model.keep_prob: 1.0}

                        train_losses.append(sess.run(nnms_model.loss, feed_dict=feed_dict))

                    logging.info("train loss : %s" % str(np.mean(train_losses)))

                    test_losses = []
                    for tfid in test_frames[0:100]:

                        frame_data = get_frame_data_fixed(frame_id=tfid,
                                                    labels_dir=labels_dir,
                                                    detections_dir=detections_dir,
                                                    n_detections=config.n_bboxes,
                                                    n_features=config.n_dt_features)

                        feed_dict = {nnms_model.dt_coords: frame_data['dt_coords'],
                                     nnms_model.dt_features: frame_data['dt_features'],
                                     nnms_model.gt_coords: frame_data['gt_coords'],
                                     nnms_model.gt_labels: frame_data['gt_labels'],
                                     # nnms_model.nms_labels: frame_data['nms_labels'],
                                     nnms_model.keep_prob: 1.0}

                        test_losses.append(sess.run(nnms_model.loss, feed_dict=feed_dict))

                    logging.info("test loss : %s" % str(np.mean(test_losses)))

                    fid = test_frames[np.random.randint(0,n_test_samples,1)[0]]
                    frame_data = get_frame_data_fixed(frame_id=fid,
                            labels_dir=labels_dir,
                            detections_dir=detections_dir,
                            n_detections=config.n_bboxes,
                            n_features=config.n_dt_features)

                    feed_dict = {nnms_model.dt_coords: frame_data['dt_coords'],
                                     nnms_model.dt_features: frame_data['dt_features'],
                                     nnms_model.gt_coords: frame_data['gt_coords'],
                                     nnms_model.gt_labels: frame_data['gt_labels'],
                                     # nnms_model.nms_labels: frame_data['nms_labels'],
                                     nnms_model.keep_prob: 1.0}

                    # true_labels = frame_data['nms_labels']

                    dt_coords, dt_features, spatial_features, nmsnet_inference, loss, iou_tf, true_labels \
                        = sess.run([nnms_model.dt_coords,
                                    nnms_model.dt_features,
                                    nnms_model.spatial_features,
                                    nnms_model.class_scores,
                                    nnms_model.loss,
                                    nnms_model.iou_feature,
                                    nnms_model.labels],
                                   feed_dict=feed_dict)

                    pos_inference = nmsnet_inference[np.where(true_labels == 1)]
                    neg_inference = nmsnet_inference[np.where(true_labels == 0)]

                    # iou_np = frame_data['dt_dt_iou']
                    # iou_tf = iou_tf.reshape([config.n_bboxes, config.n_bboxes])

                    logging.info("positive inference samples : %s" % str(pos_inference))
                    logging.info("negative inference samples : %s" % str(neg_inference))

                    train_map, test_map_nms = eval_supp.eval_model(sess,
                                                              nnms_model,
                                                              detections_dir=detections_dir,
                                                              labels_dir=labels_dir,
                                                              eval_frames=train_frames[0:100],
                                                              n_bboxes=config.n_bboxes,
                                                              n_features=config.n_dt_features,
                                                              nms_thres=0.5)

                    test_map, test_map_nms = eval_supp.eval_model(sess,
                                                             nnms_model,
                                                             detections_dir=detections_dir,
                                                             labels_dir=labels_dir,
                                                             eval_frames=test_frames[0:100],
                                                             n_bboxes=config.n_bboxes,
                                                             n_features=config.n_dt_features,
                                                             nms_thres=0.5)
                    # import ipdb; ipdb.set_trace()
    return

if __name__ == '__main__':
    gflags.mark_flag_as_required('data_dir')
    gflags.mark_flag_as_required('root_log_dir')
    gflags.mark_flag_as_required('config_path')
    app.run()

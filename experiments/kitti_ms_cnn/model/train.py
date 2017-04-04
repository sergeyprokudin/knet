"""Training routine for knet trained on top of MS-CNN inference on KITTI object detection data

"""

import logging
from timeit import default_timer as timer

import gflags
import ntpath
import numpy as np
from numpy.random import RandomState

import os
import tensorflow as tf
from google.apputils import app
from nms_network import model as nms_net
import eval_supp
from data import get_frame_data, get_frame_data_fixed
from tools import experiment_config as expconf


gflags.DEFINE_string('data_dir', None, 'directory containing train data')
gflags.DEFINE_string('root_log_dir', None, 'root directory to save logs')
gflags.DEFINE_string('config_path', None, 'path to experiment config')

FLAGS = gflags.FLAGS

CAR_CLASSES = {'Car': 0, 'Van': 1, 'Truck': 2, 'Tram': 3}


def shuffle_samples(n_frames):
    return np.random.choice(n_frames, n_frames, replace=False)


def input_ops(n_dt_features, n_classes):

    input_dict = {}
    n_dt_coords = 4
    input_dict['dt_coords'] = tf.placeholder(
        tf.float32, shape=[
                None, n_dt_coords])

    input_dict['dt_features'] = tf.placeholder(tf.float32,
                                 shape=[
                                     None,
                                     n_classes+n_dt_features])

    input_dict['dt_probs'] = tf.placeholder(tf.float32,
                                 shape=[
                                     None,
                                     n_classes])

    input_dict['gt_coords'] = tf.placeholder(tf.float32, shape=[None, 4])

    input_dict['gt_labels'] = tf.placeholder(tf.float32, shape=None)

    input_dict['nms_labels'] = tf.placeholder(tf.float32, shape=None)

    input_dict['keep_prob'] = tf.placeholder(tf.float32)

    return input_dict


def mean_loss(sess, nms_model, frames,
              labels_dir, detection_dir,
              n_bboxes_test, n_dt_features,
              n_frames=100):

    losses = []

    for tfid in frames[0:n_frames]:

        frame_data = get_frame_data_fixed(frame_id=tfid,
                                    labels_dir=labels_dir,
                                    detections_dir=detection_dir,
                                    n_detections=n_bboxes_test,
                                    n_features=n_dt_features)

        feed_dict = {nms_model.dt_coords: frame_data['dt_coords'],
                     nms_model.dt_features: frame_data['dt_features'],
                     nms_model.dt_probs_ini: frame_data['dt_probs'],
                     nms_model.gt_coords: frame_data['gt_coords'],
                     nms_model.gt_labels: frame_data['gt_labels'],
                     # nnms_model.nms_labels: frame_data['nms_labels'],
                     nms_model.keep_prob: 1.0}

        det_loss = sess.run([nms_model.det_loss], feed_dict=feed_dict)

        losses.append(det_loss)

    return np.mean(losses)


def main(_):

    config = expconf.ExperimentConfig(data_dir=FLAGS.data_dir,
                                      root_log_dir=FLAGS.root_log_dir,
                                      config_path=FLAGS.config_path)



    logging.info("config info : %s" % config.config)

    labels_dir = os.path.join(FLAGS.data_dir, 'label_2')

    detections_dir = os.path.join(FLAGS.data_dir, 'detection_2')

    frames_ids = np.asarray([int(ntpath.basename(path).split('.')[0]) for path in os.listdir(labels_dir)])

    n_frames = len(frames_ids)
    n_bboxes_test = config.n_bboxes
    n_classes = 1
    half = n_frames/2
    learning_rate = config.learning_rate_det

    # shuffled_samples = shuffle_samples(n_frames)
    # train_frames = frames_ids[shuffled_samples[0:half]]

    # test_frames = frames_ids[shuffled_samples[half:]]

    train_frames_path = os.path.join(FLAGS.data_dir, 'train.txt')
    train_frames = np.loadtxt(train_frames_path, dtype=int)

    test_frames_path = os.path.join(FLAGS.data_dir, 'val.txt')
    test_frames = np.loadtxt(test_frames_path, dtype=int)

    n_train_samples = len(train_frames)
    n_test_samples = len(test_frames)

    logging.info('building model graph..')

    in_ops = input_ops(config.n_dt_features, n_classes)

    nnms_model = nms_net.NMSNetwork(n_classes=1,
                                    input_ops=in_ops,
                                    loss_type='detection',
                                    gt_match_iou_thr=0.7,
                                    class_ix=0,
                                    **config.nms_network_config)

    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1.0)

    logging.info('training started..')

    with tf.Session() as sess:

        sess.run(nnms_model.init_op)

        step_id = 0
        step_times = []
        data_times = []

        for epoch_id in range(0, 10):

            for fid in train_frames:

                start_step = timer()

                frame_data = get_frame_data_fixed(frame_id=fid,
                                            labels_dir=labels_dir,
                                            detections_dir=detections_dir,
                                            n_detections=config.n_bboxes,
                                            n_features=config.n_dt_features)

                data_step = timer()

                feed_dict = {nnms_model.dt_coords: frame_data['dt_coords'],
                             nnms_model.dt_features: frame_data['dt_features'],
                             nnms_model.dt_probs_ini: frame_data['dt_probs'],
                             nnms_model.gt_coords: frame_data['gt_coords'],
                             nnms_model.gt_labels: frame_data['gt_labels'],
                             # nnms_model.nms_labels: frame_data['nms_labels'],
                             nnms_model.keep_prob: config.keep_prob_train,
                             nnms_model.learning_rate: config.learning_rate_det}

                _ = sess.run([nnms_model.det_train_step],
                                 feed_dict=feed_dict)

                step_id += 1

                end_step = timer()
                step_times.append(end_step-start_step)
                data_times.append(data_step-start_step)

                if step_id % 5000 == 0:

                    logging.info('curr step : %d, mean time for step : %s, for getting data : %s' % (step_id,
                                                                                                     str(np.mean(step_times)),
                                                                                                     str(np.mean(data_times))))

                    # logging.info("eval on TRAIN..")
                    # train_out_dir = os.path.join(config.log_dir, 'train')
                    # train_map_knet, train_map_nms = eval_supp.eval_model(sess,
                    #                                           nnms_model,
                    #                                           detections_dir=detections_dir,
                    #                                           labels_dir=labels_dir,
                    #                                           eval_frames=train_frames,
                    #                                           n_bboxes=config.n_bboxes,
                    #                                           n_features=config.n_dt_features,
                    #                                           global_step=step_id,
                    #                                           out_dir=train_out_dir,
                    #                                           nms_thres=0.75)

                    logging.info("eval on TEST..")
                    test_out_dir = os.path.join(config.log_dir, 'test')
                    test_map_knet, test_map_nms = eval_supp.eval_model(sess,
                                                             nnms_model,
                                                             detections_dir=detections_dir,
                                                             labels_dir=labels_dir,
                                                             eval_frames=test_frames,
                                                             n_bboxes=config.n_bboxes,
                                                             n_features=config.n_dt_features,
                                                             global_step=step_id,
                                                             out_dir=test_out_dir,
                                                             nms_thres=0.75)

                    if test_map_knet > test_map_nms:
                        learning_rate = 0.0001
                        logging.info('decreasing learning rate to %s' % str(learning_rate))

                    config.update_results(step_id,
                                          0,
                                          0,
                                          test_map_knet,
                                          test_map_nms,
                                          np.mean(step_times))
                    config.save_results()
                    saver.save(sess, config.model_file, global_step=step_id)

    train_map_knet, train_map_nms = eval_supp.eval_model(sess,
                                              nnms_model,
                                              detections_dir=detections_dir,
                                              labels_dir=labels_dir,
                                              eval_frames=train_frames,
                                              n_bboxes=config.n_bboxes,
                                              n_features=config.n_dt_features,
                                              nms_thres=0.75)

    test_map_knet, test_map_nms = eval_supp.eval_model(sess,
                                             nnms_model,
                                             detections_dir=detections_dir,
                                             labels_dir=labels_dir,
                                             eval_frames=test_frames,
                                             n_bboxes=config.n_bboxes,
                                             n_features=config.n_dt_features,
                                             nms_thres=0.75)
    return

if __name__ == '__main__':
    gflags.mark_flag_as_required('data_dir')
    gflags.mark_flag_as_required('root_log_dir')
    gflags.mark_flag_as_required('config_path')
    app.run()

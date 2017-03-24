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
from nms_network_old import model as nms_net
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


def main(_):

    config = expconf.ExperimentConfig(data_dir=FLAGS.data_dir,
                                      root_log_dir=FLAGS.root_log_dir,
                                      config_path=FLAGS.config_path)



    logging.info("config info : %s" % config.config)

    labels_dir = os.path.join(FLAGS.data_dir, 'label_2')

    detections_dir = os.path.join(FLAGS.data_dir, 'detection_2')

    frames_ids = np.asarray([int(ntpath.basename(path).split('.')[0]) for path in os.listdir(labels_dir)])

    n_frames = len(frames_ids)
    n_bboxes_test = 20
    n_classes = 1
    half = n_frames/2
    learning_rate = 0.001

    shuffled_samples = shuffle_samples(n_frames)
    train_frames = frames_ids[shuffled_samples[0:half]]
    n_train_samples = len(train_frames)
    test_frames = frames_ids[shuffled_samples[half:]]
    n_test_samples = len(test_frames)

    logging.info('building model graph..')

    in_ops = input_ops(config.n_dt_features, n_classes)

    nnms_model = nms_net.NMSNetwork(n_classes=1,
                                    input_ops=in_ops,
                                    loss_type='nms_loss',
                                    gt_match_iou_thr=0.7,
                                    **config.nms_network_config)

    logging.info('training started..')

    with tf.Session() as sess:

        sess.run(nnms_model.init_op)

        step_id = 0
        step_times = []
        data_times = []
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()

        for epoch_id in range(0, 10):

            for fid in train_frames:

                start_step = timer()

                frame_data = get_frame_data_fixed(frame_id=fid,
                                            labels_dir=labels_dir,
                                            detections_dir=detections_dir,
                                            n_detections=config.n_bboxes,
                                            n_features=config.n_dt_features)

                data_step = timer()

                # import ipdb; ipdb.set_trace()

                feed_dict = {nnms_model.dt_coords: frame_data['dt_coords'],
                             nnms_model.dt_features: frame_data['dt_features'],
                             nnms_model.dt_probs: frame_data['dt_probs'],
                             nnms_model.gt_coords: frame_data['gt_coords'],
                             nnms_model.gt_labels: frame_data['gt_labels'],
                             # nnms_model.nms_labels: frame_data['nms_labels'],
                             nnms_model.keep_prob: config.keep_prob_train,
                             nnms_model.learning_rate: learning_rate}

                # if step_id == 3000:
                #     learning_rate = 0.0001
                #     logging.info('decreasing learning rate to %s' % str(learning_rate))

                # if step_id < 1000:
                #
                #     _ = sess.run([nnms_model.pair_loss_train_step],
                #                  feed_dict=feed_dict,
                #                  options=run_options,
                #                  run_metadata=run_metadata)
                # else:

                _ = sess.run([nnms_model.det_train_step],
                                 feed_dict=feed_dict)

                # import ipdb; ipdb.set_trace()

                step_id += 1

                end_step = timer()
                step_times.append(end_step-start_step)
                data_times.append(data_step-start_step)

                if step_id % 5000 == 0:

                    logging.info('curr step : %d, mean time for step : %s, for getting data : %s' % (step_id,
                                                                                                     str(np.mean(step_times)),
                                                                                                     str(np.mean(data_times))))

                    train_losses = []

                    for tfid in train_frames[0:100]:

                        frame_data = get_frame_data_fixed(frame_id=tfid,
                                                    labels_dir=labels_dir,
                                                    detections_dir=detections_dir,
                                                    n_detections=n_bboxes_test,
                                                    n_features=config.n_dt_features)

                        feed_dict = {nnms_model.dt_coords: frame_data['dt_coords'],
                                     nnms_model.dt_features: frame_data['dt_features'],
                                     nnms_model.dt_probs: frame_data['dt_probs'],
                                     nnms_model.gt_coords: frame_data['gt_coords'],
                                     nnms_model.gt_labels: frame_data['gt_labels'],
                                     # nnms_model.nms_labels: frame_data['nms_labels'],
                                     nnms_model.keep_prob: 1.0}

                        det_loss = sess.run([nnms_model.det_loss],
                                                      feed_dict=feed_dict)

                        train_losses.append(det_loss)

                    logging.info("train loss (det part) : %s" % str(np.mean(train_losses)))

                    test_losses = []

                    for tfid in test_frames[0:100]:

                        frame_data = get_frame_data_fixed(frame_id=tfid,
                                                    labels_dir=labels_dir,
                                                    detections_dir=detections_dir,
                                                    n_detections=n_bboxes_test,
                                                    n_features=config.n_dt_features)

                        feed_dict = {nnms_model.dt_coords: frame_data['dt_coords'],
                                     nnms_model.dt_features: frame_data['dt_features'],
                                     nnms_model.dt_probs: frame_data['dt_probs'],
                                     nnms_model.gt_coords: frame_data['gt_coords'],
                                     nnms_model.gt_labels: frame_data['gt_labels'],
                                     # nnms_model.nms_labels: frame_data['nms_labels'],
                                     nnms_model.keep_prob: 1.0}

                        det_loss = sess.run([nnms_model.det_loss], feed_dict=feed_dict)

                        test_losses.append(det_loss)

                    logging.info("test loss (det part) : %s" % str(np.mean(test_losses)))

                    # fid = test_frames[np.random.randint(0,n_test_samples,1)[0]]
                    # frame_data = get_frame_data_fixed(frame_id=fid,
                    #         labels_dir=labels_dir,
                    #         detections_dir=detections_dir,
                    #         n_detections=n_bboxes_test,
                    #         n_features=config.n_dt_features)
                    #
                    # feed_dict = {nnms_model.dt_coords: frame_data['dt_coords'],
                    #                  nnms_model.dt_features: frame_data['dt_features'],
                    #                  nnms_model.dt_probs: frame_data['dt_probs'],
                    #                  nnms_model.gt_coords: frame_data['gt_coords'],
                    #                  nnms_model.gt_labels: frame_data['gt_labels'],
                    #                  # nnms_model.nms_labels: frame_data['nms_labels'],
                    #                  nnms_model.keep_prob: 1.0}
                    #
                    # #nms_labels = frame_data['nms_labels']
                    #
                    # dt_coords, dt_features, dt_probs, pairwise_features, \
                    # nmsnet_inference, not_supp_prob, nms_loss, det_labels,\
                    # not_explained_prob, nms_pairwise_labels, pairwise_explain_probs, elementwise_loss,\
                    #                         dt_dt_iou, pair_initial, nms_labels_tf,\
                    #                                                 object_and_context_features,\
                    #                                                                         kernel_matrix\
                    #     = sess.run([nnms_model.dt_coords,
                    #                 nnms_model.dt_features,
                    #                 nnms_model.dt_probs,
                    #                 nnms_model.pairwise_features,
                    #                 nnms_model.class_scores,
                    #                 nnms_model.not_supp_prob,
                    #                 nnms_model.det_labels,
                    #                 nnms_model.iou_feature,
                    #                 nnms_model.pair_initial,
                    #                 nnms_model.nms_labels,
                    #                 nnms_model.object_and_context_features,
                    #                 nnms_model.kernel_matrix],
                    #                feed_dict=feed_dict)
                    #
                    # # kernel = np.squeeze(kernel_matrix)
                    # # pos_rescaled_features = rescaled_features[np.where(true_labels == 1)]
                    # # neg_rescaled_features = rescaled_features[np.where(true_labels == 0)]
                    #
                    # pos_inference = nmsnet_inference[np.where(det_labels == 1)]
                    # pos_inference_orig = dt_features[np.where(det_labels == 1)]
                    # neg_inference = nmsnet_inference[np.where(det_labels == 0)]
                    # neg_inference_orig = dt_features[np.where(det_labels == 0)]
                    # logging.info("positive inference samples (original) : %s" % str(pos_inference_orig))
                    # logging.info("positive inference samples (updated) : %s" % str(pos_inference))
                    # logging.info("negative inference samples (original : %s" % str(neg_inference_orig))
                    # logging.info("negative inference samples (updated) : %s" % str(neg_inference))

                    # nms_labels = np.max(nms_pairwise_labels, axis=1)

                    # pos_nms_inference = not_explained_prob[np.where(nms_labels_tf == 1)]
                    # neg_nms_inference = not_explained_prob[np.where(nms_labels_tf == 0)]
                    # logging.info("probs for samples NOT to be suppressed : %s" % str(pos_nms_inference))
                    # logging.info("probs for samples to be suppressed : %s" % str(neg_nms_inference))

                    # suppression_map = pairwise_features[:, :, 2] > pairwise_features[:, :, 1]
                    # iou_map = pairwise_features[:, :, 0] > 0.5
                    #
                    # nms_pairwise = suppression_map & iou_map
                    #
                    # #import ipdb; ipdb.set_trace()
                    #
                    # # iou_np = frame_data['dt_dt_iou']
                    # dt_dt_iou = dt_dt_iou.reshape([n_bboxes_test, n_bboxes_test])


                    # if step_id % 10 == 0:
                    #    import ipdb; ipdb.set_trace()

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

                    if test_map_knet > test_map_nms:
                        learning_rate = 0.0001
                        logging.info('decreasing learning rate to %s' % str(learning_rate))

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
    #import ipdb; ipdb.set_trace()

    return

if __name__ == '__main__':
    gflags.mark_flag_as_required('data_dir')
    gflags.mark_flag_as_required('root_log_dir')
    gflags.mark_flag_as_required('config_path')
    app.run()

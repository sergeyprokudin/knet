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

from tools import bbox_utils
from tf_layers import knet, spatial

gflags.DEFINE_string('data_dir', None, 'directory containing train data')
gflags.DEFINE_string('log_dir', None, 'directory to save logs and trained models')
gflags.DEFINE_integer('data_loader_num_threads', 5, 'Number of threads used during data loading and preprocessing')
gflags.DEFINE_integer('n_kernels', 8, 'Number of kernels to used in knet layer')
gflags.DEFINE_float('best_iou_thres', 0.5, 'Number of threads used during data loading and preprocessing')
gflags.DEFINE_boolean('logging_to_stdout', True, 'Whether to write logs to stdout or to logfile')
gflags.DEFINE_integer('n_epochs', 1000, 'Number of training epochs')

FLAGS = gflags.FLAGS

DT_COORDS='dt_coords'
GT_COORDS='gt_coords'
GT_LABELS='gt_labels'
DT_LABELS='dt_labels'
DT_FEATURES='dt_features'
DT_GT_IOU='dt_gt_iou'
N_DT_COORDS=4
N_DT_FEATURES=21
N_OBJECTS=300
N_CLASSES=20

def get_frame_data(fid, data):
    frame_data = {}
    fid_dt_ix = data[DT_COORDS][:,0]==fid
    frame_data[DT_COORDS] = data[DT_COORDS][fid_dt_ix, 1:]
    frame_data[DT_FEATURES] = data[DT_FEATURES][fid_dt_ix]
    fid_gt_ix = data[GT_COORDS][:,0]==fid
    frame_data[GT_COORDS] = data[GT_COORDS][fid_gt_ix, 1:5]
    frame_data[GT_LABELS] = data[GT_COORDS][fid_gt_ix, 5]-1
    frame_data[DT_GT_IOU] = bbox_utils.compute_sets_iou(frame_data[DT_COORDS], frame_data[GT_COORDS])
    frame_data[DT_LABELS] = np.zeros([N_OBJECTS, N_CLASSES])
    for class_id in range(0, N_CLASSES):
        class_gt_boxes = frame_data[GT_LABELS]==class_id
        class_dt_gt = frame_data[DT_GT_IOU][:, class_gt_boxes]
        if (class_dt_gt.shape[1]!=0):
            frame_data[DT_LABELS][:,class_id] = np.max(bbox_utils.compute_best_iou(class_dt_gt),axis=1)
    return frame_data


def split_by_frames(data):
    unique_fids = np.unique(np.hstack([data[DT_COORDS][:,0], data[GT_COORDS][:,0]])).astype(int)
    pool = multiprocessing.Pool(FLAGS.data_loader_num_threads)
    get_frame_data_partial = partial(get_frame_data, data=data)
    frames_data = dict(zip(unique_fids, pool.map(get_frame_data_partial, unique_fids)))
    return frames_data

def load_data(data_dir):
    data = {}
    data[DT_COORDS] = joblib.load(os.path.join(data_dir, 'dt_coords.pkl'))
    data[DT_FEATURES] = joblib.load(os.path.join(data_dir, 'dt_features.pkl'))
    data[GT_COORDS] = joblib.load(os.path.join(data_dir, 'gt_coords.pkl'))
    frames_data = split_by_frames(data)
    return frames_data


def set_logging(to_stdout=True, log_file=None):
    if (to_stdout):
        logging.basicConfig(
        format='%(asctime)s : %(message)s',
        level=logging.INFO,
        stream=sys.stdout)
    else :
        logging.basicConfig(
        format='%(asctime)s : %(message)s',
        level=logging.INFO,
        filename=log_file)
        print("logs could be found at %s"%log_file)
    return

def shuffle_samples(n_frames):
    return np.random.choice(n_frames, n_frames, replace=False)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def main(_):

    if (not os.path.exists(FLAGS.log_dir)):
        os.makedirs(FLAGS.log_dir)
    else :
        shutil.rmtree(FLAGS.log_dir)
    log_file=os.path.join(FLAGS.log_dir, 'training.log')
    set_logging(FLAGS.logging_to_stdout, log_file)

    #loading data
    frames_data_cache_file = os.path.join(FLAGS.data_dir, 'frames_data.pkl')
    if (os.path.exists(frames_data_cache_file)):
        logging.info('loading frame bbox data info from cash..')
        frames_data = joblib.load(frames_data_cache_file)
    else :
        logging.info('computing frame bbox data (IoU, labels, etc) - this could take some time..')
        frames_data = load_data(FLAGS.data_dir)
        joblib.dump(frames_data, frames_data_cache_file)

    n_frames = len(frames_data.keys())
    #model definition
    dt_coords_tf = tf.placeholder(tf.float32, shape=[N_OBJECTS, N_DT_COORDS], name=DT_COORDS)
    dt_features_tf = tf.placeholder(tf.float32, shape=[N_OBJECTS, N_DT_FEATURES], name=DT_FEATURES)
    dt_labels_tf = tf.placeholder(tf.float32, shape=[N_OBJECTS, N_CLASSES], name=DT_LABELS)

    pairwise_features_tf = spatial.construct_pairwise_features_tf(dt_coords_tf)
    iou_feature_tf = spatial.compute_pairwise_spatial_features_iou_tf(pairwise_features_tf)

    pairwise_scores_tf =  spatial.construct_pairwise_features_tf(dt_features_tf)
    spatial_features_tf = tf.concat(2, [iou_feature_tf, pairwise_scores_tf])
    n_spatial_features = N_DT_FEATURES*2+1

    dt_new_features_tf, knet_ops_tf = knet.knet_layer(dt_features_tf, spatial_features_tf, n_kernels=FLAGS.n_kernels, n_objects=N_OBJECTS, n_pair_features=n_spatial_features, n_object_features=N_DT_FEATURES)

    W_fc1 = weight_variable([FLAGS.n_kernels*N_DT_FEATURES, N_CLASSES])
    b_fc1 = bias_variable([N_CLASSES])

    inference_tf = tf.matmul(dt_new_features_tf, W_fc1) + b_fc1

    loss_tf = tf.nn.weighted_cross_entropy_with_logits(inference_tf, dt_labels_tf, pos_weight=1000)

    loss_final_tf = tf.reduce_mean(loss_tf)

    tf.summary.scalar('cross_entropy_loss',loss_final_tf)

    merged_summaries = tf.summary.merge_all()

    train_step = tf.train.AdamOptimizer(0.001).minimize(loss_final_tf)

    with tf.Session() as sess:
        step_id = 0
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(
            max_to_keep=5,
            keep_checkpoint_every_n_hours=1.0)

        if (os.path.exists(FLAGS.log_dir)):
            ckpt_path = tf.train.latest_checkpoint(FLAGS.log_dir)
            if (ckpt_path is not None):
                ckpt_name=ntpath.basename(ckpt_path)
                step_id=int(ckpt_name.split('-')[1])
                saver.restore(sess, ckpt_path)

        model_file = os.path.join(FLAGS.log_dir, 'model')

        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        for epoch_id in range(0, FLAGS.n_epochs):
            for fid in shuffle_samples(n_frames):
                frame_data = frames_data[fid]
                feed_dict = {dt_coords_tf:frame_data[DT_COORDS][0:N_OBJECTS],
                            dt_features_tf:frame_data[DT_FEATURES][0:N_OBJECTS],
                            dt_labels_tf:frame_data[DT_LABELS][0:N_OBJECTS]}
                summary, _ = sess.run([merged_summaries, train_step],
                                                feed_dict=feed_dict)
                summary_writer.add_summary(summary, global_step=step_id)
                summary_writer.flush()
                step_id+=1
                if (step_id%1000==0):
                    loss, loss_final, inference, iou_feature,   knet_ops = sess.run([loss_tf, loss_final_tf, inference_tf, iou_feature_tf, knet_ops_tf],
                                                                    feed_dict=feed_dict)
                    logging.info("epoch %d loss for frame %d : %f"%(epoch_id, fid, loss_final))
                    #logging.info("initail scores for pos values : %s"%frame_data[DT_FEATURES][np.where(frame_data[DT_LABELS][0:N_OBJECTS]>0)])
                    logging.info("non-neg kernel elements : %d"%np.sum(knet_ops['kernels']>0))
                    logging.info("inference for pos values : %s"%inference[np.where(frame_data[DT_LABELS][0:N_OBJECTS]>0)])
                    #logging.info("pairwise_features : %s"%knet_ops['pairwise_features'])
                    logging.info("kernel : %s"%knet_ops['kernels'])
                    num_gt = int(np.sum(frame_data[DT_LABELS]))
                    num_pos_inference = int(np.sum(inference>0))
                    logging.info("frame %d num_gt : %d , num_pos_inf : %d"%(fid, num_gt, num_pos_inference))
                    #import ipdb; ipdb.set_trace() #; ipdb.set_trace=False
                    saver.save(sess, model_file, global_step=step_id)
    return

if __name__ == '__main__':
    gflags.mark_flag_as_required('data_dir')
    gflags.mark_flag_as_required('log_dir')
    app.run()

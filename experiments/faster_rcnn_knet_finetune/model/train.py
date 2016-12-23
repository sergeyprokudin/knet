"""Training routine for knet trained on top of FasterRCNN inference

    data_dir - directory containing train data :
                dt_coords.pkl - detections' bounding boxes coordinates in format [frame_id, x_min, y_min, width, height]
                dt_features.pkl - detections' bounding boxes features
                gt_coords.pkl - ground truth bounding boxes coordinates in format [frame_id, x_min, y_min, width, height, class_id]

"""

import numpy as np
import tensorflow as tf
import joblib
import multiprocessing
from functools import partial
import os
import gflags
import logging
from google.apputils import app
from tools import bbox_utils
from tf_layers import knet, spatial

gflags.DEFINE_string('data_dir', None, 'directory containing train data')
gflags.DEFINE_integer('data_loader_num_threads', 5, 'Number of threads used during data loading and preprocessing')
gflags.DEFINE_float('best_iou_thres', 0.5, 'Number of threads used during data loading and preprocessing')

FLAGS = gflags.FLAGS

DT_COORDS='dt_coords'
GT_COORDS='gt_coords'
GT_LABELS='gt_labels'
DT_LABELS='dt_labels'
DT_FEATURES='dt_features'
DT_GT_IOU='dt_gt_iou'
N_DT_COORDS=4
N_DT_FEATURES=20

def get_frame_data(fid, data):
    frame_data = {}
    fid_dt_ix = data[DT_COORDS][:,0]==fid
    frame_data[DT_COORDS] = data[DT_COORDS][fid_dt_ix, 1:]
    frame_data[DT_FEATURES] = data[DT_FEATURES][fid_dt_ix, 1:]
    fid_gt_ix = data[GT_COORDS][:,0]==fid
    frame_data[GT_COORDS] = data[GT_COORDS][fid_gt_ix, 1:5]
    frame_data[GT_LABELS] = data[GT_COORDS][fid_gt_ix, 5]
    frame_data[DT_GT_IOU] = bbox_utils.compute_sets_iou(frame_data[DT_COORDS], frame_data[GT_COORDS])
    frame_data[DT_LABELS] = np.max(bbox_utils.compute_best_iou(frame_data[DT_GT_IOU]),axis=1)
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

def main(_):
    frames_data = load_data(FLAGS.data_dir)
    spatial_features_ops = tf.placeholder(tf.float32, shape=[None, N_DT_COORDS], name=DT_COORDS)
    pairwise_features_ops = spatial.construct_pairwise_features_tf(spatial_features_ops)
    object_features_ops = tf.placeholder(tf.float32, shape=[None, N_DT_FEATURES], name=DT_FEATURES)
    inference_op = knet.knet_layer(object_features_ops, pairwise_features_ops, n_kernels=4)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for fid in frames_data.keys():
            batch_data = frames_data[fid]
            #best_iou = bbox_utils.compute_best_iou(batch_data[DT_GT_IOU])
            inference = sess.run(inference_op, feed_dict={spatial_features_ops:batch_data[DT_COORDS],
                                                               object_features_ops:batch_data[DT_FEATURES]})
            import ipdb; ipdb.set_trace();#ipdb.set_trace=False
    return

if __name__ == '__main__':
    gflags.mark_flag_as_required('data_dir')
    app.run()

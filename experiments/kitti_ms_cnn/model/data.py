import pandas as pd
import scipy.io as sio
import numpy as np
import os
from tools import bbox_utils
from tools import nms


def get_frame_data(frame_id,
                   labels_dir,
                   detections_dir,
                   n_detections=None,
                   n_features=None):

    dt_fname = format(frame_id, '06') + '_car.mat'
    gt_fname = format(frame_id, '06') + '.txt'

    frame_dt_path = os.path.join(detections_dir, dt_fname)
    frame_gt_path = os.path.join(labels_dir, gt_fname)

    frame_data = {}

    gt_info = pd.read_csv(frame_gt_path, sep=' ',
                          usecols=[0, 4, 5, 6, 7],
                          names=['class', 'x1', 'y1', 'w', 'h'])

    gt_info = gt_info[gt_info['class'] == 'Car']

    dt_info = sio.loadmat(frame_dt_path)

    n_all_features, n_all_detections = dt_info['feat'].shape

    if n_detections is None:
        n_detections = n_all_detections
    if n_features is None:
        n_features = n_all_features

    frame_data['gt_coords'] = np.asarray(gt_info.iloc[:, 1:])
    frame_data['gt_labels'] = np.zeros(frame_data['gt_coords'].shape[0])
    frame_data['dt_probs'] = dt_info['detection_result'][0, 0][0:n_detections, 4].reshape(-1, 1)
    frame_data['dt_features'] = dt_info['feat'][0:n_features, 0:n_detections].T

    # frame_data['dt_features'] = frame_data['dt_probs']
    frame_data['dt_coords'] = dt_info['detection_result'][0, 0][0:n_detections, 0:4]

    # convert x,y,w,h -> x_min, y_min, x_max, y_max
    frame_data['dt_coords'][:, 2] = frame_data['dt_coords'][:, 0] + frame_data['dt_coords'][:, 2]
    frame_data['dt_coords'][:, 3] = frame_data['dt_coords'][:, 1] + frame_data['dt_coords'][:, 3]

    frame_data['detection_result'] = dt_info['detection_result']

    frame_data['dt_gt_iou'] = bbox_utils.compute_sets_iou(frame_data['dt_coords'], frame_data['gt_coords'])

    frame_data['dt_dt_iou'] = bbox_utils.compute_sets_iou(frame_data['dt_coords'], frame_data['dt_coords'])
    frame_data['is_suppressed'] = nms.nms_all_classes(frame_data['dt_dt_iou'], frame_data['dt_probs'], iou_thr=0.5)
    frame_data['nms_label'] = np.zeros()

    # import ipdb; ipdb.set_trace()
    # a = b

    return frame_data
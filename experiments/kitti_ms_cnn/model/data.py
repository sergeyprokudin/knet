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
    frame_data['dt_features'] = np.hstack([frame_data['dt_features'], frame_data['dt_probs']])

    frame_data['dt_coords'] = dt_info['detection_result'][0, 0][0:n_detections, 0:4]

    # convert x,y,w,h -> x_min, y_min, x_max, y_max
    frame_data['dt_coords'][:, 2] = frame_data['dt_coords'][:, 0] + frame_data['dt_coords'][:, 2]
    frame_data['dt_coords'][:, 3] = frame_data['dt_coords'][:, 1] + frame_data['dt_coords'][:, 3]

    frame_data['detection_result'] = dt_info['detection_result']

    frame_data['dt_gt_iou'] = bbox_utils.compute_sets_iou(frame_data['dt_coords'], frame_data['gt_coords'])


    # frame_data['dt_dt_iou'] = bbox_utils.compute_sets_iou(frame_data['dt_coords'], frame_data['dt_coords'])

    # frame_data['nms_labels'] = np.invert(nms.nms_all_classes(frame_data['dt_dt_iou'],
    #                                                      frame_data['dt_probs'],
    #                                                       iou_thr=0.5)).astype('int')

    # a = b

    return frame_data


def get_frame_data_fixed(frame_id,
                   labels_dir,
                   detections_dir,
                   class_name='Pedestrian',
                   n_detections=None,
                   n_features=None):
    """
    Loads data in a fixed size bbox container
    Parameters
    ----------
    frame_id
    labels_dir
    detections_dir
    n_detections
    n_features

    Returns
    -------

    """
    if class_name == 'Car':
        dt_fname = format(frame_id, '06') + '_car.mat'
    else:
        dt_fname = format(frame_id, '06') + '.png_ped.mat'

    gt_fname = format(frame_id, '06') + '.txt'

    frame_dt_path = os.path.join(detections_dir, dt_fname)
    frame_gt_path = os.path.join(labels_dir, gt_fname)

    frame_data = {}

    gt_info = pd.read_csv(frame_gt_path, sep=' ',
                          usecols=[0, 4, 5, 6, 7],
                          names=['class', 'x1', 'y1', 'w', 'h'])

    # gt_info = gt_info[gt_info['class'].isin(['Car', 'Van', 'Truck', 'Bus'])]

    gt_info = gt_info[gt_info['class'] == class_name]

    dt_info = sio.loadmat(frame_dt_path)

    if class_name == 'Car':
        n_all_features, n_all_detections = dt_info['feat'].shape
    else:
        n_all_features = dt_info['feat'].shape[0]
        n_all_detections = dt_info['detection_result'][0, 0].shape[0]

    n_detections_actual = min(n_all_detections, n_detections)

    if n_detections is None:
        n_detections = n_all_detections
    if n_features is None:
        n_features = n_all_features

    frame_data['gt_coords'] = np.asarray(gt_info.iloc[:, 1:])
    frame_data['gt_labels'] = np.zeros(frame_data['gt_coords'].shape[0])

    frame_data['dt_probs'] = np.zeros([n_detections, 1])

    # if len(gt_info != 0):
    #     import ipdb; ipdb.set_trace()

    if class_name == 'Car':
        frame_data['dt_probs'][0:n_detections_actual] = dt_info['detection_result'][0, 0][0:n_detections_actual, 4].reshape(-1, 1)
    elif class_name == 'Pedestrian':
        frame_data['dt_probs'][0:n_detections_actual] = dt_info['detection_result'][0, 0][0:n_detections_actual, 4].reshape(-1, 1)
    elif class_name == 'Cyclist':
        frame_data['dt_probs'][0:n_detections_actual] = dt_info['detection_result'][1, 0][0:n_detections_actual, 4].reshape(-1, 1)

    frame_data['dt_features'] = np.zeros([n_detections, n_features])
    frame_data['dt_features'][0:n_detections_actual] = dt_info['feat'][0:n_features, 0:n_detections_actual].T
    frame_data['dt_features'] = np.hstack([frame_data['dt_probs'], frame_data['dt_features']])

    frame_data['dt_coords'] = np.zeros([n_detections, 4])
    frame_data['dt_coords'][:, 2] = 1
    frame_data['dt_coords'][:, 3] = 1
    frame_data['dt_coords'][0:n_detections_actual] = dt_info['detection_result'][0, 0][0:n_detections_actual, 0:4]

    # convert x,y,w,h -> x_min, y_min, x_max, y_max
    frame_data['dt_coords'][:, 2] = frame_data['dt_coords'][:, 0] + frame_data['dt_coords'][:, 2]
    frame_data['dt_coords'][:, 3] = frame_data['dt_coords'][:, 1] + frame_data['dt_coords'][:, 3]

    #frame_data['detection_result'] = dt_info['detection_result']

    frame_data['dt_gt_iou'] = bbox_utils.compute_sets_iou(frame_data['dt_coords'], frame_data['gt_coords'])

    #
    # frame_data['dt_dt_iou'] = bbox_utils.compute_sets_iou(frame_data['dt_coords'], frame_data['dt_coords'])

    # frame_data['nms_labels'] = np.invert(nms.nms_all_classes(frame_data['dt_dt_iou'],
    #                                                      frame_data['dt_probs'],
    #                                                       iou_thr=0.5)).astype('int')

    return frame_data

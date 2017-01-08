"""Library containing different operations on bounding boxes
"""

import numpy as np


def compute_iou(bb1, bb2):
    """compute intersection over union between 2 boxes
    of format [x1, y1, x2, y2]
    """
    bb1_x1, bb1_y1, bb1_x2, bb1_y2 = bb1[:4].tolist()
    bb2_x1, bb2_y1, bb2_x2, bb2_y2 = bb2[:4].tolist()
    x_overlap = max(0, min(bb1_x2, bb2_x2) - max(bb1_x1, bb2_x1))
    y_overlap = max(0, min(bb1_y2, bb2_y2) - max(bb1_y1, bb2_y1))
    intersection_area = x_overlap * y_overlap
    w1 = bb1_x2 - bb1_x1
    h1 = bb1_y2 - bb1_y1
    w2 = bb2_x2 - bb2_x1
    h2 = bb2_y2 - bb2_y1
    union_area = w1 * h1 + w2 * h2 - intersection_area
    ratio = float(intersection_area) / union_area
    return ratio


def compute_sets_iou(bb_set1, bb_set2):
    """compute intersection over union between 2 sets of bounding boxes
    """
    iou = np.zeros(shape=(len(bb_set1), len(bb_set2)))
    for bb1_ix, bb1 in enumerate(bb_set1):
        for bb2_ix, bb2 in enumerate(bb_set2):
            iou[bb1_ix, bb2_ix] = compute_iou(bb1[0:4], bb2[0:4])
    return iou


def compute_best_iou(iou, iou_threshold=0.5):
    """Given IoU matrix, find best matching pairs (above defined IoU threshold)

    Will return array of the same shape as input with 1 on positions for best-matching pairs

    Example :

    [0.9 0.7 0.2 0.1]       [1 0 0]
    [0.6 0.1 0.5 0.2]  ->   [0 0 1]
    [0.0 0.8 0.3 0.1]       [0 1 0]
    [0.1 0.2 0.1 0.1]       [0 0 0]
    """
    thres_mask = np.zeros(iou.shape)
    thres_mask[np.where(iou>=iou_threshold)] = 1
    best_iou = np.zeros(iou.shape)
    coords_sorted = np.unravel_index(
        np.argsort(iou, axis=None)[::-1], iou.shape)
    coords_sorted = np.asarray(coords_sorted).T
    mask = np.ones(iou.shape)
    for i, j in coords_sorted:
        if (mask[i,j]!=0) :
            best_iou[i, j] = 1
            mask[i, :] = 0
            mask[:, j] = 0
            if (np.count_nonzero(mask) == 0):
                break
    best_iou = np.multiply(best_iou, thres_mask)
    return best_iou

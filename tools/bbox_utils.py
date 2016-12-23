"""Library containing different operations on bounding boxes
"""

import numpy as np

def compute_iou(bb_xywh1, bb_xywh2):
    """compute intersection over union between 2 boxes
    """
    x1, y1, w1, h1 = bb_xywh1[:4].tolist()
    x2, y2, w2, h2 = bb_xywh2[:4].tolist()
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection_area = x_overlap * y_overlap
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

    [0.9 0.7 0.2]       [1 0 0]
    [0.6 0.1 0.5]  ->   [0 0 1]
    [0.0 0.8 0.3]       [0 1 0]
    """
    best_iou = np.zeros(iou.shape)
    coords_sorted = np.unravel_index(np.argsort(iou, axis=None)[::-1],iou.shape)
    coords_sorted = np.asarray(coords_sorted).T
    mask = np.ones(iou.shape)
    for i,j in coords_sorted:
        best_iou[i,j] = 1 * mask[i,j]
        mask[i,:] = 0
        mask[:,j] = 0
        if (np.count_nonzero(mask)==0):
            break
    return best_iou

import numpy as np


def nms_per_class(dt_dt_iou, dt_scores, iou_thr=0.5):
    """Perform greedy Non-Maximum Suppression given intersection over union (IoU) info and detection scores

     Parameters
    ----------
    dt_dt_iou : array, shape = [n_dt_boxes, n_gt_boxes]
        Intersection over union (IoU) ratio between each pair of dt boxes
    dt_scores : array, shape = [n_dt_boxes]
        Confidence scores
    iou_thr : int in range (0, 1]
        Threshold for dt-dt IoU under which dt-dt pairs will be considered duplicate detections
    Returns
    -------
    dt_is_suppressed : array, shape = [n_dt_boxes]
        Array with indicators showing whether the box to be suppressed
    """
    n_dt = dt_scores.shape[0]
    dt_is_suppressed = np.zeros(n_dt)
    order_by_score = np.argsort(dt_scores)[::-1]
    dt_scores = dt_scores[order_by_score]
    dt_dt_iou = dt_dt_iou[order_by_score][:, order_by_score]
    dt_dt_iou[dt_dt_iou >= iou_thr] = 1
    dt_dt_iou[dt_dt_iou < iou_thr] = 0
    til = np.triu_indices(n_dt)
    dt_dt_iou[til] = 0
    dt_is_suppressed[order_by_score] = np.sum(dt_dt_iou, axis=1) > 0
    return dt_is_suppressed

def nms_per_class_with_oracle(dt_dt_iou, dt_scores, dt_gt_iou, iou_thr=0.5):
    """Perform oracle Non-Maximum Suppression given intersection over union (IoU) info, detection scores and
    information about intersections with ground truth objects

    Idea : if box has IoU with already selected, we will ask oracle whether it should be suppressed or not by previous detections.
    If it have no intersection, we won't suppress it no matter what oracle will say.

     Parameters
    ----------
    dt_dt_iou : array, shape = [n_dt_boxes, n_gt_boxes]
        Intersection over union (IoU) ratio between each pair of dt boxes
    dt_scores : array, shape = [n_dt_boxes]
        Confidence scores
    iou_thr : int in range (0, 1]
        Threshold for dt-dt IoU under which dt-dt pairs will be considered duplicate detections
    Returns
    -------
    dt_is_suppressed : array, shape = [n_dt_boxes]
        Array with indicators showing whether the box to be suppressed
    """
    n_dt = dt_scores.shape[0]
    dt_is_suppressed = np.zeros(n_dt, dtype=bool)
    order_by_score = np.argsort(dt_scores)[::-1]
    dt_scores = dt_scores[order_by_score]
    dt_dt_iou = dt_dt_iou[order_by_score][:, order_by_score]
    dt_gt_iou = dt_gt_iou[order_by_score]
    dt_dt_iou[dt_dt_iou >= iou_thr] = 1
    dt_dt_iou[dt_dt_iou < iou_thr] = 0
    dt_gt_iou[dt_gt_iou >= iou_thr] = 1
    dt_gt_iou[dt_gt_iou < iou_thr] = 0
    til = np.triu_indices(n_dt)
    dt_dt_iou[til] = 0
    dt_is_suppressed_ini = np.sum(dt_dt_iou, axis=1) > 0
    if dt_gt_iou.shape[1] == 0:
        dt_is_suppressed[order_by_score] = dt_is_suppressed_ini
        return dt_is_suppressed
    dt_gt_iou_seq = np.copy(dt_gt_iou)
    dt_is_suppressed_oracle = np.zeros(n_dt, dtype=bool)
    for dt_ix in range(0, n_dt):
        to_be_suppressed_ini = bool(dt_is_suppressed_ini[dt_ix])
        has_gt = bool(np.max(dt_gt_iou_seq[dt_ix]))
        if (not has_gt) and to_be_suppressed_ini:
            dt_is_suppressed_oracle[dt_ix] = True
        if has_gt:
            gt_ix = np.argmax(dt_gt_iou_seq[dt_ix])
            dt_gt_iou_seq[:, gt_ix] = 0
            dt_gt_iou_seq[dt_ix, :] = 0
    dt_is_suppressed[order_by_score] = dt_is_suppressed_oracle
    return dt_is_suppressed


def nms_all_classes(dt_dt_iou,
                    dt_scores,
                    iou_thr=0.5):
    """Perform greedy Non-Maximum Suppression given intersection over union (IoU) info and detection scores for all classes

     Parameters
    ----------
    dt_dt_iou : array, shape = [n_dt_boxes, n_gt_boxes]
        Intersection over union (IoU) ratio between each pair of dt boxes
    dt_scores : array, shape = [n_dt_boxes, n_classes]
        Confidence scores
    iou_thr : int in range (0, 1]
        Threshold for dt-dt IoU under which dt-dt pairs will be considered duplicate detections
    Returns
    -------
    dt_is_suppressed : array, shape = [n_dt_boxes, n_classes]
        Array with indicators showing whether the box to be suppressed while considering specific class
    """
    n_dt, n_classes = dt_scores.shape
    dt_is_suppressed = np.zeros([n_dt, n_classes], dtype=bool)
    for class_label in range(0, n_classes):
        class_predictions = dt_scores[:, class_label]
        dt_is_suppressed[:, class_label] = nms_per_class(
            dt_dt_iou, class_predictions, iou_thr=iou_thr)
    return dt_is_suppressed


def nms_all_classes_all_thresholds(dt_dt_iou,
                                   dt_scores,
                                   thrs):
    """ Perform NMS for variety of IoU thresholds
    """
    n_dt, n_classes = dt_scores.shape

    n_thrs = len(thrs)
    dt_is_suppressed = np.zeros([n_dt, n_classes, n_thrs], dtype=bool)
    for i in range(0, n_thrs):
        dt_is_suppressed[:, :, i] = nms_all_classes(
            dt_dt_iou, dt_scores, iou_thr=thrs[i])
    return dt_is_suppressed

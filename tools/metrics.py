import numpy as np


def match_dt_gt(dt_gt_iou,
                predictions,
                iou_thr=0.5):
    """Sequentially match detections to ground truth

    Parameters
    ----------
    dt_gt_iou : array, shape = [n_dt_boxes, n_gt_boxes]
        Intersection over union (IoU) ratio between dt and gt boxes
    predictions : array, shape = [n_dt_boxes]
        Confidence scores for class
    iou_thr : int in range (0, 1]
        Threshold for dt-gt IoU under which dt-gt pair will be considered match
    ----------
    Returns
    ----------
    is_matched : array, shape = [n_dt_boxes]
        Boolean array with information about matches for detections
        (1 - dt matches some ground truth, 0 - false positive)
    """
    # select info connected with label of interest
    n_hyp, n_gt = dt_gt_iou.shape
    if (n_gt == 0):
        return np.zeros(n_hyp)
    # sort by label confidence
    order_by_score = np.argsort(predictions)[::-1]
    thresholds = predictions[order_by_score]
    dt_gt_iou = dt_gt_iou[order_by_score]
    if (n_hyp != 0 and n_gt == 0):
        precision = np.zeros(n_hyp)
        recall = np.ones(n_hyp)
        return precision, recall, thresholds
    # select unique gt (with max IoU) for every hypothesis
    cix_max_iou = np.argmax(dt_gt_iou, axis=1)
    rix_max_iou = np.arange(0, dt_gt_iou.shape[0])
    dt_gt_iou_max = np.zeros(dt_gt_iou.shape)
    dt_gt_iou_max[rix_max_iou, cix_max_iou] = 1
    # binarize matches by decision threshold
    dt_gt_matched = np.copy(dt_gt_iou)
    dt_gt_matched[dt_gt_matched >= iou_thr] = 1
    dt_gt_matched[dt_gt_matched < iou_thr] = 0
    # combine with unique match info
    dt_gt_matched = np.multiply(dt_gt_matched, dt_gt_iou_max)
    # select unique hypothesis for every gt
    rix_best_hyp = np.argmax(dt_gt_matched, axis=0)
    cix_best_hyp = np.arange(0, dt_gt_matched.shape[1])
    dt_gt_bestmatch = np.zeros(dt_gt_matched.shape)
    dt_gt_bestmatch[rix_best_hyp, cix_best_hyp] = 1
    dt_gt_umatch = np.multiply(dt_gt_matched, dt_gt_bestmatch)
    is_matched = np.sum(dt_gt_umatch, axis=1)
    is_matched_ordered = np.zeros(is_matched.shape)
    is_matched_ordered[order_by_score] = is_matched
    return is_matched_ordered


def match_dt_gt_all_classes(dt_gt_iou,
                            gt_labels,
                            dt_predictions,
                            iou_thr=0.5,
                            dt_is_suppressed_info=None):
    """ Sequentially match detections to ground truth for all available classes
    -----
    Parameters


    -----
    Returns
    -----
    is_matched_all_classes : array, shape = [n_detections, n_classes]
        Array containing info about matches :
                        0 - not matched,
                        1 - matched,
                       -1 - suppressed by previous detections
    ----
    """
    n_hyp, n_classes = dt_predictions.shape
    is_matched_all_classes = np.zeros([n_hyp, n_classes])
    for class_label in range(0, n_classes):
        class_dt_gt_iou = dt_gt_iou[:, gt_labels == class_label]
        class_predictions = dt_predictions[:, class_label]
        if (dt_is_suppressed_info is not None):
            dt_is_suppressed_per_class = dt_is_suppressed_info[:, class_label]
            class_predictions = np.copy(
                class_predictions[
                    dt_is_suppressed_per_class == False])
            class_dt_gt_iou = np.copy(
                class_dt_gt_iou[
                    dt_is_suppressed_per_class == False])
            is_matched_all_classes[
                dt_is_suppressed_per_class == False,
                class_label] = match_dt_gt(
                class_dt_gt_iou,
                class_predictions,
                iou_thr=iou_thr)
            is_matched_all_classes[
                dt_is_suppressed_per_class == True,
                class_label] = -1
        else:
            is_matched_all_classes[
                :, class_label] = match_dt_gt(
                class_dt_gt_iou, class_predictions,iou_thr=iou_thr)
    return is_matched_all_classes


def precision_recall_curve(is_matched, predictions, gt_cnt):
    """ Compute precision recall curve per class
    Parameters
    --------
    is_matched : array, shape = [n_detections]
        Information about whether detection was matched to some ground truth

    predictions : array, shape = [n_detections]
        Confidence scores for detections
   --------
   Returns
    -------
    precision : array, shape = [n_thresholds + 1]
        Precision values such that element i is the precision of
        predictions with score >= thresholds[i] and the last element is 1.
    recall : array, shape = [n_thresholds + 1]
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.
    thresholds : array, shape = [n_thresholds <= len(np.unique(probas_pred))]
        Increasing thresholds on the decision function used to compute
        precision and recall.
    """
    order_by_score = np.argsort(predictions)[::-1]
    is_matched = is_matched[order_by_score]
    predictions = predictions[order_by_score]
    # compute true positives, false positives, etc
    tp_cnt = is_matched.cumsum()
    dt_cnt = np.arange(1, predictions.shape[0] + 1)
    recall = tp_cnt / gt_cnt
    precision = tp_cnt / dt_cnt
    return precision, recall, predictions


def average_precision(recall, precision):
    prev_recall = np.zeros(len(recall))
    prev_recall[1:] = recall[0:-1]
    recall_diff = recall - prev_recall
    avg_precision = np.sum(precision * recall_diff)
    return avg_precision


def average_precision_all_classes(is_matched, dt_predictions, gt_labels):
    """ Compute average precision scores for all classes
    Parameters
    ----
    is_matched : array [n_detections, n_classes]
        Information about matches with gt for all available labels

    dt_predictions : array [n_detections, n_classes]
        Confidence scores for all classes

    Returns
    ----
    classes_ap : array [n_classes]
        Average precision for all classes
    ----
    """
    n_classes = dt_predictions.shape[1]
    classes_ap = np.zeros(n_classes)
    classes_roc_curves = {}
    for class_label in range(0, n_classes):
        n_gt_classes = gt_labels[gt_labels == class_label].shape[0]
        class_is_matched = is_matched[:, class_label]
        class_predictions = dt_predictions[:, class_label]
        not_suppressed = class_is_matched != -1
        class_predictions = class_predictions[not_suppressed]
        class_is_matched = class_is_matched[not_suppressed]
        precision, recall, thr = precision_recall_curve(
            class_is_matched, class_predictions, n_gt_classes)
        classes_ap[class_label] = average_precision(recall, precision)
        classes_roc_curves[class_label] = [precision, recall, thr]
    return classes_ap, classes_roc_curves

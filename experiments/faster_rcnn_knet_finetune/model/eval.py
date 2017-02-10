import numpy as np
import joblib
import os


from tools import bbox_utils, nms, metrics
import model as nnms
import logging

def softmax(logits):
    n_classes = logits.shape[1]
    return np.exp(logits) / np.tile(np.sum(np.exp(logits),
                                           axis=1).reshape(-1, 1), [1, n_classes])


def eval_model(sess, nnms_model, frames_data,
               global_step, out_dir, full_eval=False,
               nms_thres=0.5, n_eval_frames=100):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dt_gt_match_orig = []
    dt_gt_match_new = []
    dt_gt_match_orig_nms = []
    dt_gt_match_new_nms = []

    inference_orig_all = []
    inference_new_all = []
    coords = []
    gt_labels_all = []

    n_total_frames = len(frames_data.keys())

    if (full_eval):
        n_eval_frames = n_total_frames

    eval_data = {}

    # for fid in shuffle_samples(n_total_frames)[0:n_eval_frames]:
    for fid in range(0, n_eval_frames):

        eval_data[fid] = {}

        frame_data = frames_data[fid]

        gt_labels_all.append(frame_data[nnms.GT_LABELS].reshape(-1, 1))

        feed_dict = {nnms_model.dt_coords: frame_data[nnms.DT_COORDS],
                     nnms_model.dt_features: frame_data[nnms.DT_FEATURES]}

        dt_scores = frame_data[nnms.DT_SCORES]
        inference_orig = softmax(dt_scores)
        eval_data[fid]['dt_coords'] = frame_data[nnms.DT_COORDS]
        inference_orig_all.append(inference_orig)
        eval_data[fid]['inference_orig'] = inference_orig

        inference_new, dt_dt_iou = sess.run(
            [nnms_model.class_probs, nnms_model.iou_feature], feed_dict=feed_dict)
        inference_new_all.append(inference_new)
        eval_data[fid]['inference_new'] = inference_new

        dt_gt_match_orig.append(
            metrics.match_dt_gt_all_classes(
                frame_data[nnms.DT_GT_IOU],
                frame_data[nnms.GT_LABELS],
                inference_orig))

        dt_gt_match_new.append(
            metrics.match_dt_gt_all_classes(
                frame_data[nnms.DT_GT_IOU],
                frame_data[nnms.GT_LABELS],
                inference_new))

        is_suppressed_orig = nms.nms_all_classes(
            dt_dt_iou, inference_orig, iou_thr=nms_thres)
        is_suppressed_new = nms.nms_all_classes(
            dt_dt_iou, inference_new, iou_thr=nms_thres)

        dt_gt_match_orig_nms.append(
            metrics.match_dt_gt_all_classes(
                frame_data[nnms.DT_GT_IOU],
                frame_data[nnms.GT_LABELS],
                inference_orig,
                dt_is_suppressed_info=is_suppressed_orig))
        dt_gt_match_new_nms.append(
            metrics.match_dt_gt_all_classes(
                frame_data[nnms.DT_GT_IOU],
                frame_data[nnms.GT_LABELS],
                inference_new,
                dt_is_suppressed_info=is_suppressed_new))

    gt_labels = np.vstack(gt_labels_all)
    inference_orig = np.vstack(inference_orig_all)
    inference_new = np.vstack(inference_new_all)

    dt_gt_match_orig = np.vstack(dt_gt_match_orig)
    dt_gt_match_new = np.vstack(dt_gt_match_new)
    dt_gt_match_orig_nms = np.vstack(dt_gt_match_orig_nms)
    dt_gt_match_new_nms = np.vstack(dt_gt_match_new_nms)

    if full_eval:
        eval_data_file = os.path.join(
            out_dir, 'eval_data_step' + str(global_step) + '.pkl')
        joblib.dump(eval_data, eval_data_file)

    ap_orig, _ = metrics.average_precision_all_classes(
        dt_gt_match_orig, inference_orig, gt_labels)
    ap_orig_nms, _ = metrics.average_precision_all_classes(
        dt_gt_match_orig_nms, inference_orig, gt_labels)
    ap_new, _ = metrics.average_precision_all_classes(
        dt_gt_match_new, inference_new, gt_labels)
    ap_new_nms, _ = metrics.average_precision_all_classes(
        dt_gt_match_new_nms, inference_new, gt_labels)

    map_orig = np.nanmean(ap_orig)
    map_orig_nms = np.nanmean(ap_orig_nms)
    map_knet = np.nanmean(ap_new)
    map_knet_nms = np.nanmean(ap_new_nms)

    logging.info('mAP original inference : %f' % map_orig)
    logging.info('mAP original inference (NMS) : %f' % map_orig_nms)
    logging.info('mAP knet inference : %f' % map_knet)
    logging.info('mAP knet inference (NMS) : %f' % map_knet_nms)

    return map_knet


def print_debug_info(nnms_model, sess, frame_data, outdir, fid):

    feed_dict = {nnms_model.dt_coords: frame_data[nnms.DT_COORDS],
                 nnms_model.dt_features: frame_data[nnms.DT_FEATURES],
                 nnms_model.dt_labels: frame_data[nnms.DT_LABELS],
                 nnms_model.dt_gt_iou: frame_data[nnms.DT_GT_IOU],
                 nnms_model.gt_labels: frame_data[nnms.GT_LABELS]}

    inference_orig = frame_data[nnms.DT_SCORES]
    inference, labels, loss = sess.run(
        [nnms_model.class_probs, nnms_model.labels, nnms_model.loss], feed_dict=feed_dict)
    logging.info("loss : %f" % loss)
    # logging.info("initial scores for pos values : %s"%frame_data[DT_FEATURES]
    # [np.where(frame_data[DT_LABELS][0:N_OBJECTS]>0)])

    logging.info("initial scores for matching bboxes : %s" %
          inference_orig[np.where(labels > 0)])
    logging.info("new knet scores for matching bboxes : %s" %
          inference[np.where(labels > 0)])
    num_gt = int(np.sum(frame_data[nnms.DT_LABELS]))
    num_pos_inf_orig = int(np.sum(inference_orig[:, 1:] > 0.0))
    num_pos_inf = int(np.sum(inference[:, 1:] > 0.5))
    logging.info(
        "frame num_gt : %d , num_pos_inf_orig : %d, num_pos_inf : %d" %
        (num_gt, num_pos_inf_orig, num_pos_inf))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img_dir = os.path.join(outdir, 'images')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # best IoU labels
    plt.imshow(frame_data[nnms.DT_LABELS][:, 1:])
    plt.title('Class labels (best IoU)')
    plt.ylabel('detections')
    plt.xlabel('class labels')
    plt.savefig(
        os.path.join(
            img_dir,
            'fid_' +
            str(fid) +
            '_labels_best_iou.png'))

    # mAP labels
    plt.imshow(labels[:, 1:])
    plt.title('Class labels (mAP)')
    plt.ylabel('detections')
    plt.xlabel('class labels')
    plt.savefig(
        os.path.join(
            img_dir,
            'fid_' +
            str(fid) +
            '_labels_map.png'))

    # per patch labels
    plt.imshow(frame_data[nnms.DT_LABELS_BASIC][:, 1:])
    plt.title('Class labels (per patch)')
    plt.ylabel('detections')
    plt.xlabel('class labels')
    plt.savefig(
        os.path.join(
            img_dir,
            'fid_' +
            str(fid) +
            '_labels_per_patch.png'))

    # fRCNN detections
    plt.imshow(softmax(inference_orig)[:, 1:])
    plt.title('Original fRCNN inference')
    plt.ylabel('detections')
    plt.xlabel('class scores')
    plt.savefig(
        os.path.join(
            img_dir,
            'fid_' +
            str(fid) +
            '_detections_orig.png'))
    plt.imshow(inference[:, 1:])

    # knet detections
    plt.title('knet inference')
    plt.ylabel('detections')
    plt.xlabel('class scores')
    plt.savefig(os.path.join(img_dir, 'fid_' + str(fid) + '_detections.png'))
    plt.close()

    return

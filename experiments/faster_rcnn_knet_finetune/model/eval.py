import logging

import joblib
import numpy as np
import os
from nms_network import model as nms_net
from tools import nms, metrics


def softmax(logits):
    n_classes = logits.shape[1]
    return np.exp(logits) / np.tile(np.sum(np.exp(logits),
                                           axis=1).reshape(-1, 1), [1, n_classes])


def eval_model(sess, nnms_model, frames_data,
               global_step, out_dir, full_eval=False,
               nms_thres=0.5, n_eval_frames=100, one_class=False, class_ix=None):

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
    losses = []

    n_total_frames = len(frames_data.keys())

    if full_eval:
        n_eval_frames = n_total_frames

    eval_data = {}

    # for fid in shuffle_samples(n_total_frames)[0:n_eval_frames]:
    for fid in range(0, n_eval_frames):

        eval_data[fid] = {}

        frame_data = frames_data[fid]

        gt_labels_all.append(frame_data[nms_net.GT_LABELS].reshape(-1, 1))

        feed_dict = {nnms_model.dt_coords: frame_data[nms_net.DT_COORDS],
                     nnms_model.dt_features: frame_data[nms_net.DT_FEATURES],
                     nnms_model.dt_probs_ini: frame_data[nms_net.DT_SCORES],
                     nnms_model.gt_coords: frame_data[nms_net.GT_COORDS],
                     nnms_model.gt_labels: frame_data[nms_net.GT_LABELS],
                     nnms_model.keep_prob: 1.0}

        inference_new, dt_dt_iou, loss, labels_tf = sess.run(
            [nnms_model.class_scores, nnms_model.iou_feature, nnms_model.loss,
             nnms_model.labels],
            feed_dict=feed_dict)

        # nms_labels, dt_dt_iou, ppf, sm, ioum, cnmsl = sess.run([nnms_model.nms_labels,
        #                                              nnms_model.iou_feature,
        #                                              nnms_model.pairwise_probs_features,
        #                                              nnms_model.suppression_map,
        #                                              nnms_model.iou_map,
        #                                              nnms_model.class_nms_labels], feed_dict=feed_dict)
        #

        if one_class:
            # expecting probability for class being already softmaxed
            inference_orig_all_classes = frame_data[nms_net.DT_SCORES_ORIGINAL]
            inference_original = inference_orig_all_classes[:, class_ix].reshape(-1, 1)
            inference_new_all_classes = np.copy(inference_orig_all_classes)
            inference_new_all_classes[:, class_ix] = np.squeeze(inference_new, axis=1)
        else:
            inference_new_all_classes = inference_new
            inference_orig_all_classes = softmax(frame_data[nms_net.DT_SCORES])
            inference_original = inference_orig_all_classes

        eval_data[fid]['dt_coords'] = frame_data[nms_net.DT_COORDS]
        eval_data[fid]['inference_orig'] = inference_orig_all_classes
        eval_data[fid]['inference_new'] = inference_new_all_classes

        labels_eval = metrics.match_dt_gt_all_classes(
                frame_data[nms_net.DT_GT_IOU],
                frame_data[nms_net.GT_LABELS],
                inference_new)

        # if np.sum(labels_eval != labels_tf) != 0:
        #     import ipdb; ipdb.set_trace()

        # else:
        #     if np.sum(labels_eval) > 0:
        #         logging.info('labels are not zero and the same')

        # if np.sum(labels_eval) > 0:
        #     import ipdb; ipdb.set_trace()

        losses.append(loss)
        inference_orig_all.append(inference_original)
        inference_new_all.append(inference_new)

        # import ipdb; ipdb.set_trace()

        dt_gt_match_orig.append(
            metrics.match_dt_gt_all_classes(
                frame_data[nms_net.DT_GT_IOU],
                frame_data[nms_net.GT_LABELS],
                inference_original))

        dt_gt_match_new.append(
            metrics.match_dt_gt_all_classes(
                frame_data[nms_net.DT_GT_IOU],
                frame_data[nms_net.GT_LABELS],
                inference_new))

        is_suppressed_orig = nms.nms_all_classes(
            dt_dt_iou, inference_original, iou_thr=nms_thres)

        is_suppressed_new = nms.nms_all_classes(
            dt_dt_iou, inference_new, iou_thr=nms_thres)

        dt_gt_match_orig_nms.append(
            metrics.match_dt_gt_all_classes(
                frame_data[nms_net.DT_GT_IOU],
                frame_data[nms_net.GT_LABELS],
                inference_original,
                dt_is_suppressed_info=is_suppressed_orig))

        dt_gt_match_new_nms.append(
            metrics.match_dt_gt_all_classes(
                frame_data[nms_net.DT_GT_IOU],
                frame_data[nms_net.GT_LABELS],
                inference_new,
                dt_is_suppressed_info=is_suppressed_new))

        # import ipdb; ipdb.set_trace()

    # if loss < 0:
    #     import ipdb; ipdb.set_trace()

    # import ipdb; ipdb.set_trace()

    gt_labels = np.vstack(gt_labels_all)
    inference_orig_all = np.vstack(inference_orig_all)
    inference_new_all = np.vstack(inference_new_all)

    dt_gt_match_orig = np.vstack(dt_gt_match_orig)
    dt_gt_match_new = np.vstack(dt_gt_match_new)
    dt_gt_match_orig_nms = np.vstack(dt_gt_match_orig_nms)
    dt_gt_match_new_nms = np.vstack(dt_gt_match_new_nms)

    if full_eval:
        eval_data_file = os.path.join(
            out_dir, 'eval_data_step' + str(global_step) + '.pkl')
        joblib.dump(eval_data, eval_data_file)

    ap_orig, _ = metrics.average_precision_all_classes(
        dt_gt_match_orig, inference_orig_all, gt_labels)
    ap_orig_nms, _ = metrics.average_precision_all_classes(
        dt_gt_match_orig_nms, inference_orig_all, gt_labels)
    ap_new, _ = metrics.average_precision_all_classes(
        dt_gt_match_new, inference_new_all, gt_labels)
    ap_new_nms, _ = metrics.average_precision_all_classes(
        dt_gt_match_new_nms, inference_new_all, gt_labels)

    map_orig = np.nanmean(ap_orig)
    map_orig_nms = np.nanmean(ap_orig_nms)
    map_knet = np.nanmean(ap_new)
    map_knet_nms = np.nanmean(ap_new_nms)

    logging.info('model loss : %f' % np.mean(losses))
    logging.info('mAP original inference : %f' % map_orig)
    logging.info('mAP original inference (NMS) : %f' % map_orig_nms)
    logging.info('mAP knet inference : %f' % map_knet)
    logging.info('mAP knet inference (NMS) : %f' % map_knet_nms)

    return map_knet, map_orig_nms


def print_debug_info(nnms_model, sess, frame_data, outdir, fid):

    feed_dict = {nnms_model.dt_coords: frame_data[nms_net.DT_COORDS],
                 nnms_model.dt_features: frame_data[nms_net.DT_FEATURES],
                 nnms_model.dt_probs: frame_data[nms_net.DT_SCORES],
                 nnms_model.gt_coords: frame_data[nms_net.GT_COORDS],
                 nnms_model.gt_labels: frame_data[nms_net.GT_LABELS],
                 nnms_model.keep_prob: 1.0}

    inference_orig = frame_data[nms_net.DT_SCORES]
    inference, labels, loss = sess.run(
        [nnms_model.class_scores, nnms_model.labels, nnms_model.loss], feed_dict=feed_dict)
    # logging.info("loss : %f" % loss)
    # logging.info("initial scores for pos values : %s"%frame_data[DT_FEATURES]
    # [np.where(frame_data[DT_LABELS][0:N_OBJECTS]>0)])

    logging.info("initial scores for matching bboxes : %s" %
          inference_orig[np.where(labels > 0)])
    logging.info("new knet scores for matching bboxes : %s" %
          inference[np.where(labels > 0)])
    num_gt = int(np.sum(frame_data[nms_net.DT_LABELS]))
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
    plt.imshow(frame_data[nms_net.DT_LABELS][:, 1:])
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
    plt.imshow(frame_data[nms_net.DT_LABELS_BASIC][:, 1:])
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

import logging

import joblib
import numpy as np
import os
from nms_network import model as nms_net
from tools import nms, metrics
from data import get_frame_data_fixed

def softmax(logits):
    n_classes = logits.shape[1]
    return np.exp(logits) / np.tile(np.sum(np.exp(logits),
                                           axis=1).reshape(-1, 1), [1, n_classes])


def eval_model(sess,
               nnms_model,
               detections_dir,
               labels_dir,
               eval_frames,
               n_bboxes,
               n_features,
               global_step,
               out_dir,
               nms_thres=0.5,
               gt_match_iou_thres=0.7,
               det_thres=0.0001):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dt_gt_match_orig = []
    dt_gt_match_new = []
    dt_gt_match_oracle = []
    # dt_gt_match_perfect_nms = []
    dt_gt_match_orig_nms = []
    dt_gt_match_new_nms = []

    # nms_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]

    # dt_gt_match_orig_nms_all = [[] for i in range(0, len(nms_thresholds))]

    inference_orig_all = []
    inference_new_all = []
    inference_oracle_all = []
    gt_labels_all = []
    losses = []


    eval_data_orig = []
    eval_data_orig_nms = []
    eval_data_new = []

    total_number_of_nms_fails = 0
    total_number_of_nms_supps = 0

    for fid in eval_frames:

        frame_data = get_frame_data_fixed(frame_id=fid,
                                    labels_dir=labels_dir,
                                    detections_dir=detections_dir,
                                    n_detections=n_bboxes,
                                    n_features=n_features)

        gt_labels_all.append(frame_data[nms_net.GT_LABELS].reshape(-1, 1))

        feed_dict = {nnms_model.dt_coords: frame_data['dt_coords'],
                     nnms_model.dt_features: frame_data['dt_features'],
                     nnms_model.dt_probs_ini: frame_data['dt_probs'],
                     nnms_model.gt_coords: frame_data['gt_coords'],
                     nnms_model.gt_labels: frame_data['gt_labels'],
                     nnms_model.keep_prob: 1.0}

        inference_orig = frame_data['dt_probs']
        inference_orig_all.append(inference_orig)

        inference_new, inference_oracle,  dt_dt_iou, loss = sess.run(
            [nnms_model.class_scores, nnms_model.det_labels, nnms_model.iou_feature, nnms_model.loss], feed_dict=feed_dict)

        losses.append(loss)
        inference_new_all.append(inference_new)
        inference_oracle_all.append(inference_oracle)

        dt_gt_match_orig.append(
            metrics.match_dt_gt_all_classes(
                frame_data[nms_net.DT_GT_IOU],
                frame_data[nms_net.GT_LABELS],
                inference_orig,
                iou_thr=gt_match_iou_thres))

        dt_gt_match_new.append(
            metrics.match_dt_gt_all_classes(
                frame_data[nms_net.DT_GT_IOU],
                frame_data[nms_net.GT_LABELS],
                inference_new,
                iou_thr=gt_match_iou_thres))

        dt_gt_match_oracle.append(
            metrics.match_dt_gt_all_classes(
                frame_data[nms_net.DT_GT_IOU],
                frame_data[nms_net.GT_LABELS],
                inference_oracle,
                iou_thr=gt_match_iou_thres))

        is_suppressed_orig = nms.nms_all_classes(
            dt_dt_iou, inference_orig, iou_thr=nms_thres)

        # is_suppressed_orig_all = nms.nms_all_classes_all_thresholds(dt_dt_iou,
        #                                                             inference_orig,
        #                                                             thrs=nms_thresholds)

        dt_coords_xywh = frame_data['dt_coords']
        dt_coords_xywh[:, 2] = dt_coords_xywh[:, 2] - dt_coords_xywh[:, 0]
        dt_coords_xywh[:, 3] = dt_coords_xywh[:, 3] - dt_coords_xywh[:, 1]
        frame_col = (fid+1) * np.ones([len(dt_coords_xywh), 1])

        data_orig = np.hstack([frame_col, dt_coords_xywh, inference_orig])
        data_orig_nms = data_orig[np.where(is_suppressed_orig == False)[0]]
        eval_data_orig_nms.append(data_orig_nms)
        data_orig = data_orig[np.where(inference_orig > det_thres)[0]]
        eval_data_orig.append(data_orig)
        data_new = np.hstack([frame_col, dt_coords_xywh, inference_new])
        # data_new = data_new[np.where(inference_new > det_thres)[0]]
        eval_data_new.append(data_new)

        # for i in range(0, is_suppressed_orig_all.shape[2]):
        #     dt_gt_match_orig_nms_all[i].append(metrics.match_dt_gt_all_classes(
        #             frame_data[nms_net.DT_GT_IOU],
        #         frame_data[nms_net.GT_LABELS],
        #         inference_orig,
        #         dt_is_suppressed_info=is_suppressed_orig_all[:, :, i],
        #         iou_thr=gt_match_iou_thres))

        is_suppressed_new = nms.nms_all_classes(
            dt_dt_iou, inference_new, iou_thr=nms_thres)

        # is_suppressed_oracle = nms.nms_all_classes(
        #     dt_dt_iou, inference_oracle, iou_thr=nms_thres)

        # wrongly_suppressed_hypotheses = np.zeros(n_bboxes)
        #
        # wrongly_suppressed_hypotheses[np.squeeze(np.logical_and(is_suppressed_oracle, inference_oracle))] = True

        # is_suppressed_ideal_case = nms.nms_per_class_with_oracle(dt_dt_iou.reshape([n_bboxes, n_bboxes]),
        #                                                         np.squeeze(inference_orig),
        #                                                         frame_data[nms_net.DT_GT_IOU],
        #                                                         iou_thr=nms_thres).reshape([n_bboxes, 1])

        # total_number_of_nms_fails += len(wrongly_suppressed_hypotheses[wrongly_suppressed_hypotheses == True])
        # total_number_of_nms_supps += len(is_suppressed_orig[is_suppressed_orig == True])


        # if np.sum(is_suppressed_ideal_case!=is_suppressed_orig) > 0:
        #     import ipdb; ipdb.set_trace()

        # dt_gt_match_perfect_nms.append(
        #     metrics.match_dt_gt_all_classes(
        #         frame_data[nms_net.DT_GT_IOU],
        #         frame_data[nms_net.GT_LABELS],
        #         inference_orig,
        #         iou_thr=gt_match_iou_thres,
        #         dt_is_suppressed_info=is_suppressed_ideal_case))

        dt_gt_match_orig_nms.append(
            metrics.match_dt_gt_all_classes(
                frame_data[nms_net.DT_GT_IOU],
                frame_data[nms_net.GT_LABELS],
                inference_orig,
                iou_thr=gt_match_iou_thres,
                dt_is_suppressed_info=is_suppressed_orig))

        dt_gt_match_new_nms.append(
            metrics.match_dt_gt_all_classes(
                frame_data[nms_net.DT_GT_IOU],
                frame_data[nms_net.GT_LABELS],
                inference_new,
                iou_thr=gt_match_iou_thres,
                dt_is_suppressed_info=is_suppressed_new))

    gt_labels = np.vstack(gt_labels_all)

    inference_orig = np.vstack(inference_orig_all)
    inference_new = np.vstack(inference_new_all)
    inference_oracle = np.vstack(inference_oracle_all)

    dt_gt_match_oracle = np.vstack(dt_gt_match_oracle)
    # dt_gt_match_perfect_nms = np.vstack(dt_gt_match_perfect_nms)
    dt_gt_match_orig = np.vstack(dt_gt_match_orig)
    dt_gt_match_new = np.vstack(dt_gt_match_new)
    dt_gt_match_orig_nms = np.vstack(dt_gt_match_orig_nms)

    # for i in range(0, len(dt_gt_match_orig_nms_all)):
    #     dt_gt_match_orig_nms_all[i] = np.vstack(dt_gt_match_orig_nms_all[i])

    dt_gt_match_new_nms = np.vstack(dt_gt_match_new_nms)

    ap_oracle, roc_oracle = metrics.average_precision_all_classes(
        dt_gt_match_oracle, inference_oracle, gt_labels)
    # ap_perfect_nms, _ = metrics.average_precision_all_classes(
    #     dt_gt_match_perfect_nms, inference_orig, gt_labels)

    ap_orig, _ = metrics.average_precision_all_classes(
        dt_gt_match_orig, inference_orig, gt_labels)
    ap_orig_nms, _ = metrics.average_precision_all_classes(
        dt_gt_match_orig_nms, inference_orig, gt_labels)
    ap_new, _ = metrics.average_precision_all_classes(
        dt_gt_match_new, inference_new, gt_labels)
    ap_new_nms, _ = metrics.average_precision_all_classes(
        dt_gt_match_new_nms, inference_new, gt_labels)

    # max_map_orig_nms = 0

    # for i in range(0, len(dt_gt_match_orig_nms_all)):
    #     ap_orig_nms, _ = metrics.average_precision_all_classes(
    #         dt_gt_match_orig_nms_all[i], inference_orig, gt_labels)
    #     map_orig_nms = np.nanmean(ap_orig_nms)
    #     if map_orig_nms > max_map_orig_nms:
    #         max_map_orig_nms = map_orig_nms
    #         max_nms_thres = nms_thresholds[i]

    map_oracle = np.nanmean(ap_oracle)
    # map_perfect_nms = np.nanmean(ap_perfect_nms)
    map_orig = np.nanmean(ap_orig)
    map_orig_nms = np.nanmean(ap_orig_nms)
    map_knet = np.nanmean(ap_new)
    map_knet_nms = np.nanmean(ap_new_nms)
    # mean_nms_fails = total_number_of_nms_fails / float(total_number_of_nms_supps)

    logging.info('loss : %f' % np.mean(losses))
    logging.info('mAP oracle : %f' % map_oracle)
    logging.info('mAP original inference : %f' % map_orig)
    logging.info('mAP original inference (NMS) : %f' % map_orig_nms)
    # logging.info('mAP original inference (best NMS IoU = %f) : %f' % (max_nms_thres, max_map_orig_nms))
    # logging.info('mAP original inference (perfect NMS) : %f' % map_perfect_nms)
    logging.info('mAP knet inference : %f' % map_knet)
    logging.info('mAP knet inference (NMS) : %f' % map_knet_nms)
    #logging.info('total number of NMS fails : %d, percent of all suppressions : %s' % (total_number_of_nms_fails,
    #                                                                             str(mean_nms_fails)))

    eval_data_orig = np.vstack(eval_data_orig)
    out_file_orig = os.path.join(out_dir, 'kitti_car_mscnn_nonms_' + str(global_step) + '.txt')
    np.savetxt(out_file_orig, eval_data_orig, fmt='%.6f', delimiter=',')

    eval_data_orig_nms = np.vstack(eval_data_orig_nms)
    out_file_orig_nms = os.path.join(out_dir, 'kitti_car_mscnn_nms_' + str(global_step) + '.txt')
    np.savetxt(out_file_orig_nms, eval_data_orig_nms, fmt='%.6f', delimiter=',')

    eval_data_new = np.vstack(eval_data_new)
    out_file_new = os.path.join(out_dir, 'kitti_car_mscnn_knet_' + str(global_step) + '.txt')
    np.savetxt(out_file_new, eval_data_new, fmt='%.6f', delimiter=',')

    return map_knet, map_orig_nms


def print_debug_info(nnms_model, sess, frame_data, outdir, fid):

    feed_dict = {nnms_model.dt_coords: frame_data[nms_net.DT_COORDS],
                 nnms_model.dt_features: frame_data[nms_net.DT_FEATURES],
                 nnms_model.gt_coords: frame_data[nms_net.GT_COORDS],
                 nnms_model.gt_labels: frame_data[nms_net.GT_LABELS],
                 nnms_model.keep_prob: 1.0}

    inference_orig = frame_data[nms_net.DT_SCORES]
    inference, labels, loss = sess.run(
        [nnms_model.class_scores, nnms_model.labels, nnms_model.loss], feed_dict=feed_dict)
    logging.info("loss : %f" % loss)
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

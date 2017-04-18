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

    gt_labels_all = []
    losses_opt = []
    losses_final = []

    n_total_frames = len(frames_data.keys())

    if full_eval:
        n_eval_frames = n_total_frames

    eval_data = {}

    for fid in range(0, n_eval_frames):

        eval_data[fid] = {}

        frame_data = frames_data[fid]

        if one_class:
            dt_probs_ini = frame_data[nms_net.DT_SCORES]
            gt_labels = frame_data[nms_net.GT_LABELS]
        else:
            dt_probs_ini = softmax(frame_data[nms_net.DT_SCORES])[:, 1:]
            gt_labels = frame_data[nms_net.GT_LABELS] - 1

        feed_dict = {nnms_model.dt_coords: frame_data[nms_net.DT_COORDS],
                     nnms_model.dt_features: frame_data[nms_net.DT_FEATURES],
                     nnms_model.dt_probs_ini: dt_probs_ini,
                     nnms_model.gt_coords: frame_data[nms_net.GT_COORDS],
                     nnms_model.gt_labels: gt_labels,
                     nnms_model.keep_prob: 1.0}

        inference_filtered, filter_inference, loss_opt, loss_final, dt_dt_iou = sess.run(
            [nnms_model.class_scores, nnms_model.sigmoid,
             nnms_model.loss, nnms_model.final_loss, nnms_model.iou_feature],
            feed_dict=feed_dict)

        nms_inference = 1 - filter_inference
        nms_labels_np = nms.nms_all_classes(dt_dt_iou, softmax(frame_data[nms_net.DT_SCORES])[:, 1:],
                                            iou_thr=nms_thres)
        nms_labels_np = nms_labels_np.astype('int')

        if one_class:
            # expecting probability for class being already softmaxed
            inference_orig_all_classes = frame_data[nms_net.DT_SCORES_ORIGINAL]
            # inference_original = inference_orig_all_classes[:, class_ix].reshape(-1, 1)
            inference_new_all_classes = np.copy(inference_orig_all_classes)
            inference_new_all_classes[:, class_ix] = np.squeeze(filter_inference, axis=1)
        else:
            import ipdb; ipdb.set_trace()
            inference_orig_all_classes = softmax(frame_data[nms_net.DT_SCORES])
            inference_new_all_classes = np.copy(inference_orig_all_classes)
            inference_new_all_classes[:, 1:] = nms_labels_np
            # inference_original = inference_orig_all_classes

        # import ipdb; ipdb.set_trace()

        eval_data[fid]['dt_coords'] = frame_data[nms_net.DT_COORDS]
        eval_data[fid]['inference_orig'] = inference_orig_all_classes
        eval_data[fid]['inference_new'] = inference_new_all_classes

        losses_opt.append(loss_opt)
        losses_final.append(loss_final)

    if full_eval:
        eval_data_file = os.path.join(
            out_dir, 'eval_data_step' + str(global_step) + '.pkl')
        joblib.dump(eval_data, eval_data_file)
        # import ipdb; ipdb.set_trace()

    mean_loss_opt = np.mean(losses_opt)
    mean_loss_fin = np.mean(losses_final)

    logging.info('optimization loss : %f' % mean_loss_opt)
    # logging.info('final loss : %f' % mean_loss_fin)

    return mean_loss_opt, mean_loss_fin


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

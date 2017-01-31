"""Knet on top of FasterRCNN inference
"""

import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim


from tf_layers import spatial, knet, misc, losses

DT_COORDS = 'dt_coords'
GT_COORDS = 'gt_coords'
GT_LABELS = 'gt_labels'
DT_LABELS = 'dt_labels'
DT_LABELS_BASIC = 'dt_labels_basic'
DT_FEATURES = 'dt_features'
DT_SCORES = 'dt_scores'
DT_INFERENCE = 'dt_inference'
DT_GT_IOU = 'dt_gt_iou'
DT_DT_IOU = 'dt_dt_iou'


class NeuralNMS:

    def __init__(self, n_detections, n_dt_features, n_classes,
                 n_kernels=100, knet_hlayer_size=100, fc_layer_size=100,
                 pos_weight=100, softmax_kernel=True, optimizer_step=0.0001, n_neg_examples=10,
                 use_iou_features=True, use_coords_features=True, use_object_features=True):

        # model parameters
        self._n_detections = n_detections
        self._n_dt_features = n_dt_features
        self._n_dt_coords = 4
        self._n_classes = n_classes
        self._n_kernels = n_kernels
        self._knet_hlayer_size = knet_hlayer_size
        self._fc_layer_size = fc_layer_size
        self._pos_weight = pos_weight
        self._softmax_kernel = softmax_kernel
        self._use_iou_features = use_iou_features
        self._use_coords_features = use_coords_features
        self._use_object_features = use_object_features
        self._optimizer_step = optimizer_step
        self._n_neg_examples = n_neg_examples

        # input ops
        self.dt_coords = tf.placeholder(
            tf.float32, shape=[
                self._n_detections, self._n_dt_coords], name=DT_COORDS)
        self.dt_features = tf.placeholder(
            tf.float32,
            shape=[
                self._n_detections,
                self._n_dt_features],
            name=DT_FEATURES)
        self.dt_labels = tf.placeholder(
            tf.float32, shape=[
                self._n_detections, self._n_classes], name=DT_LABELS)

        self.gt_labels = tf.placeholder(tf.float32, shape=None)

        self.dt_gt_iou = tf.placeholder(
            tf.float32, shape=[self._n_detections, None], name=DT_GT_IOU)

        #  inference ops
        self.pairwise_spatial_features = spatial.construct_pairwise_features_tf(
            self.dt_coords)

        self.spatial_features_list = []
        self._n_spatial_features = 0

        self.iou_feature = spatial.compute_pairwise_spatial_features_iou_tf(
                self.pairwise_spatial_features)
        if self._use_iou_features:
            self.spatial_features_list.append(self.iou_feature)
            self._n_spatial_features += 1

        self.pairwise_obj_features = spatial.construct_pairwise_features_tf(
                self.dt_features)
        if self._use_object_features:
            self.spatial_features_list.append(self.pairwise_obj_features)
            self._n_spatial_features += self._n_dt_features*2

        self.pairwise_coords_features = spatial.construct_pairwise_features_tf(
                self.dt_coords)
        if self._use_coords_features:
            self.spatial_features_list.append(self.pairwise_coords_features)
            self._n_spatial_features += self._n_dt_coords*2

        self.spatial_features = tf.concat(
            2, self.spatial_features_list)

        self.dt_new_features = knet.knet_layer(self.dt_features,
                                               self.spatial_features,
                                               n_kernels=self._n_kernels,
                                               n_objects=self._n_detections,
                                               n_pair_features=self._n_spatial_features,
                                               n_object_features=self._n_dt_features,
                                               softmax_kernel=self._softmax_kernel,
                                               hlayer_size=self._knet_hlayer_size)

        self.fc_layer = slim.layers.fully_connected(
            self.dt_new_features, self._fc_layer_size, activation_fn=tf.nn.relu)

        self.logits = slim.layers.fully_connected(
           self.fc_layer, self._n_classes, activation_fn=None)

        self.class_prob = tf.nn.sigmoid(self.logits)

        # loss ops

        # self.gt_per_labels = []
        self.class_labels = []

        for class_id in range(0, self._n_classes):
            gt_per_label = losses.construct_ground_truth_per_label_tf(self.dt_gt_iou, self.gt_labels, class_id)
            # self.gt_per_labels.append(gt_per_label)
            self.class_labels.append(losses.compute_match_gt_net_per_label_tf(self.class_prob,
                                                                                   gt_per_label,
                                                                                   class_id))

        self.labels = tf.pack(self.class_labels, axis=1)

        self.cross_entropy = tf.nn.weighted_cross_entropy_with_logits(self.logits,
                                                                      self.labels,
                                                                      pos_weight=self._pos_weight)

        hard_indices_tf = misc.data_subselection_hard_negative_tf(
            self.dt_labels, self.cross_entropy, n_neg_examples=self._n_neg_examples)

        self.loss_hard_tf = tf.gather(self.cross_entropy, hard_indices_tf)

        # loss_hard_tf_max = tf.reduce_max(loss_hard_tf, reduction_indices=[1])

        self.loss_final = tf.reduce_mean(self.cross_entropy)

        self.train_step = tf.train.AdamOptimizer(
            self._optimizer_step).minimize(
            self.loss_final)

        tf.summary.scalar('cross_entropy_loss', self.loss_final)

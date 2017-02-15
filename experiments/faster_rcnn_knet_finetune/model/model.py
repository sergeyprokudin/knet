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

    def _input_ops(self):

        dt_coords = tf.placeholder(
            tf.float32, shape=[
                self.n_detections, self.n_dt_coords], name=DT_COORDS)

        dt_features = tf.placeholder(tf.float32,
                                     shape=[
                                         self.n_detections,
                                         self.n_dt_features],
                                     name=DT_FEATURES)

        dt_labels = tf.placeholder(
                tf.float32, shape=[
                    self.n_detections, self.n_classes],
                name=DT_LABELS)

        gt_labels = tf.placeholder(tf.float32, shape=None)

        dt_gt_iou = tf.placeholder(
                tf.float32, shape=[self.n_detections, None], name=DT_GT_IOU)

        return dt_coords, dt_features, dt_labels, gt_labels, dt_gt_iou

    def _inference_ops(self):

        # we'll do one iteration of fc layers to prepare input for knet
        dt_features_ini = self._fc_layer_chain(input_tensor=self.dt_features,
                                               layer_size=self.fc_ini_layer_size,
                                               n_layers=self.fc_ini_layers_cnt,
                                               scope='fc_ini_layer')

        # we also reduce dimensionality of our features before passing it through knet
        dt_features_pre_knet = self._fc_layer_chain(input_tensor=self.dt_features,
                                                    layer_size=self.fc_pre_layer_size,
                                                    n_layers=self.fc_pre_layers_cnt,
                                                    scope='fc_pre_layer_knet')

        pairwise_spatial_features = spatial.construct_pairwise_features_tf(
            self.dt_coords)

        spatial_features_list = []
        n_spatial_features = 0

        iou_feature = spatial.compute_pairwise_spatial_features_iou_tf(pairwise_spatial_features)
        if self.use_iou_features:
            spatial_features_list.append(iou_feature)
            n_spatial_features += 1

        pairwise_obj_features = spatial.construct_pairwise_features_tf(dt_features_pre_knet)
        if self.use_object_features:
            spatial_features_list.append(pairwise_obj_features)
            n_spatial_features += self.fc_pre_layer_size * 2

        self.pairwise_coords_features = spatial.construct_pairwise_features_tf(
                self.dt_coords)
        if self.use_coords_features:
            spatial_features_list.append(self.pairwise_coords_features)
            n_spatial_features += self.n_dt_coords * 2

        spatial_features = tf.concat(2, spatial_features_list)

        # initial kernel iteration
        kernel = knet.knet_layer(pairwise_features=spatial_features,
                                 n_kernels=self.n_kernels,
                                 n_objects=self.n_detections,
                                 n_pair_features=n_spatial_features,
                                 softmax_kernel=self.softmax_kernel,
                                 hlayer_size=self.knet_hlayer_size)

        features_filtered = knet.apply_kernel(kernels=kernel,
                                              object_features=dt_features_ini,
                                              n_kernels=self.n_kernels,
                                              n_object_features=self.fc_ini_layer_size,
                                              n_objects=self.n_detections)

        updated_features = self._fc_layer_chain(input_tensor=features_filtered,
                                                layer_size=self.fc_apres_layer_size,
                                                n_layers=self.fc_apres_layers_cnt,
                                                scope='apres_fc_layers')

        features_iters_list = [updated_features]

        for i in range(1, self.n_kernel_iterations):

            if not self.reuse_kernels:
                kernel = knet.knet_layer(pairwise_features=spatial_features,
                                         n_kernels=self.n_kernels,
                                         n_objects=self.n_detections,
                                         n_pair_features=n_spatial_features,
                                         softmax_kernel=self.softmax_kernel,
                                         hlayer_size=self.knet_hlayer_size)

            features_filtered = knet.apply_kernel(kernels=kernel,
                                                  object_features=features_iters_list[-1],
                                                  n_kernels=self.n_kernels,
                                                  n_object_features=self.fc_apres_layer_size,
                                                  n_objects=self.n_detections)

            if self.reuse_apres_fc_layers:
                updated_features = self._fc_layer_chain(input_tensor=features_filtered,
                                                        layer_size=self.fc_apres_layer_size,
                                                        n_layers=self.fc_apres_layers_cnt,
                                                        scope='apres_fc_layers',
                                                        reuse=True)

            else:
                updated_features = self._fc_layer_chain(input_tensor=features_filtered,
                                                        layer_size=self.fc_apres_layer_size,
                                                        n_layers=self.fc_apres_layers_cnt,
                                                        scope='apres_fc_layers'+str(i))

            features_iters_list.append(updated_features)

        logits = slim.layers.fully_connected(
                updated_features, self.n_classes, activation_fn=None)

        sigmoid = tf.nn.sigmoid(logits)

        return iou_feature, logits, sigmoid

    def _loss_ops(self):

        class_labels = []

        for class_id in range(0, self.n_classes):
            gt_per_label = losses.construct_ground_truth_per_label_tf(self.dt_gt_iou, self.gt_labels, class_id)
            # self.gt_per_labels.append(gt_per_label)
            class_labels.append(losses.compute_match_gt_net_per_label_tf(self.class_probs,
                                                                         gt_per_label,
                                                                         class_id))

        labels = tf.pack(class_labels, axis=1)

        cross_entropy = tf.nn.weighted_cross_entropy_with_logits(self.logits,
                                                                 labels,
                                                                 pos_weight=self.pos_weight)

        # hard_indices_tf = misc.data_subselection_hard_negative_tf(
        #    self.dt_labels, cross_entropy, n_neg_examples=self.n_neg_examples)

        # loss_hard_tf = tf.gather(cross_entropy, hard_indices_tf)

        loss_final = tf.reduce_mean(cross_entropy)

        return labels,  loss_final

    def _fc_layer_chain(self,
                        input_tensor,
                        layer_size,
                        n_layers,
                        reuse=False,
                        scope=None):

        if n_layers == 0:
            return input_tensor
        elif n_layers == 1:
            fc_chain = slim.layers.fully_connected(input_tensor,
                                                   layer_size,
                                                   activation_fn=None,
                                                   reuse=reuse,
                                                   scope=scope+'/fc1')
            return fc_chain
        elif n_layers >= 2:
            fc_chain = slim.layers.fully_connected(input_tensor,
                                                   layer_size,
                                                   activation_fn=tf.nn.relu,
                                                   reuse=reuse,
                                                   scope=scope+'/fc1')

            for i in range(2, n_layers):
                fc_chain = slim.layers.fully_connected(fc_chain,
                                                       layer_size,
                                                       activation_fn=tf.nn.relu,
                                                       reuse=reuse,
                                                       scope=scope+'/fc'+str(i))

            fc_chain = slim.layers.fully_connected(fc_chain,
                                                   layer_size,
                                                   activation_fn=None,
                                                   reuse=reuse,
                                                   scope=scope+'/fc'+str(n_layers))
        return fc_chain

    def _train_ops(self):
        train_step = tf.train.AdamOptimizer(self.optimizer_step).minimize(self.loss)
        return train_step

    def _summary_ops(self):
        tf.summary.scalar('loss', self.loss)
        merged_summaries = tf.summary.merge_all()
        return merged_summaries

    def __init__(self, n_detections, n_dt_features, n_classes,
                 **kwargs):

        # model main parameters
        self.n_detections = n_detections
        self.n_dt_features = n_dt_features
        self.n_dt_coords = 4
        self.n_classes = n_classes

        # architecture params
        arch_args = kwargs.get('architecture', {})
        self.fc_ini_layer_size = arch_args.get('fc_ini_layer_size', 128)
        self.fc_ini_layers_cnt = arch_args.get('fc_ini_layers_cnt', 1)
        self.fc_pre_layer_size = arch_args.get('fc_pre_layer_size', 128)
        self.fc_pre_layers_cnt = arch_args.get('fc_pre_layers_cnt', 2)
        self.knet_hlayer_size = arch_args.get('knet_hlayer_size', 128)
        self.n_kernels = arch_args.get('n_kernels', 16)
        self.n_kernel_iterations = arch_args.get('fc_pre_layers_cnt', 1)
        self.reuse_kernels = arch_args.get('reuse_kernels', True)
        self.fc_apres_layer_size = arch_args.get('fc_apres_layer_size', 128)
        self.fc_apres_layers_cnt = arch_args.get('fc_apres_layers_cnt', 2)
        self.reuse_apres_fc_layers = arch_args.get('reuse_apres_fc_layers', True)

        # training procedure params
        train_args = kwargs.get('training', {})
        self.pos_weight = train_args.get('pos_weight', 1)
        self.softmax_kernel = train_args.get('softmax_kernel', True)
        self.use_iou_features = train_args.get('use_iou_features', True)
        self.use_coords_features = train_args.get('use_coords_features', True)
        self.use_object_features = train_args.get('use_object_features', True)
        self.optimizer_step = train_args.get('optimizer_step', 0.0001)
        self.n_neg_examples = train_args.get('n_neg_examples',  10)

        self.dt_coords, self.dt_features, self.dt_labels,\
            self.gt_labels, self.dt_gt_iou = self._input_ops()

        self.iou_feature, self.logits, self.class_probs = self._inference_ops()

        self.labels, self.loss = self._loss_ops()

        self.train_step = self._train_ops()

        self.merged_summaries = self._summary_ops()


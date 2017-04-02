"""Basic definition for KernelNetwork
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import losses
import knet
import spatial
from tf_layers import misc

DT_COORDS = 'dt_coords'
GT_COORDS = 'gt_coords'
GT_LABELS = 'gt_labels'
DT_LABELS = 'dt_labels'
DT_LABELS_BASIC = 'dt_labels_basic'
DT_FEATURES = 'dt_features'
DT_SCORES = 'dt_scores'
DT_SCORES_ORIGINAL = 'dt_scores_original'
DT_INFERENCE = 'dt_inference'
DT_GT_IOU = 'dt_gt_iou'
DT_DT_IOU = 'dt_dt_iou'


class NMSNetwork:

    VAR_SCOPE = 'nms_network'

    def __init__(self,
                 n_classes,
                 loss_type='best_iou',
                 input_ops=None,
                 gt_match_iou_thr=0.5,
                 class_ix=15,
                 **kwargs):

        # model main parameters
        self.n_dt_coords = 4
        self.n_classes = n_classes

        self.gt_match_iou_thr = gt_match_iou_thr
        self.class_ix = class_ix
        #self.n_bboxes = n_bboxes

        # architecture params
        arch_args = kwargs.get('architecture', {})
        self.fc_ini_layer_size = arch_args.get('fc_ini_layer_size', 1024)
        self.fc_ini_layers_cnt = arch_args.get('fc_ini_layers_cnt', 1)
        self.fc_pre_layer_size = arch_args.get('fc_pre_layer_size', 256)
        self.fc_pre_layers_cnt = arch_args.get('fc_pre_layers_cnt', 2)
        self.knet_hlayer_size = arch_args.get('knet_hlayer_size', 256)
        self.n_kernels = arch_args.get('n_kernels', 16)
        self.fc_apres_layer_size = arch_args.get('fc_apres_layer_size', 1024)
        self.class_scores_func = arch_args.get('class_scores_func', 'sigmoid')
        self.use_iou_features = arch_args.get('use_iou_features', True)
        self.use_coords_features = arch_args.get('use_coords_features', True)
        self.use_object_features = arch_args.get('use_object_features', True)

        # training procedure params
        train_args = kwargs.get('training', {})

        self.top_k_hypotheses = train_args.get('top_k_hypotheses', 20)
        self.optimizer_to_use = train_args.get('optimizer', 'Adam')
        self.nms_label_iou = train_args.get('nms_label_iou', 0.3)
        self.learning_rate = tf.placeholder(tf.float32)

        if input_ops is None:
            self.dt_coords, self.dt_features, self.dt_probs, \
                self.gt_labels, self.gt_coords, self.keep_prob = self._input_ops()
        else:
            self.dt_coords = input_ops['dt_coords']
            self.dt_features = input_ops['dt_features']
            self.dt_probs = input_ops['dt_probs']
            self.gt_labels = input_ops['gt_labels']
            self.gt_coords = input_ops['gt_coords']
            self.keep_prob = input_ops['keep_prob']

        self.n_dt_features = self.dt_features.get_shape().as_list()[1]

        self.n_bboxes = tf.shape(self.dt_features)[0]

        with tf.variable_scope(self.VAR_SCOPE):

            self.iou_feature, self.logits, self.sigmoid = self._inference_ops_top_k()
            self.class_scores = self.sigmoid #tf.multiply(self.sigmoid, self.dt_probs)
            self.det_labels, self.det_loss = self._detection_loss_ops()
            self.nms_labels, self.elementwise_nms_loss, self.nms_loss = self._nms_loss()
            self.nms_scores = self.sigmoid
            self.labels = self.det_labels
            self.loss = self.det_loss
            self.train_step = self._train_step(self.loss)
            self.nms_train_step = self._train_step(self.nms_loss)
            self.det_train_step = self._train_step(self.det_loss)
            # self.class_scores_nms = tf.multiply(self.nms_scores, self.dt_probs)
            self.merged_summaries = self._summary_ops()

        self.init_op = self._init_ops()

    # def switch_scoring(self, score_name):
    #     if score_name == 'detection':
    #         self.class_scores = self.sigmoid
    #     elif score_name == 'nms':
    #         self.class_scores = self.class_scores_nms
    #     return

    def _input_ops(self):

        dt_coords = tf.placeholder(
            tf.float32, shape=[
                None, self.n_dt_coords], name=DT_COORDS)

        dt_features = tf.placeholder(tf.float32,
                                     shape=[
                                         None,
                                         self.n_dt_features],
                                     name=DT_FEATURES)

        dt_probs = tf.placeholder(tf.float32,
                                     shape=[
                                         None,
                                         self.n_dt_classes],
                                     name=DT_FEATURES)

        gt_coords = tf.placeholder(tf.float32, shape=[None, 4])

        gt_labels = tf.placeholder(tf.float32, shape=None)

        keep_prob = tf.placeholder(tf.float32)

        return dt_coords, dt_features, dt_probs, gt_labels, gt_coords, keep_prob

    def _inference_ops(self):

        if self.n_classes == 1:
            highest_prob = tf.reduce_max(self.dt_probs, axis=1)
        else:
            # we are considering all classes, skip the backgorund class
            highest_prob = tf.reduce_max(self.dt_probs[:, 1:], axis=1)

        _, top_ix = tf.nn.top_k(highest_prob, k=self.top_k_hypotheses)

        pairwise_coords_features = spatial.construct_pairwise_features_tf(self.dt_coords)

        spatial_features_list = []
        n_pairwise_features = 0

        iou_feature = spatial.compute_pairwise_spatial_features_iou_tf(pairwise_coords_features)

        if self.use_iou_features:
            spatial_features_list.append(iou_feature)
            n_pairwise_features += 1

        pairwise_obj_features = spatial.construct_pairwise_features_tf(self.dt_features)

        if self.use_object_features:
            spatial_features_list.append(pairwise_obj_features)
            n_pairwise_features += self.dt_features.get_shape().as_list()[1] * 2
            score_diff_sign_feature = tf.sign(
                    pairwise_obj_features[:, :, 0:self.n_dt_features]-
                    pairwise_obj_features[:, :, self.n_dt_features:])
            score_diff_feature = pairwise_obj_features[:, :, 0:self.n_dt_features] -\
                                 pairwise_obj_features[:, :, self.n_dt_features:]
            spatial_features_list.append(score_diff_sign_feature)
            spatial_features_list.append(score_diff_feature)
            n_pairwise_features += self.dt_features.get_shape().as_list()[1] * 2
        pairwise_features = tf.concat(axis=2, values=spatial_features_list)

        diagonals = []
        for i in range(0, n_pairwise_features):
            d = tf.expand_dims(tf.diag(tf.diag_part(pairwise_features[:, :, i])), axis=2)
            diagonals.append(d)
        diag = tf.concat(axis=2, values=diagonals)

        pairwise_features = pairwise_features - diag

        self.pairwise_obj_features = pairwise_features

        kernel_features = self._kernel(pairwise_features,
                                       n_pairwise_features,
                                       hlayer_size=self.knet_hlayer_size,
                                       n_kernels=self.n_kernels)

        kernel_features_sigmoid = tf.nn.sigmoid(kernel_features)

        kernel_max = tf.reshape(tf.reduce_max(kernel_features_sigmoid, axis=1), [self.n_bboxes, self.n_kernels])

        kernel_sum = tf.reshape(tf.reduce_sum(kernel_features_sigmoid, axis=1), [self.n_bboxes, self.n_kernels])

        object_and_context_features = tf.concat(axis=1, values=[self.dt_features, kernel_max, kernel_sum])

        self.object_and_context_features = object_and_context_features

        fc1 = slim.layers.fully_connected(object_and_context_features,
                                          self.fc_apres_layer_size,
                                          activation_fn=tf.nn.relu)

        fc2 = slim.layers.fully_connected(fc1,
                                          self.fc_apres_layer_size,
                                          activation_fn=tf.nn.relu)

        fc2_drop = tf.nn.dropout(fc2, self.keep_prob)

        logits = slim.fully_connected(fc2_drop, self.n_classes, activation_fn=None)

        class_scores = tf.nn.sigmoid(logits)

        return iou_feature, logits, class_scores

    def _inference_ops_top_k(self):

        if self.n_classes == 1:
            highest_prob = tf.reduce_max(self.dt_probs, axis=1)
        else:
            # we are considering all classes, skipping the backgorund class
            highest_prob = tf.reduce_max(self.dt_probs[:, 1:], axis=1)

        _, top_ix = tf.nn.top_k(highest_prob, k=self.top_k_hypotheses)

        pairwise_coords_features = spatial.construct_pairwise_features_tf(self.dt_coords)

        pairwise_coords_features_top_k = spatial.construct_pairwise_features_tf(
            self.dt_coords, tf.gather(self.dt_coords, top_ix))

        spatial_features_list = []
        n_pairwise_features = 0

        iou_feature = spatial.compute_pairwise_spatial_features_iou_tf(pairwise_coords_features)
        iou_feature_top_k = spatial.compute_pairwise_spatial_features_iou_tf(pairwise_coords_features_top_k)

        if self.use_iou_features:
            spatial_features_list.append(iou_feature_top_k)
            n_pairwise_features += 1

        pairwise_obj_features_top_k = spatial.construct_pairwise_features_tf(self.dt_features,
                                                                             tf.gather(self.dt_features, top_ix))

        if self.use_object_features:
            spatial_features_list.append(pairwise_obj_features_top_k)
            n_pairwise_features += self.dt_features.get_shape().as_list()[1] * 2
            score_diff_sign_feature = tf.sign(
                    pairwise_obj_features_top_k[:, :, 0:self.n_dt_features] -
                    pairwise_obj_features_top_k[:, :, self.n_dt_features:])
            score_diff_feature = pairwise_obj_features_top_k[:, :, 0:self.n_dt_features] - \
                                 pairwise_obj_features_top_k[:, :, self.n_dt_features:]
            spatial_features_list.append(score_diff_sign_feature)
            spatial_features_list.append(score_diff_feature)
            n_pairwise_features += self.dt_features.get_shape().as_list()[1] * 2

        pairwise_features = tf.concat(axis=2, values=spatial_features_list)

        self.pairwise_obj_features = pairwise_features

        kernel_features = self._kernel(pairwise_features,
                                       n_pairwise_features,
                                       hlayer_size=self.knet_hlayer_size,
                                       n_kernels=self.n_kernels)

        kernel_features_sigmoid = tf.nn.sigmoid(kernel_features)

        kernel_max = tf.reshape(tf.reduce_max(kernel_features_sigmoid, axis=1), [self.n_bboxes, self.n_kernels])

        kernel_sum = tf.reshape(tf.reduce_sum(kernel_features_sigmoid, axis=1), [self.n_bboxes, self.n_kernels])

        object_and_context_features = tf.concat(axis=1, values=[self.dt_features, kernel_max, kernel_sum])

        self.object_and_context_features = object_and_context_features

        fc1 = slim.layers.fully_connected(object_and_context_features,
                                          self.fc_apres_layer_size,
                                          activation_fn=tf.nn.relu)

        fc2 = slim.layers.fully_connected(fc1,
                                          self.fc_apres_layer_size,
                                          activation_fn=tf.nn.relu)

        fc2_drop = tf.nn.dropout(fc2, self.keep_prob)

        logits = slim.fully_connected(fc2_drop, self.n_classes, activation_fn=None)

        if self.class_scores_func == 'softmax':
            class_scores = tf.nn.softmax(logits)
        else:
            class_scores = tf.nn.sigmoid(logits)

        return iou_feature, logits, class_scores

    def _kernel(self,
                pairwise_features,
                n_pair_features,
                hlayer_size,
                n_kernels=1):

        n_objects_1 = tf.shape(pairwise_features)[0]
        n_objects_2 = tf.shape(pairwise_features)[1]

        pairwise_features_reshaped = tf.reshape(
            pairwise_features, [
                1, n_objects_1, n_objects_2, n_pair_features])

        conv1 = slim.layers.conv2d(
            pairwise_features_reshaped,
            hlayer_size,
            [1, 1],
            activation_fn=tf.nn.relu)

        conv2 = slim.layers.conv2d(
            conv1,
            hlayer_size,
            [1, 1],
            activation_fn=tf.nn.relu)

        conv3 = slim.layers.conv2d(
            conv2,
            n_kernels,
            [1, 1],
            activation_fn=None)

        pairwise_potentials = tf.squeeze(conv3, axis=0)

        return pairwise_potentials

    def _detection_loss_ops(self):

        classes_labels_independent = []
        classes_labels_final = []

        pairwise_dt_gt_coords = spatial.construct_pairwise_features_tf(
            self.dt_coords, self.gt_coords)

        dt_gt_iou = tf.squeeze(spatial.compute_pairwise_spatial_features_iou_tf(pairwise_dt_gt_coords), 2)

        for class_id in range(0, self.n_classes):

            class_labels_independent = losses.construct_independent_labels(dt_gt_iou,
                                                                           self.gt_labels,
                                                                           class_id,
                                                                           iou_threshold=self.gt_match_iou_thr)

            classes_labels_independent.append(class_labels_independent)

            gt_per_label = losses.construct_ground_truth_per_label_tf(dt_gt_iou, self.gt_labels, class_id,
                                                                      iou_threshold=self.gt_match_iou_thr)

            classes_labels_final.append(losses.compute_match_gt_net_per_label_tf(self.sigmoid,
                                                                         gt_per_label,
                                                                         class_id))

        # self.classes_labels_independent = tf.stack(classes_labels_independent, axis=1)
        self.classes_labels_final = tf.stack(classes_labels_final, axis=1)
        # self.filter_labels = tf.to_float(tf.equal(self.classes_labels_independent, self.classes_labels_final))

        labels = self.classes_labels_final

        if self.class_scores_func == 'softmax':
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                           logits=self.logits)
        else:
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                           logits=self.logits)

        self.det_loss_elementwise = loss

        loss_final = tf.reduce_mean(loss)

        return labels, loss_final

    def _pairwise_nms_loss(self):

        suppression_map = self.pairwise_features[:, :, self.n_dt_features+1] > self.pairwise_features[:, :, 1]

        iou_map = self.pairwise_features[:, :, 0] > 0.5

        nms_pairwise_labels = tf.logical_and(suppression_map, iou_map)

        nms_pairwise_labels = tf.to_float(nms_pairwise_labels)

        pairwise_loss = tf.nn.weighted_cross_entropy_with_logits(self.pairwise_explain_logits,
                                                                 nms_pairwise_labels,
                                                                 pos_weight=self.pos_weight)

        # symmetry breaking constraint
        exclusive_explainig_constraint = 0.5 * tf.multiply(self.pairwise_explain_probs,
                                                            tf.transpose(self.pairwise_explain_probs))

        self.exclusive_explainig_constraint = exclusive_explainig_constraint

        loss_final = tf.reduce_mean(pairwise_loss)

        return suppression_map, iou_map, nms_pairwise_labels, pairwise_loss, loss_final

    def _nms_loss(self):

        nms_labels = []

        for class_id in range(0, self.n_classes):

            suppression_map = self.pairwise_obj_features[:, :,
                          self.class_ix + self.n_dt_features+1] > self.pairwise_obj_features[:, :, self.class_ix + 1]

            iou_map = self.pairwise_obj_features[:, :, 0] > self.nms_label_iou

            nms_pairwise_labels = tf.to_float(tf.logical_and(suppression_map, iou_map))

            nms_labels.append(1 - tf.reshape(tf.reduce_max(nms_pairwise_labels, axis=1), [self.n_bboxes, 1]))

        # suppression_map = self.pairwise_obj_features[:, :,
        #                   self.n_dt_features+1] > self.pairwise_obj_features[:, :, 1]

        self.nms_pairwise_labels = nms_pairwise_labels

        nms_labels = tf.stack(nms_labels, axis=1)

        if self.n_classes == 1:
            nms_labels = tf.squeeze(nms_labels, axis=2)

        elementwise_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=nms_labels,
                                                                   logits=self.logits)

        loss_final = tf.reduce_mean(elementwise_loss)

        return nms_labels, elementwise_loss, loss_final

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

    def _det_train_ops(self):
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.det_loss)
        return train_step

    def _train_step(self, loss):
        if self.optimizer_to_use == 'Adam':
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        elif self.optimizer_to_use == 'SGD':
            train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        else:
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return train_step

    def _summary_ops(self):
        tf.summary.scalar('loss', self.loss)
        merged_summaries = tf.summary.merge_all()
        return merged_summaries

    def _init_ops(self):

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.VAR_SCOPE)

        init_op = tf.variables_initializer(variables)

        return init_op


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
        self.n_kernel_iterations = arch_args.get('n_kernel_iterations', 1)
        self.reuse_kernels = arch_args.get('reuse_kernels', False)
        self.fc_apres_layer_size = arch_args.get('fc_apres_layer_size', 1024)
        self.fc_apres_layers_cnt = arch_args.get('fc_apres_layers_cnt', 2)
        self.reuse_apres_fc_layers = arch_args.get('reuse_apres_fc_layers', False)
        self.use_iou_features = arch_args.get('use_iou_features', True)
        self.use_coords_features = arch_args.get('use_coords_features', True)
        self.use_object_features = arch_args.get('use_object_features', True)

        # training procedure params
        train_args = kwargs.get('training', {})
        self.pos_weight = train_args.get('pos_weight', 1)
        self.softmax_kernel = train_args.get('softmax_kernel', True)

        self.n_neg_examples = train_args.get('n_neg_examples',  10)
        self.use_hinge_loss = train_args.get('use_hinge_loss',  False)

        #self.optimizer_step = train_args.get('optimizer_step', 0.001)
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

            self.iou_feature, self.logits, self.class_scores = self._inference_ops()
            self.det_labels, self.det_loss = self._detection_loss_ops()
            self.labels = self.det_labels
            self.loss = self.det_loss

            # self.nms_labels, self.elementwise_loss, self.nms_loss = self._nms_loss()
            # self.labels = self.nms_labels
            # self.loss = self.nms_loss
            # self.nms_prob = self.class_scores
            # self.class_scores = tf.multiply(self.dt_probs, 1-self.nms_prob)
            self.train_step = self._train_step(self.loss)
            # self.loss = self.pairwise_nms_loss + self.nms_loss
            # self.joint_train_step = self._train_step(self.loss)
            # self.pair_loss_train_step = self._train_step(self.pairwise_nms_loss)
            # self.nms_train_step = self._train_step(self.nms_loss)
            # self.det_train_step = self._det_train_ops()
            # self.train_step = self.det_train_step
            self.merged_summaries = self._summary_ops()

        self.init_op = self._init_ops()

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

        pairwise_coords_features = spatial.construct_pairwise_features_tf(
            self.dt_coords)

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
                    pairwise_obj_features[:, :, 0:self.n_dt_features]-pairwise_obj_features[:, :, self.n_dt_features:])
            score_diff_feature = pairwise_obj_features[:, :, 0:self.n_dt_features] - \
                                 pairwise_obj_features[:, :, self.n_dt_features:]
            spatial_features_list.append(score_diff_sign_feature)
            spatial_features_list.append(score_diff_feature)
            n_pairwise_features += self.dt_features.get_shape().as_list()[1] * 2

        pairwise_features = tf.concat(2, spatial_features_list)

        diagonals = []

        for i in range(0, n_pairwise_features):
            d = tf.expand_dims(tf.diag(tf.diag_part(pairwise_features[:, :, i])), axis=2)
            diagonals.append(d)

        diag = tf.concat(2, diagonals)

        pairwise_features = pairwise_features - diag

        self.pairwise_obj_features = pairwise_features

        kernel_features = self._kernel(pairwise_features,
                                       n_pairwise_features,
                                       hlayer_size=self.knet_hlayer_size,
                                       n_kernels=self.n_kernels)

        kernel_features_sigmoid = tf.nn.sigmoid(kernel_features)

        kernel_max = tf.reshape(tf.reduce_max(kernel_features_sigmoid, axis=1), [self.n_bboxes, self.n_kernels])

        kernel_sum = tf.reshape(tf.reduce_sum(kernel_features_sigmoid, axis=1), [self.n_bboxes, self.n_kernels])

        object_and_context_features = tf.concat(1, [self.dt_features, kernel_max, kernel_sum])

        self.object_and_context_features = object_and_context_features

        fc1 = slim.layers.fully_connected(object_and_context_features,
                                          self.fc_apres_layer_size,
                                          activation_fn=tf.nn.relu)

        fc2 = slim.layers.fully_connected(fc1,
                                          self.fc_apres_layer_size,
                                          activation_fn=tf.nn.relu)

        logits = slim.fully_connected(fc2, self.n_classes, activation_fn=None)

        class_scores = tf.nn.sigmoid(logits)

        return iou_feature, logits, class_scores

    def _kernel(self,
                pairwise_features,
                n_pair_features,
                hlayer_size,
                n_kernels=1):

        n_objects = tf.shape(pairwise_features)[0]

        pairwise_features_reshaped = tf.reshape(
            pairwise_features, [
                1, n_objects, n_objects, n_pair_features])

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

        class_labels = []

        pairwise_dt_gt_coords = spatial.construct_pairwise_features_tf(
            self.dt_coords, self.gt_coords)

        dt_gt_iou = tf.squeeze(spatial.compute_pairwise_spatial_features_iou_tf(pairwise_dt_gt_coords), 2)

        for class_id in range(0, self.n_classes):
            gt_per_label = losses.construct_ground_truth_per_label_tf(dt_gt_iou, self.gt_labels, class_id,
                                                                      iou_threshold=self.gt_match_iou_thr)
            # self.gt_per_labels.append(gt_per_label)
            class_labels.append(losses.compute_match_gt_net_per_label_tf(self.class_scores,
                                                                         gt_per_label,
                                                                         class_id))

        labels = tf.pack(class_labels, axis=1)

        if self.use_hinge_loss:
            loss = slim.losses.hinge_loss(self.logits, labels)
        else:
            loss = tf.nn.weighted_cross_entropy_with_logits(self.logits,
                                                            labels,
                                                            pos_weight=self.pos_weight)

        self.det_loss_elementwise = loss

        # hard_indices_tf = misc.data_subselection_hard_negative_tf(
        #    labels, loss, n_neg_examples=self.n_neg_examples)
        #
        # loss_hard_tf = tf.gather(loss, hard_indices_tf)

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

        suppression_map = self.pairwise_obj_features[:, :,
                          self.class_ix + self.n_dt_features+1] > self.pairwise_obj_features[:, :, self.class_ix + 1]

        # suppression_map = self.pairwise_obj_features[:, :,
        #                   self.n_dt_features+1] > self.pairwise_obj_features[:, :, 1]

        iou_map = self.pairwise_obj_features[:, :, 0] > 0.5

        nms_pairwise_labels = tf.logical_and(suppression_map, iou_map)

        nms_pairwise_labels = tf.to_float(nms_pairwise_labels)

        self.nms_pairwise_labels = nms_pairwise_labels

        nms_labels = tf.reshape(tf.reduce_max(nms_pairwise_labels, axis=1), [self.n_bboxes, 1])

        elementwise_loss = tf.nn.weighted_cross_entropy_with_logits(self.logits,
                                                                    nms_labels,
                                                                    pos_weight=self.pos_weight)

        # symmetry breaking constraint
        # exclusive_explainig_constraint = 0.5 * tf.multiply(self.pairwise_explain_probs,
        #                                                     tf.transpose(self.pairwise_explain_probs))

        # self.exclusive_explainig_constraint = exclusive_explainig_constraint

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
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return train_step

    def _summary_ops(self):
        tf.summary.scalar('loss', self.loss)
        merged_summaries = tf.summary.merge_all()
        return merged_summaries

    def _init_ops(self):

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.VAR_SCOPE)

        init_op = tf.initialize_variables(variables)

        return init_op


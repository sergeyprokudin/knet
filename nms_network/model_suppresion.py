"""Basic definition for KernelNetwork
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import losses
import knet
import spatial

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


class NMSNetwork:

    VAR_SCOPE = 'nms_network'

    def __init__(self,
                 n_classes,
                 n_bboxes,
                 input_ops=None,
                 **kwargs):

        # model main parameters
        self.n_dt_coords = 4
        self.n_bboxes = n_bboxes
        self.n_classes = n_classes

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
        self.optimizer_step = train_args.get('optimizer_step', 0.001)
        self.n_neg_examples = train_args.get('n_neg_examples',  10)
        self.use_hinge_loss = train_args.get('use_hinge_loss',  False)

        if input_ops is None:
            self.dt_coords, self.dt_features, self.nms_labels, \
                self.gt_labels, self.gt_coords, self.keep_prob = self._input_ops()
        else:
            self.dt_coords = input_ops['dt_coords']
            self.dt_features = input_ops['dt_features']
            self.gt_labels = input_ops['gt_labels']
            self.gt_coords = input_ops['gt_coords']
            self.nms_labels = input_ops['nms_labels']
            self.keep_prob = input_ops['keep_prob']

        self.n_dt_features = self.dt_features.get_shape().as_list()[1]

        with tf.variable_scope(self.VAR_SCOPE):

            self.spatial_features, self.iou_feature, self.logits, self.class_scores = self._inference_ops()

            self.nms_loss = self._nms_loss_ops()
            self.labels, self.class_loss = self._loss_ops()

            self.loss = self.class_loss

            self.train_step = self._train_ops()

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

        gt_coords = tf.placeholder(tf.float32, shape=[None, 4])

        gt_labels = tf.placeholder(tf.float32, shape=None)

        nms_labels = tf.placeholder(tf.float32, shape=None)

        keep_prob = tf.placeholder(tf.float32, shape=None)

        return dt_coords, dt_features, nms_labels, gt_labels, gt_coords, keep_prob

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

        pairwise_coords_features = spatial.construct_pairwise_features_tf(
            self.dt_coords)

        spatial_features_list = []
        n_spatial_features = 0

        iou_feature = spatial.compute_pairwise_spatial_features_iou_tf(pairwise_coords_features)

        if self.use_iou_features:
            spatial_features_list.append(iou_feature)
            n_spatial_features += 1

        pairwise_obj_features = spatial.construct_pairwise_features_tf(dt_features_pre_knet)
        if self.use_object_features:
            spatial_features_list.append(pairwise_obj_features)
            n_spatial_features += dt_features_pre_knet.get_shape().as_list()[1] * 2

        if self.use_coords_features:
            spatial_features_list.append(pairwise_coords_features)
            n_spatial_features += self.n_dt_coords * 2

        spatial_features = tf.concat(axis=2, values=spatial_features_list)

        diagonals = []

        for i in range(0, n_spatial_features):
            d = tf.expand_dims(tf.diag(tf.diag_part(spatial_features[:, :, i])), axis=2)
            diagonals.append(d)

        diag = tf.concat(axis=2, values=diagonals)

        spatial_features = spatial_features - diag

        spatial_features = tf.reshape(spatial_features, [self.n_bboxes, n_spatial_features*self.n_bboxes])

        input_features = tf.concat(axis=1, values=[self.dt_features, spatial_features])

        fc_chain = self._fc_layer_chain(input_tensor=input_features,
                                        layer_size=512,
                                        n_layers=3,
                                        scope='fc_spatial')

        fc_chain = tf.nn.dropout(fc_chain, self.keep_prob)

        logits = slim.layers.fully_connected(fc_chain, self.n_classes, activation_fn=None)

        class_scores = tf.nn.sigmoid(logits)

        return input_features, iou_feature, logits, class_scores

    def _loss_ops(self):

        class_labels = []

        pairwise_dt_gt_coords = spatial.construct_pairwise_features_tf(
            self.dt_coords, self.gt_coords)

        dt_gt_iou = tf.squeeze(spatial.compute_pairwise_spatial_features_iou_tf(pairwise_dt_gt_coords), 2)

        for class_id in range(0, self.n_classes):
            gt_per_label = losses.construct_ground_truth_per_label_tf(dt_gt_iou, self.gt_labels, class_id)
            # self.gt_per_labels.append(gt_per_label)
            class_labels.append(losses.compute_match_gt_net_per_label_tf(self.class_scores,
                                                                         gt_per_label,
                                                                         class_id))

        labels = tf.stack(class_labels, axis=1)

        if self.use_hinge_loss:
            loss = slim.losses.hinge_loss(self.logits, labels)
        else:
            loss = tf.nn.weighted_cross_entropy_with_logits(self.logits,
                                                            labels,
                                                            pos_weight=self.pos_weight)

        # hard_indices_tf = misc.data_subselection_hard_negative_tf(
        #    self.dt_labels, cross_entropy, n_neg_examples=self.n_neg_examples)
        #
        # loss_hard_tf = tf.gather(cross_entropy, hard_indices_tf)

        loss_final = tf.reduce_mean(loss)

        return labels,  loss_final

    def _nms_loss_ops(self):

        if self.use_hinge_loss:
            loss = slim.losses.hinge_loss(self.logits, self.nms_labels)
        else:
            loss = tf.nn.weighted_cross_entropy_with_logits(self.logits,
                                                            self.nms_labels,
                                                            pos_weight=self.pos_weight)

        loss_final = tf.reduce_mean(loss)

        return loss_final

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

    def _init_ops(self):

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.VAR_SCOPE)

        init_op = tf.variables_initializer(variables)

        return init_op


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
                 **kwargs):

        # model main parameters
        self.n_dt_coords = 4
        self.n_classes = n_classes
        self.gt_match_iou_thr = gt_match_iou_thr
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

            if loss_type == 'best_iou':
                self.iou_feature, self.logits, self.class_scores = self._old_inference_ops()
                self.det_labels, self.det_loss = self._detection_loss_ops()
                self.labels = self.det_labels
                self.loss = self.det_loss
                self.det_train_step = self._det_train_ops()
                self.train_step = self.det_train_step

            elif loss_type == 'nms_pairwise_labels':

                self.iou_feature, self.pairwise_features,\
                self.pairwise_explain_logits, self.pairwise_explain_probs, \
                self.logits, self.not_explained_prob, self.class_scores = self._inference_ops()

                #self.iou_feature, self.pairwise_features, self.pairwise_explain_logits,
                #self.pairwise_explain_probs = self._alternative_inference_ops4()

                #self.logits = self.class_scores

                self.suppression_map, self.iou_map, self.nms_pairwise_labels, \
                self.elementwise_loss, self.nms_loss = self._pairwise_nms_loss()
                self.det_labels, self.det_loss = self._detection_loss_ops()
                self.labels = self.det_labels
                self.loss = self.det_loss + self.nms_loss
                self.nms_train_step = self._train_step()
                self.det_train_step = self._det_train_ops()
                self.train_step = self.nms_train_step

            elif loss_type == 'nms_loss':

                self.iou_feature, self.logits, self.class_scores = self._inference_ops()

                # self.iou_feature, self.pairwise_features, self.pairwise_explain_logits,
                # self.pairwise_explain_probs = self._alternative_inference_ops4()
                # self.logits = self.class_scores
                #
                # self.suppression_map, self.iou_map, self.nms_pairwise_labels, \
                # self.elementwise_loss, self.nms_loss = self._nms_loss()

                # _, _, _, _, self.pairwise_nms_loss = self._pairwise_nms_loss()

                self.det_labels, self.det_loss = self._detection_loss_ops()
                self.labels = self.det_labels
                self.loss = self.det_loss
                # self.loss = self.pairwise_nms_loss + self.nms_loss
                # self.joint_train_step = self._train_step(self.loss)
                # self.pair_loss_train_step = self._train_step(self.pairwise_nms_loss)
                # self.nms_train_step = self._train_step(self.nms_loss)
                self.det_train_step = self._det_train_ops()
                self.train_step = self.det_train_step

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

    def _old_inference_ops(self):

        # we'll do one iteration of fc layers to prepare input for knet
        dt_features_ini = self._fc_layer_chain(input_tensor=self.dt_features,
                                               layer_size=self.fc_ini_layer_size,
                                               n_layers=self.fc_ini_layers_cnt,
                                               scope='fc_ini_layer')

        # we also reduce dimensionality of our features before passing it through knet
        # dt_features_pre_knet = self._fc_layer_chain(input_tensor=self.dt_features,
        #                                             layer_size=self.fc_pre_layer_size,
        #                                             n_layers=self.fc_pre_layers_cnt,
        #                                             scope='fc_pre_layer_knet')

        pairwise_spatial_features = spatial.construct_pairwise_features_tf(
            self.dt_coords)

        spatial_features_list = []
        n_spatial_features = 0

        iou_feature = spatial.compute_pairwise_spatial_features_iou_tf(pairwise_spatial_features)
        if self.use_iou_features:
            spatial_features_list.append(iou_feature)
            n_spatial_features += 1

        pairwise_obj_features = spatial.construct_pairwise_features_tf(self.dt_features)
        if self.use_object_features:
            spatial_features_list.append(pairwise_obj_features)
            n_spatial_features += self.n_dt_features * 2
            score_diff_sign_feature = tf.sign(
                    pairwise_obj_features[:, :, 0:self.n_dt_features]-pairwise_obj_features[:, :, self.n_dt_features:])
            score_diff_feature = pairwise_obj_features[:, :, 0:self.n_dt_features] - \
                                 pairwise_obj_features[:, :, self.n_dt_features:]
            spatial_features_list.append(score_diff_sign_feature)
            spatial_features_list.append(score_diff_feature)
            n_spatial_features += self.dt_features.get_shape().as_list()[1] * 2


        self.pairwise_coords_features = spatial.construct_pairwise_features_tf(
                self.dt_coords)
        if self.use_coords_features:
            spatial_features_list.append(self.pairwise_coords_features)
            n_spatial_features += self.n_dt_coords * 2

        spatial_features = tf.concat(2, spatial_features_list)

        diagonals = []

        for i in range(0, n_spatial_features):
            d = tf.expand_dims(tf.diag(tf.diag_part(spatial_features[:, :, i])), axis=2)
            diagonals.append(d)

        diag = tf.concat(2, diagonals)
        spatial_features = spatial_features - diag

        # initial kernel iteration
        kernel = knet.knet_layer(pairwise_features=spatial_features,
                                 n_kernels=self.n_kernels,
                                 n_pair_features=n_spatial_features,
                                 softmax_kernel=self.softmax_kernel,
                                 hlayer_size=self.knet_hlayer_size)

        features_filtered = knet.apply_kernel(kernels=kernel,
                                              object_features=dt_features_ini,
                                              n_kernels=self.n_kernels,
                                              n_object_features=self.fc_ini_layer_size)

        updated_features = self._fc_layer_chain(input_tensor=features_filtered,
                                                layer_size=self.fc_apres_layer_size,
                                                n_layers=self.fc_apres_layers_cnt,
                                                scope='apres_fc_layers')

        features_iters_list = [updated_features]

        for i in range(1, self.n_kernel_iterations):

            spatial_features = spatial.construct_pairwise_features_tf(updated_features)
            n_spatial_features = self.fc_apres_layer_size*2

            diagonals = []

            for i in range(0, n_spatial_features):
                d = tf.expand_dims(tf.diag(tf.diag_part(spatial_features[:, :, i])), axis=2)
                diagonals.append(d)

            diag = tf.concat(2, diagonals)

            spatial_features = spatial_features - diag

            if not self.reuse_kernels:
                kernel = knet.knet_layer(pairwise_features=spatial_features,
                                         n_kernels=self.n_kernels,
                                         n_pair_features=n_spatial_features,
                                         softmax_kernel=self.softmax_kernel,
                                         hlayer_size=self.knet_hlayer_size)

            features_filtered = knet.apply_kernel(kernels=kernel,
                                                  object_features=features_iters_list[-1],
                                                  n_kernels=self.n_kernels,
                                                  n_object_features=self.fc_apres_layer_size)

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

        updated_features = tf.nn.dropout(updated_features, self.keep_prob)

        logits = slim.layers.fully_connected(
                updated_features, self.n_classes, activation_fn=None)

        if self.use_hinge_loss:
            class_scores = logits
        else:
            class_scores = tf.nn.sigmoid(logits)

        return iou_feature, logits, class_scores

    def _alternative_inference_ops(self):
        '''
        Inference based on joint analysis of all present hypothese
        Returns
        -------

        '''
        pairwise_coords_features = spatial.construct_pairwise_features_tf(
            self.dt_coords)

        spatial_features_list = []
        n_spatial_features = 0

        iou_feature = spatial.compute_pairwise_spatial_features_iou_tf(pairwise_coords_features)
        if self.use_iou_features:
            spatial_features_list.append(iou_feature)
            n_spatial_features += 1

        pairwise_obj_features = spatial.construct_pairwise_features_tf(self.dt_features)
        if self.use_object_features:
            spatial_features_list.append(pairwise_obj_features)
            n_spatial_features += self.dt_features.get_shape().as_list()[1] * 2

        spatial_features = tf.concat(2, spatial_features_list)

        diagonals = []

        for i in range(0, n_spatial_features):
            d = tf.expand_dims(tf.diag(tf.diag_part(spatial_features[:, :, i])), axis=2)
            diagonals.append(d)

        diag = tf.concat(2, diagonals)

        spatial_features = spatial_features - diag

        spatial_features = tf.reshape(spatial_features, [self.n_bboxes, n_spatial_features*self.n_bboxes])

        net_input_features = tf.concat(1, [self.dt_features, spatial_features])

        fc_chain = self._fc_layer_chain(input_tensor=net_input_features,
                                        layer_size=512,
                                        n_layers=3,
                                        scope='fc_spatial')

        fc_chain = tf.nn.dropout(fc_chain, self.keep_prob)

        logits = slim.layers.fully_connected(fc_chain, self.n_classes, activation_fn=None)

        class_scores = tf.nn.sigmoid(logits)

        return net_input_features, iou_feature, logits, class_scores

    def _alternative_inference_ops2(self):

        pairwise_coords_features = spatial.construct_pairwise_features_tf(
            self.dt_coords)

        spatial_features_list = []
        n_spatial_features = 0

        iou_feature = spatial.compute_pairwise_spatial_features_iou_tf(pairwise_coords_features)
        if self.use_iou_features:
            spatial_features_list.append(iou_feature)
            n_spatial_features += 1

        pairwise_obj_features = spatial.construct_pairwise_features_tf(self.dt_features)
        if self.use_object_features:
            spatial_features_list.append(pairwise_obj_features)
            n_spatial_features += self.dt_features.get_shape().as_list()[1] * 2

        spatial_features = tf.concat(2, spatial_features_list)

        diagonals = []

        for i in range(0, n_spatial_features):
            d = tf.expand_dims(tf.diag(tf.diag_part(spatial_features[:, :, i])), axis=2)
            diagonals.append(d)

        diag = tf.concat(2, diagonals)

        spatial_features = spatial_features - diag

        self.spatial_features_square = spatial_features

        self.suppression_map = spatial_features[:, :, 2] > spatial_features[:, :, 1]
        self.iou_map = spatial_features[:, :, 0] > 0.5

        self.nms_pairwise_labels = tf.reshape(tf.logical_and(self.suppression_map, self.iou_map), [self.n_bboxes, self.n_bboxes])

        self.nms_pairwise_labels = tf.to_float(self.nms_pairwise_labels)

        self.kernel_matrix = self._kernel(spatial_features, n_spatial_features, hlayer_size=512)

        self.kernel_matrix_sigmoid = tf.nn.sigmoid(self.kernel_matrix)

        # self.kernel_matrix = tf.reshape(self.kernel_matrix, [self.n_bboxes, self.n_bboxes])

        # self_potential = tf.zeros(shape=[self.n_bboxes, 1])

        # self.explaining_matrix_logits = tf.concat(1, [self_potential, self.kernel_matrix])

        self.initial_prob = tf.ones(shape=[self.n_bboxes, 1])
        self.max_explained_prob = tf.reshape(tf.reduce_max(self.kernel_matrix_sigmoid, axis=1), [self.n_bboxes, 1])

        self.conditional_prob = tf.sub(self.initial_prob, self.max_explained_prob)

        # self.explaining_matrix = tf.nn.softmax(self.explaining_matrix_logits)

        # import ipdb; ipdb.set_trace()

        # self.dt_features_tiled = tf.reshape(tf.tile(tf.transpose(self.dt_features), [self.n_bboxes, 1]), [self.n_bboxes, self.n_bboxes])

        # self.dt_features_tiled = tf.concat(1, [self.dt_features, self.dt_features_tiled])

        # self.self_explained = tf.reshape(self.explaining_matrix[:, 0], [self.n_bboxes, 1])

        class_scores = tf.mul(self.conditional_prob, self.dt_features)

        logits = class_scores

        net_input_features = tf.concat(1, [self.dt_features, self.conditional_prob])
        #
        # fc_chain = self._fc_layer_chain(net_input_features,
        #                               layer_size=32,
        #                               n_layers=2,
        #                               scope='fc_inference')
        #
        # logits = slim.layers.fully_connected(fc_chain, self.n_classes, activation_fn=None)
        #
        # class_scores = tf.nn.sigmoid(logits)

        #net_input_features = self.dt_features

        # self.rescaled_features = tf.reshape(tf.reduce_max(tf.mul(self.kernel_matrix, self.dt_features_tiled), axis=1), [self.n_bboxes, 1])

        # spatial_features = tf.reshape(self.kernel_matrix, [self.n_bboxes, -1])

        # import ipdb; ipdb.set_trace()

        # sum pooling of a features
        # spatial_features = tf.reduce_sum(self.kernel_matrix, axis=1)

        # spatial_features = tf.reshape(spatial_features, [self.n_bboxes, self.n_kernels])

        # spatial_features = tf.reshape(spatial_features, [self.n_bboxes, n_spatial_features*self.n_bboxes])

        # net_input_features = tf.concat(1, [self.dt_features, self.rescaled_features])

        # fc_chain = self._fc_layer_chain(input_tensor=net_input_features,
        #                                   layer_size=128,
        #                                   n_layers=3,
        #                                   scope='fc_spatial')

        #fc_chain = tf.nn.dropout(fc_chain, self.keep_prob)

        # logits = slim.layers.fully_connected(fc_chain, self.n_classes, activation_fn=None)

        # class_scores = tf.nn.sigmoid(logits)

        return net_input_features, iou_feature, logits, class_scores

    def _alternative_inference_ops3(self):

        pairwise_coords_features = spatial.construct_pairwise_features_tf(
            self.dt_coords)

        spatial_features_list = []
        n_pairwise_features = 0

        iou_feature = spatial.compute_pairwise_spatial_features_iou_tf(pairwise_coords_features)
        if self.use_iou_features:
            spatial_features_list.append(iou_feature)
            n_pairwise_features += 1

        pairwise_obj_features = spatial.construct_pairwise_features_tf(self.dt_features)

        self.pair_initial = pairwise_obj_features

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

        pairwise_explain_logits = self._kernel(pairwise_features, n_pairwise_features, hlayer_size=512)

        pairwise_explain_probs = tf.nn.sigmoid(pairwise_explain_logits)

        initial_prob = tf.ones(shape=[self.n_bboxes, 1])

        max_explained_prob = tf.reshape(tf.reduce_max(pairwise_explain_probs, axis=1), [self.n_bboxes, 1])

        not_explained_prob = tf.sub(initial_prob, max_explained_prob)

        class_scores = tf.mul(not_explained_prob, self.dt_probs)

        return iou_feature, pairwise_features, pairwise_explain_logits, pairwise_explain_probs, not_explained_prob, class_scores

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

        self.pair_initial = pairwise_obj_features

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

        kernel_features = self._kernel(pairwise_features,
                                       n_pairwise_features,
                                       hlayer_size=self.knet_hlayer_size,
                                       n_kernels=self.n_kernels)

        kernel_features_sigmoid = tf.nn.sigmoid(kernel_features)

        kernel_max = tf.reshape(tf.reduce_max(kernel_features_sigmoid, axis=1), [self.n_bboxes, self.n_kernels])

        kernel_sum = tf.reshape(tf.reduce_sum(kernel_features_sigmoid, axis=1), [self.n_bboxes, self.n_kernels])

        object_and_context_features = tf.concat(1, [self.dt_features, kernel_max])

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

        suppression_map = self.pairwise_features[:, :, self.n_dt_features+1] > self.pairwise_features[:, :, 1]

        iou_map = self.pairwise_features[:, :, 0] > 0.5

        nms_pairwise_labels = tf.logical_and(suppression_map, iou_map)

        nms_pairwise_labels = tf.to_float(nms_pairwise_labels)

        nms_labels = tf.reshape(tf.reduce_max(nms_pairwise_labels, axis=1), [self.n_bboxes, 1])

        self.nms_labels = 1-nms_labels

        nms_loss = tf.nn.weighted_cross_entropy_with_logits(self.logits,
                                                            nms_labels,
                                                            pos_weight=self.pos_weight)

        # symmetry breaking constraint
        # exclusive_explainig_constraint = 0.5 * tf.multiply(self.pairwise_explain_probs,
        #                                                     tf.transpose(self.pairwise_explain_probs))

        #self.exclusive_explainig_constraint = exclusive_explainig_constraint

        loss_final = tf.reduce_mean(nms_loss)

        return suppression_map, iou_map, nms_pairwise_labels, nms_loss, loss_final

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


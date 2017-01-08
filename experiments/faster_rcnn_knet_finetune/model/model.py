"""Knet on top of FasterRCNN inference
"""

import numpy as np
import tensorflow as tf

DT_COORDS = 'dt_coords'
GT_COORDS = 'gt_coords'
GT_LABELS = 'gt_labels'
DT_LABELS = 'dt_labels'
DT_LABELS_BASIC = 'dt_labels_basic'
DT_FEATURES = 'dt_features'
DT_INFERENCE = 'dt_inference'
DT_GT_IOU = 'dt_gt_iou'
DT_DT_IOU = 'dt_dt_iou'

class NeuralNMS:

    def __init__(self, n_kernels=100):
        self._n_kernels = 10
        self._knet_hlayer_size = 10


    def input_ops(self):
        input_ops[DT_COORDS] = tf.placeholder(
            tf.float32, shape=[
                N_OBJECTS, N_DT_COORDS], name=DT_COORDS)
        input_ops[DT_FEATURES] = tf.placeholder(
            tf.float32,
            shape=[
                N_OBJECTS,
                N_DT_FEATURES],
            name=DT_FEATURES)
        input_ops[DT_LABELS] = tf.placeholder(
            tf.float32, shape=[
                N_OBJECTS, N_CLASSES], name=DT_LABELS)
        return input_ops

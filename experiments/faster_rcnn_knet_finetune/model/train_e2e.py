import os
import numpy as np
import tensorflow as tf

def main():

    pretrained_path = '/Users/sergey/knet/experiments/faster_rcnn_knet_finetune/data/pretrained_models/'
    roi_pool_op = '/Users/sergey/Faster-RCNN_TF/lib/roi_pooling_layer/roi_pooling.so'
    roi_op = tf.load_op_library(roi_pool_op)
    model_graph = os.path.join(pretrained_path, 'VGGnet_fast_rcnn_iter_70000.ckpt.meta')
    model_ckpt = os.path.join(pretrained_path, 'VGGnet_fast_rcnn_iter_70000.ckpt')
    faster_rcnn = tf.train.import_meta_graph(model_graph)
    graph = tf.get_default_graph()
    input_image_ph = graph.get_operation_by_name('Placeholder').outputs
    im_info_ph = graph.get_operation_by_name('Placeholder_1').outputs
    gt_boxes_ph = graph.get_operation_by_name('Placeholder_2').outputs
    keep_prob_ph = graph.get_operation_by_name('Placeholder_3').outputs

    graph_names = [n.name for n in tf.get_default_graph().as_graph_def().node]

    fc_out = graph.get_operation_by_name('fc7/fc7').outputs
    cls_score_out = graph.get_operation_by_name('cls_score/cls_score').outputs

    with tf.Session() as sess:
        faster_rcnn.restore(sess, model_ckpt)
        print("!!!")

    import ipdb; ipdb.set_trace()
    return


if __name__=='__main__':
    main()

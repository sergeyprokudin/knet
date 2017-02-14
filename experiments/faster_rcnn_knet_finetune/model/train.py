"""Training routine for knet trained on top of FasterRCNN inference

    data_dir - directory containing train data :
                dt_coords.pkl - detections' bounding boxes coordinates in format [frame_id, x_min, y_min, width, height]
                dt_features.pkl - detections' bounding boxes features
                gt_coords.pkl - ground truth bounding boxes coordinates in format [frame_id, x_min, y_min, width, height, class_id]

"""

import numpy as np
import tensorflow as tf

import joblib
import pandas as pd
from functools import partial
import os
import sys
import shutil
import ntpath
import binascii
import subprocess

import gflags
import yaml
import logging
from google.apputils import app
from timeit import default_timer as timer

from tools import bbox_utils, nms, metrics
import model as nnms
import eval

gflags.DEFINE_string('data_dir', None, 'directory containing train data')
gflags.DEFINE_string('log_dir', None, 'directory to save logs and trained models')
gflags.DEFINE_string('config_path', None, 'config with main model params')

FLAGS = gflags.FLAGS

N_DT_COORDS = 4
N_FC_FEATURES_FULL = 4096
N_FC_FEATURES_SHORT = 100
N_CLASS_SCORES = 21
N_DT_FEATURES_FULL = N_CLASS_SCORES + N_FC_FEATURES_FULL
N_DT_FEATURES_SHORT = N_CLASS_SCORES + N_FC_FEATURES_SHORT
N_OBJECTS = 20
N_CLASSES = 21


def get_frame_data(fid, data, n_bboxes):
    frame_data = {}
    fid_dt_ix = data[nnms.DT_COORDS][:, 0] == fid
    frame_data[nnms.DT_COORDS] = data[nnms.DT_COORDS][fid_dt_ix, 1:][0:n_bboxes]
    frame_data[nnms.DT_FEATURES] = data[nnms.DT_FEATURES][fid_dt_ix][0:n_bboxes]
    frame_data[nnms.DT_SCORES] = data[nnms.DT_SCORES][fid_dt_ix][0:n_bboxes]
    fid_gt_ix = data[nnms.GT_COORDS][:, 0] == fid
    frame_data[nnms.GT_COORDS] = data[nnms.GT_COORDS][fid_gt_ix, 1:5]
    frame_data[nnms.GT_LABELS] = data[nnms.GT_COORDS][fid_gt_ix, 5]
    frame_data[nnms.DT_GT_IOU] = bbox_utils.compute_sets_iou(
        frame_data[nnms.DT_COORDS], frame_data[nnms.GT_COORDS])
    # frame_data[nnms.DT_DT_IOU] = bbox_utils.compute_sets_iou(frame_data[nnms.DT_COORDS], frame_data[nnms.DT_COORDS])
    frame_data[nnms.DT_LABELS] = np.zeros([n_bboxes, N_CLASSES])
    frame_data[nnms.DT_LABELS_BASIC] = np.zeros([n_bboxes, N_CLASSES])
    for class_id in range(0, N_CLASSES):
        class_gt_boxes = frame_data[nnms.GT_LABELS] == class_id
        class_dt_gt = frame_data[nnms.DT_GT_IOU][:, class_gt_boxes]
        if class_dt_gt.shape[1] != 0:
            frame_data[nnms.DT_LABELS][:, class_id] = np.max(
                bbox_utils.compute_best_iou(class_dt_gt), axis=1)
            frame_data[nnms.DT_LABELS_BASIC][:, class_id][
                np.max(class_dt_gt, axis=1) > 0.5] = 1
    # logging.info('finished processing frame %d' % fid)
    return frame_data


def split_by_frames(data, n_bboxes):
    unique_fids = np.unique(np.hstack([data[nnms.DT_COORDS][:, 0], data[nnms.GT_COORDS][:, 0]])).astype(int)
    get_frame_data_partial = partial(get_frame_data, data=data, n_bboxes=n_bboxes)
    frames_data_train = dict(
        zip(unique_fids, map(get_frame_data_partial, unique_fids)))
    return frames_data_train


def preprocess_data(data_dir, n_bboxes, use_short_features=False):
    if use_short_features:
        dt_features_path = os.path.join(data_dir, 'dt_features_short.pkl')
    else :
        dt_features_path = os.path.join(data_dir, 'dt_features_full.pkl')
    data = {}
    data[nnms.DT_COORDS] = joblib.load(os.path.join(data_dir, 'dt_coords.pkl'))
    data[nnms.DT_SCORES] = joblib.load(os.path.join(data_dir, 'dt_scores.pkl'))
    data[nnms.DT_FEATURES] = joblib.load(os.path.join(data_dir, dt_features_path))
    data[nnms.GT_COORDS] = joblib.load(os.path.join(data_dir, 'gt_coords.pkl'))
    logging.info('finished loading data')
    frames_data_train = split_by_frames(data, n_bboxes)
    return frames_data_train


def load_data(data_dir, n_bboxes, use_short_features=False):
    if use_short_features:
        frames_data_cache_file = os.path.join(data_dir, 'frames_data_short_' + str(n_bboxes) + '.pkl')
    else:
        frames_data_cache_file = os.path.join(data_dir, 'frames_data_full_' + str(n_bboxes) + '.pkl')
    if os.path.exists(frames_data_cache_file):
        logging.info('loading frame bbox data info from cash..')
        frames_data = joblib.load(frames_data_cache_file)
    else:
        logging.info(
            'computing frame bbox data (IoU, labels, etc) - this could take some time..')
        frames_data = preprocess_data(data_dir, n_bboxes, use_short_features=use_short_features)
        joblib.dump(frames_data, frames_data_cache_file)
    return frames_data


def shuffle_samples(n_frames):
    return np.random.choice(n_frames, n_frames, replace=False)

def write_scalar_summary(value, name, summary_writer, step_id):
    test_map_summ = tf.Summary(
        value=[
            tf.Summary.Value(
                tag=name,
                simple_value=value),
        ])
    summary_writer.add_summary(
        test_map_summ, global_step=step_id)
    return


class ExperimentConfig:

    def _set_logging(self, to_stdout=True):
        if self.logging_to_stdout:
            logging.basicConfig(
                format='%(asctime)s : %(message)s',
                level=logging.INFO,
                stream=sys.stdout)
        else:
            logging.basicConfig(
                format='%(asctime)s : %(message)s',
                level=logging.INFO,
                filename=self.log_file)
            print("logs could be found at %s" % self.log_file)
        return

    def _create_log_dir(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        else:
            if self.start_from_scratch:
                shutil.rmtree(self.log_dir)
                os.makedirs(self.log_dir)

    def _backup_config(self):
        shutil.copy(self.config_path, self.log_dir)

    def _get_git_revision_hash(self):
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'])

    def update_results(self,
                       step_id,
                       train_map,
                       train_map_nms,
                       test_map,
                       test_map_nms,
                       mean_step_time):

            if test_map > self.results['max_test_map']:
                self.results['max_test_map'] = test_map
                self.results['max_test_map_step_id'] = step_id

            if test_map_nms > self.results['max_test_nms_map']:
                self.results['max_test_nms_map'] = test_map_nms

            if train_map > self.results['max_train_map']:
                self.results['max_train_map'] = train_map
                self.results['max_train_map_step_id'] = step_id

            if train_map_nms > self.results['max_train_nms_map']:
                self.results['max_train_nms_map'] = train_map_nms

            self.results['curr_train_map'] = train_map
            self.results['curr_train_nms_map'] = train_map_nms
            self.results['curr_test_map'] = test_map
            self.results['curr_test_nms_map'] = test_map_nms

            self.results['curr_step_id'] = step_id

            self.mean_train_step_time = mean_step_time


    def save_results(self):

        curr_res = pd.DataFrame(index=[self.id])

        curr_res['git_hash'] = self.git_hash

        for key, val in self.results.iteritems():
            curr_res[key] = val

        for key, val in self.dp_config.iteritems():
            curr_res[key] = val

        for key, val in self.nms_network_config['architecture'].iteritems():
            curr_res[key] = val

        for key, val in self.nms_network_config['training'].iteritems():
            curr_res[key] = val

        curr_res['mean_step_time'] = self.mean_train_step_time

        if os.path.exists(self.res_csv_path):
            res_df = pd.read_csv(self.res_csv_path, index_col=0)
            res_df.ix[self.id] = curr_res.ix[self.id]
            res_df.to_csv(self.res_csv_path)
        else:
            curr_res.to_csv(self.res_csv_path)
        return

    def __init__(self, data_dir, root_log_dir, config_path):

        self.id = binascii.hexlify(os.urandom(10))
        self.git_hash = self._get_git_revision_hash()

        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.load(f)

        self.general_params = self.config.get('general', {})
        self.start_from_scratch = self.general_params.get('start_from_scratch', True)
        self.logging_to_stdout = self.general_params.get('logging_to_stdout', True)

        self.log_dir = os.path.join(root_log_dir, self.id)
        self.log_file = os.path.join(self.log_dir, 'training.log')
        self.res_csv_path = os.path.join(root_log_dir, 'results_'+str(self.git_hash)+'.csv')
        self._create_log_dir()
        self._set_logging()
        self._backup_config()

        self.data_dir = data_dir
        self.train_data_dir = os.path.join(self.data_dir, 'train')
        self.test_data_dir = os.path.join(self.data_dir, 'test')

        self.dp_config = self.config.get('data_provider', {})
        self.n_bboxes = self.dp_config.get('n_bboxes', 20)
        self.use_reduced_fc_features = self.dp_config.get('use_reduced_fc_features', True)

        if self.use_reduced_fc_features:
            self.n_dt_features = N_DT_FEATURES_SHORT
        else:
            self.n_dt_features = N_DT_FEATURES_FULL

        self.nms_network_config = self.config.get('nms_network', {})
        self.model_file = os.path.join(self.log_dir, 'model')

        train_config = self.nms_network_config.get('training', {})

        self.n_epochs = train_config.get('n_epochs', 50)

        self.eval_config = self.nms_network_config.get('evaluation', {})
        self.eval_step = self.eval_config.get('eval_step', 1000)
        self.full_eval = self.eval_config.get('full_eval', False)
        self.n_eval_frames = self.eval_config.get('n_eval_frames', 1000)
        self.nms_thres = self.eval_config.get('nms_thres', 0.5)

        # results details
        self.mean_train_step_time = 0.0

        self.results = {}

        self.results['max_test_map'] = 0.0
        self.results['max_train_map'] = 0.0
        self.results['max_train_nms_map'] = 0.0
        self.results['max_train_map_step_id'] = 0.0
        self.results['max_test_nms_map'] = 0.0
        self.results['max_test_map_step_id'] = 0.0
        self.results['curr_step_id'] = 0.0
        self.results['curr_train_map'] = 0.0
        self.results['curr_train_nms_map'] = 0.0
        self.results['curr_test_map'] = 0.0
        self.results['curr_test_nms_map'] = 0.0


def eval_and_save(sess, model, config):
    return


def main(_):

    config = ExperimentConfig(data_dir=FLAGS.data_dir,
                              root_log_dir=FLAGS.log_dir,
                              config_path=FLAGS.config_path)

    config.save_results()

    logging.info('loading data..')
    logging.info('train..')
    frames_data_train = load_data(config.train_data_dir,
                                  n_bboxes=config.n_bboxes,
                                  use_short_features=config.use_reduced_fc_features)
    logging.info('test..')
    frames_data_test = load_data(config.test_data_dir,
                                 n_bboxes=config.n_bboxes,
                                 use_short_features=config.use_reduced_fc_features)

    n_frames_train = len(frames_data_train.keys())
    n_frames_test = len(frames_data_test.keys())

    logging.info('building model graph..')

    nnms_model = nnms.NeuralNMS(n_detections=config.n_bboxes,
                                n_dt_features=config.n_dt_features,
                                n_classes=N_CLASSES,
                                **config.nms_network_config)

    with tf.Session() as sess:
        step_id = 0
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(
            max_to_keep=5,
            keep_checkpoint_every_n_hours=1.0)

        if not config.start_from_scratch:
            ckpt_path = tf.train.latest_checkpoint(config.log_dir)
            if ckpt_path is not None:
                logging.info('model exists, restoring..')
                ckpt_name = ntpath.basename(ckpt_path)
                step_id = int(ckpt_name.split('-')[1])
                saver.restore(sess, ckpt_path)

        summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph)

        logging.info('training started..')
        for epoch_id in range(0, config.n_epochs):

            step_times = []

            for fid in shuffle_samples(n_frames_train):
                frame_data = frames_data_train[fid]
                feed_dict = {nnms_model.dt_coords: frame_data[nnms.DT_COORDS],
                             nnms_model.dt_features: frame_data[nnms.DT_FEATURES],
                             nnms_model.dt_labels: frame_data[nnms.DT_LABELS],
                             nnms_model.dt_gt_iou: frame_data[nnms.DT_GT_IOU],
                             nnms_model.gt_labels: frame_data[nnms.GT_LABELS]}

                start_step = timer()

                summary, _ = sess.run([nnms_model.merged_summaries, nnms_model.train_step],
                                      feed_dict=feed_dict)
                end_step = timer()

                step_times.append(end_step-start_step)

                summary_writer.add_summary(summary, global_step=step_id)
                summary_writer.flush()

                step_id += 1

            if epoch_id % config.eval_step == 0:

                logging.info('step : %d' % step_id)

                fid = shuffle_samples(n_frames_test)[0]

                frame_data = frames_data_test[fid]

                eval.print_debug_info(sess=sess,
                                      nnms_model=nnms_model,
                                      frame_data=frame_data,
                                      outdir=config.log_dir,
                                      fid=fid)

                logging.info('evaluating on TRAIN..')
                train_out_dir = os.path.join(config.log_dir, 'train')
                train_map, train_map_nms = eval.eval_model(sess, nnms_model,
                                            frames_data_train,
                                            global_step=step_id,
                                            n_eval_frames=config.n_eval_frames,
                                            out_dir=train_out_dir,
                                            full_eval=config.full_eval,
                                            nms_thres=config.nms_thres)
                write_scalar_summary(train_map, 'train_map', summary_writer, step_id)

                logging.info('evaluating on TEST..')
                test_out_dir = os.path.join(config.log_dir, 'test')
                test_map, test_map_nms = eval.eval_model(sess, nnms_model,
                                           frames_data_test,
                                           global_step=step_id,
                                           n_eval_frames=config.n_eval_frames,
                                           out_dir=test_out_dir,
                                           full_eval=config.full_eval,
                                           nms_thres=config.nms_thres)
                write_scalar_summary(test_map, 'test_map', summary_writer, step_id)

                config.update_results(step_id,
                                      train_map,
                                      train_map_nms,
                                      test_map,
                                      test_map_nms,
                                      np.mean(step_times))

                config.save_results()

                saver.save(sess, config.model_file, global_step=step_id)
    return

if __name__ == '__main__':
    gflags.mark_flag_as_required('data_dir')
    gflags.mark_flag_as_required('log_dir')
    gflags.mark_flag_as_required('config_path')
    app.run()

import logging
import sys
import subprocess
import binascii
import yaml
import os
import pandas as pd
import shutil


class ExperimentConfig:

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
        self.res_csv_path = os.path.join(root_log_dir, 'results.csv')
        self._create_log_dir()
        self._set_logging()
        self._backup_config()

        self.data_dir = data_dir
        self.train_data_dir = os.path.join(self.data_dir, 'train')
        self.test_data_dir = os.path.join(self.data_dir, 'test')

        self.dp_config = self.config.get('data_provider', {})
        self.n_bboxes = self.dp_config.get('n_bboxes', 20)
        self.n_dt_features = self.dp_config.get('n_features', 100)
        self.use_reduced_fc_features = self.dp_config.get('use_reduced_fc_features', True)
        self.shuffle_train_test = self.dp_config.get('shuffle_train_test', False)

        self.nms_network_config = self.config.get('nms_network', {})
        self.model_file = os.path.join(self.log_dir, 'model')

        train_config = self.nms_network_config.get('training', {})

        self.n_epochs = train_config.get('n_epochs', 10)

        self.eval_config = self.nms_network_config.get('evaluation', {})
        self.eval_step = self.eval_config.get('eval_step', 1000)
        self.full_eval = self.eval_config.get('full_eval', False)
        self.n_eval_frames = self.eval_config.get('n_eval_frames', 1000)
        self.nms_thres = self.eval_config.get('nms_thres', 0.5)
        self.train_config = self.nms_network_config.get('training', {})
        self.keep_prob_train = self.train_config.get('keep_prob')

        self.learning_rate = self.train_config.get('learning_rate', 0.001)
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



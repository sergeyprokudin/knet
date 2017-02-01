#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"

EXPERIMENT_DIR="${SCRIPT_DIR}/.."

PROJECT_DIR="${SCRIPT_DIR}/../../.."

source "${PROJECT_DIR}/scripts/dbash.sh" || exit 1

dbash::cluster_cuda
dbash::mac_cuda

cd ${PROJECT_DIR}

set -x

${PYENV_BIN} experiments/faster_rcnn_knet_finetune/model/train.py  \
            --data_dir="${EXPERIMENT_DIR}/data/pascal_voc_2007/" \
            --log_dir="${EXPERIMENT_DIR}/logs/pascal_voc_2007/" \
            --use_reduced_fc_features=True \
            --num_cpus=10 \
            --pos_weight=0.1 \
            --n_neg_samples=10 \
            --optimizer_step=0.0001 \
            --knet_hlayer_size=100 \
            --fc_layer_size=100 \
            --n_kernels=10 \
            --softmax_loss=False \
            --softmax_kernel=True \
            --use_object_features=True \
            --use_coords_features=False \
            --use_iou_features=False \
            --start_from_scratch=True \
            --eval_step=10 \
            --n_eval_frames=100 \
            --logging_to_stdout=True

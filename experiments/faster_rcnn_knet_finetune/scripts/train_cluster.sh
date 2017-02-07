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
            --n_bboxes=300 \
            --num_cpus=1 \
            --pos_weight=1 \
            --n_neg_samples=10 \
            --optimizer_step=0.0001 \
            --knet_hlayer_size=100 \
            --fc_layer_size=100 \
            --n_kernels=10 \
            --n_kernel_iterations=2\
            --softmax_loss=False \
            --softmax_kernel=True \
            --use_object_features=True \
            --use_coords_features=True \
            --use_iou_features=True \
            --start_from_scratch=True \
            --eval_step=10000 \
            --full_eval=True \
            --logging_to_stdout=False
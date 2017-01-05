#!/bin/bash

# Include our base tools.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"

source "${SCRIPT_DIR}/dbash.sh" || exit 1

CVPRPY="${SCRIPT_DIR}/../py_env"
source ${CVPRPY}/bin/activate

dbash::pp "# Running autopep for experiments/faster_rcnn_knet_finetune/model"
autopep8 -i -r -a ${SCRIPT_DIR}/../experiments/faster_rcnn_knet_finetune/model/*.py

dbash::pp "# Running autopep for tf_layers"
autopep8 -i -r -a ${SCRIPT_DIR}/../tf_layers

dbash::pp "# Running autopep for tools"
autopep8 -i -r -a ${SCRIPT_DIR}/../tools

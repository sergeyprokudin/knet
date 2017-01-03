#!/bin/bash

# Include our base tools.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"

source "${SCRIPT_DIR}/dbash.sh" || exit 1

CVPRPY="${SCRIPT_DIR}/../py_env"
source ${CVPRPY}/bin/activate

dbash::pp "# Running autopep for knet."
autopep8 -i -r -a ${SCRIPT_DIR}/../experiments/faster_rcnn_knet_finetune/model/*.py

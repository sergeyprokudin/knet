DBASH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"

PROJECT_DIR="${DBASH_DIR}/.."

source "${PROJECT_DIR}/py_env/bin/activate"

tensorboard --logdir=$1

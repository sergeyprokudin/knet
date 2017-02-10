
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"

PROJECT_DIR="${SCRIPT_DIR}/.."

source "${SCRIPT_DIR}/dbash.sh" || exit 1

cd ${SCRIPT_DIR}

PYENV="${SCRIPT_DIR}/../py_env"
if [[ ! -e ${PYENV} ]];then
    dbash::pp "# We setup a virtual environment for this project!"
    if ! dbash::command_exists virtualenv;then
        dbash::pp "# We install virtualenv!"
        sudo pip install virtualenv
    fi
    virtualenv ${PYENV} --clear
    virtualenv ${PYENV} --relocatable
fi

source ${PYENV}/bin/activate

dbash::pp "# Should we upgrade all dependencies?"
dbash::user_confirm ">> Update dependencies?" "n"
if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
    ${PYENV}/bin/pip install --upgrade pip
    ${PYENV}/bin/pip install --upgrade \
             numpy scipy matplotlib joblib ipdb python-gflags google-apputils autopep8 yaml
fi

dbash::user_confirm ">> Install tensorflow gpu MAC?" "n"
if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
    dbash::pp "Please install cuda 8.0 from nvidia!"
    dbash::pp "Please install cudnn 5.0 from nvidia!"
    dbash::pp "Notice, symbolic links for libcudnn.dylib and libcuda.dylib have to be added."
    export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow_gpu-0.12.0-py2-none-any.whl
    ${PYENV}/bin/pip install --ignore-installed --upgrade $TF_BINARY_URL
fi

dbash::user_confirm ">> Install tensorflow cpu MAC?" "n"
if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
    export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0-py2-none-any.whl
    ${PYENV}/bin/pip install --ignore-installed --upgrade $TF_BINARY_URL
fi

dbash::user_confirm ">> Install tensorflow gpu ubuntu?" "n"
if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
    export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.0-cp27-none-linux_x86_64.whl
    ${PYENV}/bin/pip install --ignore-installed --upgrade $TF_BINARY_URL
fi

dbash::pp "# We register our modules as develop modules."
python "${PROJECT_DIR}/setup.py" develop

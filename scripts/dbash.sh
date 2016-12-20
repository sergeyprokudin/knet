
DBASH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"

PROJECT_DIR="${DBASH_DIR}/.."

PYENV_BIN="${DBASH_DIR}/../py_env/bin/python"
PYENV_IBIN="${DBASH_DIR}/../py_env/bin/ipython"

dbash::check_computation_device() {
    if [[ "$1" == "cpu" ]];then
        USE_GPU="f"
    elif [[ "$1" == "gpu" ]];then
        USE_GPU="t"
    else
        echo "$1 not in [cpu, gpu]."
        exit
    fi
}

dbash::source_py_env() {
    source "${DBASH_DIR}/../py_env/bin/activate"
}

dbash::command_exists() {
	command -v "$@" > /dev/null 2>&1
}

dbash::pp() {
    echo -e "$1"
}

dbash::user_confirm() {
    NOT_FINISHED=true
    while ${NOT_FINISHED} ;do
        echo -e -n "$1 [y/n] default($2) "
        read USER_INPUT;
        if [[ "y" == "${USER_INPUT}" ]];then
            USER_CONFIRM_RESULT="y";
            NOT_FINISHED=false;
        elif [[ "n" == "${USER_INPUT}" ]];then
            USER_CONFIRM_RESULT="n";
            NOT_FINISHED=false;
        elif [[ "" == "${USER_INPUT}" ]];then
            USER_CONFIRM_RESULT="$2";
            NOT_FINISHED=false;
        else
            echo -e "# only y, n, and nothing, are possible choices."
            echo -e "# default is $2"
        fi
    done
}


dbash::cluster_cuda() {
    local LPATH="/lustre/shared/caffe_shared/cuda_stuff/cuda-8.0.27.1_RC/lib64"
    if [[ -d ${LPATH} ]];then
        dbash::pp "cuda cluster ${LPATH}"
        export LD_LIBRARY_PATH=${LPATH}:$LD_LIBRARY_PATH
    fi

    local LPATH="/usr/local/cudnn-5.1/lib64"
    if [[ -d ${LPATH} ]];then
        dbash::pp "cudnn cluster ${LPATH}"
        export LD_LIBRARY_PATH=${LPATH}:$LD_LIBRARY_PATH
    fi
}


dbash::mac_cuda() {
    local LPATH="/lustre/shared/caffe_shared/cuda_stuff/cuda-8.0.27.1_RC/lib64"
    if [[ ! -d ${LPATH} ]];then
        local LPATH="/usr/local/cuda/lib"
        if [[ -d ${LPATH} ]];then
            dbash::pp "cuda mac ${LPATH}"
            export LD_LIBRARY_PATH=${LPATH}:$LD_LIBRARY_PATH
        fi
    fi
}

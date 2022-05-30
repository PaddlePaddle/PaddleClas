#!/bin/bash
source test_tipc/common_func.sh

function func_parser_key_cpp(){
    strs=$1
    IFS=" "
    array=(${strs})
    tmp=${array[0]}
    echo ${tmp}
}

function func_parser_value_cpp(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}

FILENAME=$1

dataline=$(cat ${FILENAME})
lines=(${dataline})

# parser params
dataline=$(awk 'NR==1, NR==14{print}'  $FILENAME)
IFS=$'\n'
lines=(${dataline})

# parser load config
model_name=$(func_parser_value_cpp "${lines[1]}")
use_gpu_key=$(func_parser_key_cpp "${lines[2]}")
use_gpu_value=$(func_parser_value_cpp "${lines[2]}")
LOG_PATH="./test_tipc/output/${model_name}/infer_cpp"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_infer_cpp.log"

line_inference_model_dir=3
line_use_gpu=5
function func_infer_cpp(){
    # inference cpp
    IFS='|'
    for use_gpu in ${use_gpu_value[*]}; do
        if [[ ${use_gpu} = "True" ]]; then
            _save_log_path="${LOG_PATH}/infer_cpp_use_gpu.log"
        else
            _save_log_path="${LOG_PATH}/infer_cpp_use_cpu.log"
        fi
        # run infer cpp
        inference_cpp_cmd="./deploy/cpp/build/clas_system"
        inference_cpp_cfg="./deploy/configs/inference_cls.yaml"
        set_model_name_cmd="sed -i '${line_inference_model_dir}s#: .*#: ./deploy/models/${model_name}_infer#' '${inference_cpp_cfg}'"
        set_use_gpu_cmd="sed -i '${line_use_gpu}s#: .*#: ${use_gpu}#' '${inference_cpp_cfg}'"
        eval $set_model_name_cmd
        eval $set_use_gpu_cmd
        infer_cpp_full_cmd="${inference_cpp_cmd} -c ${inference_cpp_cfg} > ${_save_log_path} 2>&1 "
        eval $infer_cpp_full_cmd
        last_status=${PIPESTATUS[0]}
        status_check $last_status "${infer_cpp_full_cmd}" "${status_log}"  "${model_name}"
    done
}

echo "################### run test cpp inference ###################"

func_infer_cpp
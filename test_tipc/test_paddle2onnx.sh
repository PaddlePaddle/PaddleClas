#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1

# parser params
dataline=$(awk 'NR==1, NR==16{print}'  $FILENAME)
IFS=$'\n'
lines=(${dataline})

# parser paddle2onnx
model_name=$(func_parser_value "${lines[1]}")
python=$(func_parser_value "${lines[2]}")
padlle2onnx_cmd=$(func_parser_value "${lines[3]}")
infer_model_dir_key=$(func_parser_key "${lines[4]}")
infer_model_dir_value=$(func_parser_value "${lines[4]}")
model_filename_key=$(func_parser_key "${lines[5]}")
model_filename_value=$(func_parser_value "${lines[5]}")
params_filename_key=$(func_parser_key "${lines[6]}")
params_filename_value=$(func_parser_value "${lines[6]}")
save_file_key=$(func_parser_key "${lines[7]}")
save_file_value=$(func_parser_value "${lines[7]}")
opset_version_key=$(func_parser_key "${lines[8]}")
opset_version_value=$(func_parser_value "${lines[8]}")
enable_onnx_checker_key=$(func_parser_key "${lines[9]}")
enable_onnx_checker_value=$(func_parser_value "${lines[9]}")
# parser onnx inference
inference_py=$(func_parser_value "${lines[11]}")
use_onnx_key=$(func_parser_key "${lines[12]}")
use_onnx_value=$(func_parser_value "${lines[12]}")
inference_model_dir_key=$(func_parser_key "${lines[13]}")
inference_model_dir_value=$(func_parser_value "${lines[13]}")
inference_hardware_key=$(func_parser_key "${lines[14]}")
inference_hardware_value=$(func_parser_value "${lines[14]}")
inference_config_key=$(func_parser_key "${lines[15]}")
inference_config_value=$(func_parser_value "${lines[15]}")

LOG_PATH="./test_tipc/output/${model_name}"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_paddle2onnx.log"


function func_paddle2onnx(){
    IFS='|'
    _script=$1

    # paddle2onnx
    _save_log_path=".${LOG_PATH}/paddle2onnx_infer_cpu.log"
    set_dirname=$(func_set_params "${infer_model_dir_key}" "${infer_model_dir_value}")
    set_model_filename=$(func_set_params "${model_filename_key}" "${model_filename_value}")
    set_params_filename=$(func_set_params "${params_filename_key}" "${params_filename_value}")
    set_save_model=$(func_set_params "${save_file_key}" "${save_file_value}")
    set_opset_version=$(func_set_params "${opset_version_key}" "${opset_version_value}")
    set_enable_onnx_checker=$(func_set_params "${enable_onnx_checker_key}" "${enable_onnx_checker_value}")
    trans_model_cmd="${padlle2onnx_cmd} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_save_model} ${set_opset_version} ${set_enable_onnx_checker}"
    eval $trans_model_cmd
    last_status=${PIPESTATUS[0]}
    status_check $last_status "${trans_model_cmd}" "${status_log}" "${model_name}"

    # python inference
    set_model_dir=$(func_set_params "${inference_model_dir_key}" "${inference_model_dir_value}")
    set_use_onnx=$(func_set_params "${use_onnx_key}" "${use_onnx_value}")
    set_hardware=$(func_set_params "${inference_hardware_key}" "${inference_hardware_value}")
    set_inference_config=$(func_set_params "${inference_config_key}" "${inference_config_value}")
    infer_model_cmd="cd deploy && ${python} ${inference_py} -o ${set_model_dir} -o ${set_use_onnx} -o ${set_hardware} ${set_inference_config} > ${_save_log_path} 2>&1 && cd ../"
    eval $infer_model_cmd
    status_check $last_status "${infer_model_cmd}" "${status_log}" "${model_name}"
}


echo "################### run test ###################"

export Count=0
IFS="|"
func_paddle2onnx
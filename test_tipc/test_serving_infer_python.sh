#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
MODE=$2
dataline=$(awk 'NR==1, NR==19{print}'  $FILENAME)

# parser params
IFS=$'\n'
lines=(${dataline})

function func_get_url_file_name(){
    strs=$1
    IFS="/"
    array=(${strs})
    tmp=${array[${#array[@]}-1]}
    echo ${tmp}
}

# parser serving
model_name=$(func_parser_value "${lines[1]}")
python=$(func_parser_value "${lines[2]}")
trans_model_py=$(func_parser_value "${lines[4]}")
infer_model_dir_key=$(func_parser_key "${lines[5]}")
infer_model_dir_value=$(func_parser_value "${lines[5]}")
model_filename_key=$(func_parser_key "${lines[6]}")
model_filename_value=$(func_parser_value "${lines[6]}")
params_filename_key=$(func_parser_key "${lines[7]}")
params_filename_value=$(func_parser_value "${lines[7]}")
serving_server_key=$(func_parser_key "${lines[8]}")
serving_server_value=$(func_parser_value "${lines[8]}")
serving_client_key=$(func_parser_key "${lines[9]}")
serving_client_value=$(func_parser_value "${lines[9]}")
serving_dir_value=$(func_parser_value "${lines[10]}")
web_service_py=$(func_parser_value "${lines[11]}")
web_use_gpu_key=$(func_parser_key "${lines[12]}")
web_use_gpu_list=$(func_parser_value "${lines[12]}")
pipeline_py=$(func_parser_value "${lines[13]}")


function func_serving_cls(){
    LOG_PATH="test_tipc/output/${model_name}/${MODE}"
    mkdir -p ${LOG_PATH}
    LOG_PATH="../../${LOG_PATH}"
    status_log="${LOG_PATH}/results_serving.log"
    IFS='|'

    # pdserving
    set_dirname=$(func_set_params "${infer_model_dir_key}" "${infer_model_dir_value}")
    set_model_filename=$(func_set_params "${model_filename_key}" "${model_filename_value}")
    set_params_filename=$(func_set_params "${params_filename_key}" "${params_filename_value}")
    set_serving_server=$(func_set_params "${serving_server_key}" "${serving_server_value}")
    set_serving_client=$(func_set_params "${serving_client_key}" "${serving_client_value}")

    for python_ in ${python[*]}; do
        if [[ ${python_} =~ "python" ]]; then
            trans_model_cmd="${python_} ${trans_model_py} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_serving_server} ${set_serving_client}"
            eval ${trans_model_cmd}
            break
        fi
    done

    # modify the alias_name of fetch_var to "outputs"
    server_fetch_var_line_cmd="sed -i '/fetch_var/,/is_lod_tensor/s/alias_name: .*/alias_name: \"prediction\"/' ${serving_server_value}/serving_server_conf.prototxt"
    eval ${server_fetch_var_line_cmd}

    client_fetch_var_line_cmd="sed -i '/fetch_var/,/is_lod_tensor/s/alias_name: .*/alias_name: \"prediction\"/' ${serving_client_value}/serving_client_conf.prototxt"
    eval ${client_fetch_var_line_cmd}

    prototxt_dataline=$(awk 'NR==1, NR==3{print}'  ${serving_server_value}/serving_server_conf.prototxt)
    IFS=$'\n'
    prototxt_lines=(${prototxt_dataline})
    feed_var_name=$(func_parser_value "${prototxt_lines[2]}")
    IFS='|'

    cd ${serving_dir_value}
    unset https_proxy
    unset http_proxy

    # python serving
    # modify the input_name in "classification_web_service.py" to be consistent with feed_var.name in prototxt
    set_web_service_feed_var_cmd="sed -i '/preprocess/,/input_imgs}/s/{.*: input_imgs}/{${feed_var_name}: input_imgs}/' ${web_service_py}"
    eval ${set_web_service_feed_var_cmd}

    model_config=21
    serving_server_dir_name=$(func_get_url_file_name "$serving_server_value")
    set_model_config_cmd="sed -i '${model_config}s/model_config: .*/model_config: ${serving_server_dir_name}/' config.yml"
    eval ${set_model_config_cmd}

    for use_gpu in ${web_use_gpu_list[*]}; do
        if [[ ${use_gpu} = "null" ]]; then
            device_type_line=24
            set_device_type_cmd="sed -i '${device_type_line}s/device_type: .*/device_type: 0/' config.yml"
            eval ${set_device_type_cmd}

            devices_line=27
            set_devices_cmd="sed -i '${devices_line}s/devices: .*/devices: \"\"/' config.yml"
            eval ${set_devices_cmd}

            web_service_cmd="${python_} ${web_service_py} &"
            eval ${web_service_cmd}
            last_status=${PIPESTATUS[0]}
            status_check $last_status "${web_service_cmd}" "${status_log}" "${model_name}"
            sleep 5s
            for pipeline in ${pipeline_py[*]}; do
                _save_log_path="${LOG_PATH}/server_infer_cpu_${pipeline%_client*}_batchsize_1.log"
                pipeline_cmd="${python_} ${pipeline} > ${_save_log_path} 2>&1 "
                eval ${pipeline_cmd}
                last_status=${PIPESTATUS[0]}
                eval "cat ${_save_log_path}"
                status_check $last_status "${pipeline_cmd}" "${status_log}" "${model_name}"
                sleep 5s
            done
            eval "${python_} -m paddle_serving_server.serve stop"
        elif [ ${use_gpu} -eq 0 ]; then
            if [[ ${_flag_quant} = "False" ]] && [[ ${precision} =~ "int8" ]]; then
                continue
            fi
            if [[ ${precision} =~ "fp16" || ${precision} =~ "int8" ]] && [ ${use_trt} = "False" ]; then
                continue
            fi
            if [[ ${use_trt} = "False" || ${precision} =~ "int8" ]] && [[ ${_flag_quant} = "True" ]]; then
                continue
            fi

            device_type_line=24
            set_device_type_cmd="sed -i '${device_type_line}s/device_type: .*/device_type: 1/' config.yml"
            eval ${set_device_type_cmd}

            devices_line=27
            set_devices_cmd="sed -i '${devices_line}s/devices: .*/devices: \"${use_gpu}\"/' config.yml"
            eval ${set_devices_cmd}

            web_service_cmd="${python_} ${web_service_py} & "
            eval ${web_service_cmd}
            last_status=${PIPESTATUS[0]}
            status_check $last_status "${web_service_cmd}" "${status_log}" "${model_name}"
            sleep 5s
            for pipeline in ${pipeline_py[*]}; do
                _save_log_path="${LOG_PATH}/server_infer_gpu_${pipeline%_client*}_batchsize_1.log"
                pipeline_cmd="${python_} ${pipeline} > ${_save_log_path} 2>&1"
                eval ${pipeline_cmd}
                last_status=${PIPESTATUS[0]}
                eval "cat ${_save_log_path}"
                status_check $last_status "${pipeline_cmd}" "${status_log}" "${model_name}"
                sleep 5s
            done
            eval "${python_} -m paddle_serving_server.serve stop"
        else
            echo "Does not support hardware [${use_gpu}] other than CPU and GPU Currently!"
        fi
    done
}


function func_serving_rec(){
    LOG_PATH="test_tipc/output/${model_name}/${MODE}"
    mkdir -p ${LOG_PATH}
    LOG_PATH="../../../${LOG_PATH}"
    status_log="${LOG_PATH}/results_serving.log"
    trans_model_py=$(func_parser_value "${lines[5]}")
    cls_infer_model_dir_key=$(func_parser_key "${lines[6]}")
    cls_infer_model_dir_value=$(func_parser_value "${lines[6]}")
    det_infer_model_dir_key=$(func_parser_key "${lines[7]}")
    det_infer_model_dir_value=$(func_parser_value "${lines[7]}")
    model_filename_key=$(func_parser_key "${lines[8]}")
    model_filename_value=$(func_parser_value "${lines[8]}")
    params_filename_key=$(func_parser_key "${lines[9]}")
    params_filename_value=$(func_parser_value "${lines[9]}")

    cls_serving_server_key=$(func_parser_key "${lines[10]}")
    cls_serving_server_value=$(func_parser_value "${lines[10]}")
    cls_serving_client_key=$(func_parser_key "${lines[11]}")
    cls_serving_client_value=$(func_parser_value "${lines[11]}")

    det_serving_server_key=$(func_parser_key "${lines[12]}")
    det_serving_server_value=$(func_parser_value "${lines[12]}")
    det_serving_client_key=$(func_parser_key "${lines[13]}")
    det_serving_client_value=$(func_parser_value "${lines[13]}")

    serving_dir_value=$(func_parser_value "${lines[14]}")
    web_service_py=$(func_parser_value "${lines[15]}")
    web_use_gpu_key=$(func_parser_key "${lines[16]}")
    web_use_gpu_list=$(func_parser_value "${lines[16]}")
    pipeline_py=$(func_parser_value "${lines[17]}")

    IFS='|'
    for python_ in ${python[*]}; do
        if [[ ${python_} =~ "python" ]]; then
            python_interp=${python_}
            break
        fi
    done

    # pdserving
    cd ./deploy
    set_dirname=$(func_set_params "${cls_infer_model_dir_key}" "${cls_infer_model_dir_value}")
    set_model_filename=$(func_set_params "${model_filename_key}" "${model_filename_value}")
    set_params_filename=$(func_set_params "${params_filename_key}" "${params_filename_value}")
    set_serving_server=$(func_set_params "${cls_serving_server_key}" "${cls_serving_server_value}")
    set_serving_client=$(func_set_params "${cls_serving_client_key}" "${cls_serving_client_value}")
    cls_trans_model_cmd="${python_interp} ${trans_model_py} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_serving_server} ${set_serving_client}"
    eval ${cls_trans_model_cmd}

    set_dirname=$(func_set_params "${det_infer_model_dir_key}" "${det_infer_model_dir_value}")
    set_model_filename=$(func_set_params "${model_filename_key}" "${model_filename_value}")
    set_params_filename=$(func_set_params "${params_filename_key}" "${params_filename_value}")
    set_serving_server=$(func_set_params "${det_serving_server_key}" "${det_serving_server_value}")
    set_serving_client=$(func_set_params "${det_serving_client_key}" "${det_serving_client_value}")
    det_trans_model_cmd="${python_interp} ${trans_model_py} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_serving_server} ${set_serving_client}"
    eval ${det_trans_model_cmd}

    # modify the alias_name of fetch_var to "outputs"
    server_fetch_var_line_cmd="sed -i '/fetch_var/,/is_lod_tensor/s/alias_name: .*/alias_name: \"features\"/' $cls_serving_server_value/serving_server_conf.prototxt"
    eval ${server_fetch_var_line_cmd}
    client_fetch_var_line_cmd="sed -i '/fetch_var/,/is_lod_tensor/s/alias_name: .*/alias_name: \"features\"/' $cls_serving_client_value/serving_client_conf.prototxt"
    eval ${client_fetch_var_line_cmd}

    prototxt_dataline=$(awk 'NR==1, NR==3{print}'  ${cls_serving_server_value}/serving_server_conf.prototxt)
    IFS=$'\n'
    prototxt_lines=(${prototxt_dataline})
    feed_var_name=$(func_parser_value "${prototxt_lines[2]}")
    IFS='|'

    cd ${serving_dir_value}
    unset https_proxy
    unset http_proxy

    # modify the input_name in "recognition_web_service.py" to be consistent with feed_var.name in prototxt
    set_web_service_feed_var_cmd="sed -i '/preprocess/,/input_imgs}/s/{.*: input_imgs}/{${feed_var_name}: input_imgs}/' ${web_service_py}"
    eval ${set_web_service_feed_var_cmd}
    # python serving
    for use_gpu in ${web_use_gpu_list[*]}; do
        if [[ ${use_gpu} = "null" ]]; then
            device_type_line=24
            set_device_type_cmd="sed -i '${device_type_line}s/device_type: .*/device_type: 0/' config.yml"
            eval ${set_device_type_cmd}

            devices_line=27
            set_devices_cmd="sed -i '${devices_line}s/devices: .*/devices: \"\"/' config.yml"
            eval ${set_devices_cmd}

            web_service_cmd="${python} ${web_service_py} &"
            eval ${web_service_cmd}
            last_status=${PIPESTATUS[0]}
            status_check $last_status "${web_service_cmd}" "${status_log}" "${model_name}"
            sleep 5s
            for pipeline in ${pipeline_py[*]}; do
                _save_log_path="${LOG_PATH}/server_infer_cpu_${pipeline%_client*}_batchsize_1.log"
                pipeline_cmd="${python} ${pipeline} > ${_save_log_path} 2>&1 "
                eval ${pipeline_cmd}
                last_status=${PIPESTATUS[0]}
                eval "cat ${_save_log_path}"
                status_check $last_status "${pipeline_cmd}" "${status_log}" "${model_name}"
                sleep 5s
            done
            eval "${python_} -m paddle_serving_server.serve stop"
        elif [ ${use_gpu} -eq 0 ]; then
            if [[ ${_flag_quant} = "False" ]] && [[ ${precision} =~ "int8" ]]; then
                continue
            fi
            if [[ ${precision} =~ "fp16" || ${precision} =~ "int8" ]] && [ ${use_trt} = "False" ]; then
                continue
            fi
            if [[ ${use_trt} = "False" || ${precision} =~ "int8" ]] && [[ ${_flag_quant} = "True" ]]; then
                continue
            fi

            device_type_line=24
            set_device_type_cmd="sed -i '${device_type_line}s/device_type: .*/device_type: 1/' config.yml"
            eval ${set_device_type_cmd}

            devices_line=27
            set_devices_cmd="sed -i '${devices_line}s/devices: .*/devices: \"${use_gpu}\"/' config.yml"
            eval ${set_devices_cmd}

            web_service_cmd="${python} ${web_service_py} & "
            eval ${web_service_cmd}
            last_status=${PIPESTATUS[0]}
            status_check $last_status "${web_service_cmd}" "${status_log}" "${model_name}"
            sleep 10s
            for pipeline in ${pipeline_py[*]}; do
                _save_log_path="${LOG_PATH}/server_infer_gpu_${pipeline%_client*}_batchsize_1.log"
                pipeline_cmd="${python} ${pipeline} > ${_save_log_path} 2>&1"
                eval ${pipeline_cmd}
                last_status=${PIPESTATUS[0]}
                eval "cat ${_save_log_path}"
                status_check $last_status "${pipeline_cmd}" "${status_log}" "${model_name}"
                sleep 10s
            done
            eval "${python_} -m paddle_serving_server.serve stop"
        else
            echo "Does not support hardware [${use_gpu}] other than CPU and GPU Currently!"
        fi
    done
}


# set cuda device
GPUID=$3
if [ ${#GPUID} -le 0 ];then
    env="export CUDA_VISIBLE_DEVICES=0"
else
    env="export CUDA_VISIBLE_DEVICES=${GPUID}"
fi
set CUDA_VISIBLE_DEVICES
eval ${env}


echo "################### run test ###################"

export Count=0
IFS="|"
if [[ ${model_name} = "PPShiTu" ]]; then
    func_serving_rec
else
    func_serving_cls
fi

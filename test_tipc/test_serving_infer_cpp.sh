#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
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
    LOG_PATH="test_tipc/output/${model_name}"
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

    for item in ${python[*]}; do
        if [[ ${item} =~ "python" ]]; then
            python_=${item}
            break
        fi
    done
    serving_client_dir_name=$(func_get_url_file_name "$serving_client_value")
    set_client_feed_type_cmd="sed -i '/feed_type/,/: .*/s/feed_type: .*/feed_type: 20/' ${serving_client_dir_name}/serving_client_conf.prototxt"
    eval ${set_client_feed_type_cmd}
    set_client_shape_cmd="sed -i '/shape: 3/,/shape: 3/s/shape: 3/shape: 1/' ${serving_client_dir_name}/serving_client_conf.prototxt"
    eval ${set_client_shape_cmd}
    set_client_shape224_cmd="sed -i '/shape: 224/,/shape: 224/s/shape: 224//' ${serving_client_dir_name}/serving_client_conf.prototxt"
    eval ${set_client_shape224_cmd}
    set_client_shape224_cmd="sed -i '/shape: 224/,/shape: 224/s/shape: 224//' ${serving_client_dir_name}/serving_client_conf.prototxt"
    eval ${set_client_shape224_cmd}

    set_pipeline_load_config_cmd="sed -i '/load_client_config/,/.prototxt/s/.\/.*\/serving_client_conf.prototxt/.\/${serving_client_dir_name}\/serving_client_conf.prototxt/' ${pipeline_py}"
    eval ${set_pipeline_load_config_cmd}

    set_pipeline_feed_var_cmd="sed -i '/feed=/,/: image}/s/feed={.*: image}/feed={${feed_var_name}: image}/' ${pipeline_py}"
    eval ${set_pipeline_feed_var_cmd}

    serving_server_dir_name=$(func_get_url_file_name "$serving_server_value")

    for use_gpu in ${web_use_gpu_list[*]}; do
        if [[ ${use_gpu} = "null" ]]; then
            web_service_cpp_cmd="${python_} -m paddle_serving_server.serve --model ${serving_server_dir_name} --op GeneralClasOp --port 9292 &"
            eval ${web_service_cpp_cmd}
            last_status=${PIPESTATUS[0]}
            status_check $last_status "${web_service_cpp_cmd}" "${status_log}" "${model_name}"
            sleep 5s
            _save_log_path="${LOG_PATH}/server_infer_cpp_cpu_pipeline_batchsize_1.log"
            pipeline_cmd="${python_} test_cpp_serving_client.py > ${_save_log_path} 2>&1 "
            eval ${pipeline_cmd}
            last_status=${PIPESTATUS[0]}
            eval "cat ${_save_log_path}"
            status_check ${last_status} "${pipeline_cmd}" "${status_log}" "${model_name}"
            eval "${python_} -m paddle_serving_server.serve stop"
            sleep 5s
        else
            web_service_cpp_cmd="${python_} -m paddle_serving_server.serve --model ${serving_server_dir_name} --op GeneralClasOp --port 9292 --gpu_id=${use_gpu} &"
            eval ${web_service_cpp_cmd}
            last_status=${PIPESTATUS[0]}
            status_check $last_status "${web_service_cpp_cmd}" "${status_log}" "${model_name}"
            sleep 8s

            _save_log_path="${LOG_PATH}/server_infer_cpp_gpu_pipeline_batchsize_1.log"
            pipeline_cmd="${python_} test_cpp_serving_client.py > ${_save_log_path} 2>&1 "
            eval ${pipeline_cmd}
            last_status=${PIPESTATUS[0]}
            eval "cat ${_save_log_path}"
            status_check ${last_status} "${pipeline_cmd}" "${status_log}" "${model_name}"
            sleep 5s
            eval "${python_} -m paddle_serving_server.serve stop"
        fi
    done
}


function func_serving_rec(){
    LOG_PATH="test_tipc/output/${model_name}"
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

    cp_prototxt_cmd="cp ./paddleserving/recognition/preprocess/general_PPLCNet_x2_5_lite_v1.0_serving/*.prototxt ${cls_serving_server_value}"
    eval ${cp_prototxt_cmd}
    cp_prototxt_cmd="cp ./paddleserving/recognition/preprocess/general_PPLCNet_x2_5_lite_v1.0_client/*.prototxt ${cls_serving_client_value}"
    eval ${cp_prototxt_cmd}
    cp_prototxt_cmd="cp ./paddleserving/recognition/preprocess/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/*.prototxt ${det_serving_client_value}"
    eval ${cp_prototxt_cmd}
    cp_prototxt_cmd="cp ./paddleserving/recognition/preprocess/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/*.prototxt ${det_serving_server_value}"
    eval ${cp_prototxt_cmd}

    prototxt_dataline=$(awk 'NR==1, NR==3{print}'  ${cls_serving_server_value}/serving_server_conf.prototxt)
    IFS=$'\n'
    prototxt_lines=(${prototxt_dataline})
    feed_var_name=$(func_parser_value "${prototxt_lines[2]}")
    IFS='|'

    cd ${serving_dir_value}
    unset https_proxy
    unset http_proxy

    # export SERVING_BIN=${PWD}/../Serving/server-build-gpu-opencv/core/general-server/serving
    for use_gpu in ${web_use_gpu_list[*]}; do
        if [ ${use_gpu} = "null" ]; then
            det_serving_server_dir_name=$(func_get_url_file_name "$det_serving_server_value")
            web_service_cpp_cmd="${python_interp} -m paddle_serving_server.serve --model ../../${det_serving_server_value} ../../${cls_serving_server_value} --op GeneralPicodetOp GeneralFeatureExtractOp --port 9400 &"
            eval ${web_service_cpp_cmd}
            last_status=${PIPESTATUS[0]}
            status_check $last_status "${web_service_cpp_cmd}" "${status_log}" "${model_name}"
            sleep 5s
            _save_log_path="${LOG_PATH}/server_infer_cpp_cpu_batchsize_1.log"
            pipeline_cmd="${python_interp} ${pipeline_py} > ${_save_log_path} 2>&1 "
            eval ${pipeline_cmd}
            last_status=${PIPESTATUS[0]}
            eval "cat ${_save_log_path}"
            status_check ${last_status} "${pipeline_cmd}" "${status_log}" "${model_name}"
            eval "${python_} -m paddle_serving_server.serve stop"
            sleep 5s
        else
            det_serving_server_dir_name=$(func_get_url_file_name "$det_serving_server_value")
            web_service_cpp_cmd="${python_interp} -m paddle_serving_server.serve --model ../../${det_serving_server_value} ../../${cls_serving_server_value} --op GeneralPicodetOp GeneralFeatureExtractOp --port 9400 --gpu_id=${use_gpu} &"
            eval ${web_service_cpp_cmd}
            last_status=${PIPESTATUS[0]}
            status_check $last_status "${web_service_cpp_cmd}" "${status_log}" "${model_name}"
            sleep 5s
            _save_log_path="${LOG_PATH}/server_infer_cpp_gpu_batchsize_1.log"
            pipeline_cmd="${python_interp} ${pipeline_py} > ${_save_log_path} 2>&1 "
            eval ${pipeline_cmd}
            last_status=${PIPESTATUS[0]}
            eval "cat ${_save_log_path}"
            status_check ${last_status} "${pipeline_cmd}" "${status_log}" "${model_name}"
            eval "${python_} -m paddle_serving_server.serve stop"
            sleep 5s
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
if [[ ${model_name} =~ "ShiTu" ]]; then
    func_serving_rec
else
    func_serving_cls
fi

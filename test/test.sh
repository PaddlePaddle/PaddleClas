#!/bin/bash
FILENAME=$1
# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer', 'infer']
MODE=$2
dataline=$(cat ${FILENAME})
# parser params
IFS=$'\n'
lines=(${dataline})
function func_parser_key(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[0]}
    echo ${tmp}
}
function func_parser_value(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}
function status_check(){
    last_status=$1   # the exit code
    run_command=$2
    run_log=$3
    if [ $last_status -eq 0 ]; then
        echo -e "\033[33m Run successfully with command - ${run_command}!  \033[0m" | tee -a ${run_log}
    else
        echo -e "\033[33m Run failed with command - ${run_command}!  \033[0m" | tee -a ${run_log}
    fi
}

IFS=$'\n'
# The training params
model_name_list=$(func_parser_value "${lines[1]}")
model_name_pact_list=$(func_parser_value "${lines[2]}")
model_name_fpgm_list=$(func_parser_value "${lines[3]}")
model_name_kl_list=$(func_parser_value "${lines[4]}")
python=$(func_parser_value "${lines[5]}")
gpu_list=$(func_parser_value "${lines[6]}")
epoch_key=$(func_parser_key "${lines[7]}")
epoch_value=$(func_parser_value "${lines[7]}")
save_model_key=$(func_parser_key "${lines[8]}")
save_model_value=$(func_parser_value "${lines[8]}")
pretrain_model_key=$(func_parser_key "${lines[9]}")
save_infer_key=$(func_parser_key "${lines[10]}")

#scripts
train_py=$(func_parser_value "${lines[20]}")
eval_py=$(func_parser_value "${lines[21]}")
norm_export=$(func_parser_value "${lines[22]}")
inference_py=$(func_parser_value "${lines[23]}")

#The inference params
use_gpu_key=$(func_parser_key "${lines[33]}")
use_gpu_list=$(func_parser_value "${lines[33]}")
use_mkldnn_key=$(func_parser_key "${lines[34]}")
use_mkldnn_list=$(func_parser_value "${lines[34]}")
cpu_threads_key=$(func_parser_key "${lines[35]}")
cpu_threads_list=$(func_parser_value "${lines[35]}")
batch_size_key=$(func_parser_key "${lines[36]}")
batch_size_list=$(func_parser_value "${lines[36]}")
use_trt_key=$(func_parser_key "${lines[37]}")
use_trt_list=$(func_parser_value "${lines[37]}")
precision_key=$(func_parser_key "${lines[38]}")
precision_list=$(func_parser_value "${lines[38]}")
infer_model_key=$(func_parser_key "${lines[39]}")
infer_model=$(func_parser_value "${lines[39]}")
image_dir_key=$(func_parser_key "${lines[40]}")
infer_img_dir=$(func_parser_value "${lines[40]}")
save_log_key=$(func_parser_key "${lines[32]}")

LOG_PATH="./test/output"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results.log"


function func_inference(){
    IFS='|'
    _python=$1
    _script=$2
    _model_dir=$3
    _log_path=$4
    _img_dir=$5
    _model_name=$6
    
    # inference 
    for use_gpu in ${use_gpu_list[*]}; do 
        if [ ${use_gpu} = "False" ]; then
            for use_mkldnn in ${use_mkldnn_list[*]}; do
                for threads in ${cpu_threads_list[*]}; do
                    for batch_size in ${batch_size_list[*]}; do
                        _save_log_path="${_log_path}/${_model_name}_infer_cpu_usemkldnn_${use_mkldnn}_threads_${threads}_batchsize_${batch_size}.log"
                        command="${_python} ${_script} -o ${use_gpu_key}=${use_gpu} -o ${use_mkldnn_key}=${use_mkldnn} -o ${cpu_threads_key}=${threads} -o ${infer_model_key}=${_model_dir} -o ${batch_size_key}=${batch_size} -o ${image_dir_key}=${_img_dir} -o ${save_log_key}=${_save_log_path} -o benchmark=True -o Global.model_name=${_model_name}"
                        eval $command
                        status_check $? "${command}" "${status_log}"
                    done
                done
            done
        else
            for use_trt in ${use_trt_list[*]}; do
                for precision in ${precision_list[*]}; do
                    if [ ${use_trt} = "False" ] && [ ${precision} != "fp32" ]; then
                        continue
                    fi
                    for batch_size in ${batch_size_list[*]}; do
                        _save_log_path="${_log_path}/${_model_name}_infer_gpu_usetrt_${use_trt}_precision_${precision}_batchsize_${batch_size}.log"
                        command="${_python} ${_script} -o ${use_gpu_key}=${use_gpu} -o ${use_trt_key}=${use_trt} -o ${precision_key}=${precision} -o ${infer_model_key}=${_model_dir} -o ${batch_size_key}=${batch_size} -o ${image_dir_key}=${_img_dir} -o ${save_log_key}=${_save_log_path}  -o benchmark=True -o Global.model_name=${_model_name}"
                        eval $command
                        status_check $? "${command}" "${status_log}"
                    done
                done
            done
        fi
    done
}

if [ ${MODE} != "infer" ]; then

IFS="|"
for gpu in ${gpu_list[*]}; do
    use_gpu=True
    if [ ${gpu} = "-1" ];then
	use_gpu=False
        env=""
    elif [ ${#gpu} -le 1 ];then
        env="export CUDA_VISIBLE_DEVICES=${gpu}"
        eval ${env}
    elif [ ${#gpu} -le 15 ];then
        IFS=","
        array=(${gpu})
        env="export CUDA_VISIBLE_DEVICES=${array[0]}"
        IFS="|"
    else
        IFS=";"
        array=(${gpu})
        ips=${array[0]}
        gpu=${array[1]}
        IFS="|"
        env=" "
    fi
    for model_name in ${model_name_list[*]}; do 
        # not set epoch when whole_train_infer
        if [ ${MODE} != "whole_train_infer" ]; then
            set_epoch="-o ${epoch_key}=${epoch_num}"
        else
            set_epoch=" "
        fi
        save_log="${LOG_PATH}/${model_name}_gpus_${gpu}"
	# train with cpu
	if [ ${gpu} = "-1" ];then
            cmd="${python} ${train_py} -o Arch.name=${model_name} -o Global.device=cpu -o ${save_model_key}=${save_log} ${set_epoch}"
	# train with single gpu
        elif [ ${#gpu} -le 2 ];then  # train with single gpu
            cmd="${python} ${train_py} -o Arch.name=${model_name} -o ${save_model_key}=${save_log} ${set_epoch}"
        elif [ ${#gpu} -le 15 ];then  # train with multi-gpu
            cmd="${python} -m paddle.distributed.launch --gpus=${gpu} ${train_py} -o Arch.name=${model_name} -o ${save_model_key}=${save_log}  ${set_epoch}"
        else     # train with multi-machine
		cmd="${python} -m paddle.distributed.launch --ips=${ips} --gpus=${gpu} ${train_py} -o Arch.name=${model_name} -c ${save_model_key}=${save_log} ${set_epoch}"
        fi
        # run train
        eval $cmd
        status_check $? "${cmd}" "${status_log}"

        # run eval
        eval_cmd="${python} ${eval_py} -o Arch.name=${model_name} -o ${pretrain_model_key}=${save_log}/${model_name}/latest" 
        eval $eval_cmd
        status_check $? "${eval_cmd}" "${status_log}"

        # run export model
        save_infer_path="${save_log}/inference"
        export_cmd="${python} ${norm_export} -o Arch.name=${model_name} -o ${pretrain_model_key}=${save_log}/${model_name}/latest -o ${save_infer_key}=${save_infer_path}"
        eval $export_cmd
        status_check $? "${export_cmd}" "${status_log}"

        #run inference
        eval $env
        save_infer_path="${save_log}/inference"
	cd deploy
        func_inference "${python}" "${inference_py}" "../${save_infer_path}" "../${LOG_PATH}" "../${infer_img_dir}" "${model_name}"
        eval "unset CUDA_VISIBLE_DEVICES"
	cd ..
    done
done

else
    GPUID=$3
    if [ ${#GPUID} -le 0 ];then
        env=" "
    else
        env="export CUDA_VISIBLE_DEVICES=${GPUID}"
    fi
    echo $env
    # export inference model
    mkdir -p inference_models
    for model_name in ${model_name_list[*]}; do
        export_cmd="${python} ${norm_export} -o Arch.name=${model_name} -o ${pretrain_model_key}=pretrained_models/${model_name}_pretrained -o ${save_infer_key}=./inference_models/${model_name}"
	eval $export_cmd
    done
    #run inference
    cd deploy
    for model_name in ${model_name_list[*]}; do
        func_inference "${python}" "${inference_py}" "../inference_models/${model_name}" "../${LOG_PATH}" "../${infer_img_dir}" "${model_name}"
    done
    cd ..
fi

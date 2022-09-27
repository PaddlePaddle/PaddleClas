#!/usr/bin/env bash
GPU_IDS="0,1,2,3"

# Basic Config
CONFIG="ppcls/configs/cls_demo/person/PPLCNet/PPLCNet_x1_0.yaml"
EPOCHS=1
OUTPUT="output_debug4"
STATUS_LOG="${OUTPUT}/status_result.log"
RESULT="${OUTPUT}/result.log"


# Search Options
LR_LIST=( 0.0075 0.01 0.0125 )
RESOLUTION_LIST=( 176 192 224 )
RA_PROB_LIST=( 0.0 0.1 0.5 )
RE_PROB_LIST=( 0.0 0.1 0.5 )
LR_MULT_LIST=( [0.0,0.2,0.4,0.6,0.8,1.0] [0.0,0.4,0.4,0.8,0.8,1.0] )
TEACHER_LIST=( "ResNet101_vd" "ResNet50_vd" )


# Train Mode
declare -A MODE_MAP
MODE_MAP=(["search_lr"]=1 ["search_resolution"]=1 ["search_ra_prob"]=1 ["search_re_prob"]=1 ["search_lr_mult_list"]=1  ["search_teacher"]=1 ["train_distillation_model"]=1)

export CUDA_VISIBLE_DEVICES=${GPU_IDS}


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


function get_max_value(){
    array=($*)
    max=${array[0]}
    index=0
    for (( i=0; i<${#array[*]-1}; i++ )); do
        if [[ $(echo "${array[$i]} > $max"|bc) -eq 1 ]]; then
            max=${array[$i]}
            index=${i}
        else
            continue
        fi
    done
    echo ${max}
    echo ${index}
}

function get_best_info(){
    _parameter=$1
    params_index=2
    if [[ ${_parameter} == "TEACHER" ]]; then
        params_index=3
    fi
    parameters_list=$(find ${OUTPUT}/${_parameter}* -name train.log  | awk -v params_index=${params_index} -F "/" '{print $params_index}')
    metric_list=$(find ${OUTPUT}/${_parameter}* -name train.log | xargs cat | grep "best" | grep "Epoch ${EPOCHS}" | awk -F " " '{print substr($NF,0,7)}')
    best_info=$(get_max_value ${metric_list[*]})
    best_metric=$(echo $best_info | awk -F " " '{print $1}')
    best_index=$(echo $best_info | awk -F " " '{print $2}')
    best_parameter=$(echo $parameters_list | awk -v best=$(($best_index+1))  '{print $best}' | awk -F "_" '{print $2}')
    echo ${best_metric}
    echo ${best_parameter}
}


function search_lr(){
    for lr in ${LR_LIST[*]}; do
        cmd_train="python3.7 -m paddle.distributed.launch --gpus=${GPU_IDS} tools/train.py \
                       -c ${CONFIG} \
                       -o Global.output_dir=${OUTPUT}/LR_${lr} \
                       -o Optimizer.lr.learning_rate=${lr} \
                       -o Global.epochs=${EPOCHS}"
        eval ${cmd_train}
        status_check $? "${cmd_train}" "${STATUS_LOG}"
        cmd="find ${OUTPUT} -name epoch* | xargs rm -rf"
        eval ${cmd}
    done
}


function search_resolution(){
    _lr=$1
    for resolution in ${RESOLUTION_LIST[*]}; do
        cmd_train="python3.7 -m paddle.distributed.launch --gpus=${GPU_IDS} tools/train.py \
            -c ${CONFIG} \
            -o Global.output_dir=${OUTPUT}/RESOLUTION_${resolution} \
            -o Optimizer.lr.learning_rate=${_lr} \
            -o Global.epochs=${EPOCHS} \
            -o DataLoader.Train.dataset.transform_ops.1.RandCropImage.size=${resolution}"
        eval ${cmd_train}
        status_check $? "${cmd_train}" "${STATUS_LOG}"
        cmd="find ${OUTPUT} -name epoch* | xargs rm -rf"
        eval ${cmd}
    done
}



function search_ra_prob(){
    _lr=$1
    _resolution=$2
    for ra_prob in ${RA_PROB_LIST[*]}; do
        cmd_train="python3.7 -m paddle.distributed.launch --gpus=${GPU_IDS} tools/train.py \
            -c ${CONFIG} \
            -o Global.output_dir=${OUTPUT}/RA_${ra_prob} \
            -o Optimizer.lr.learning_rate=${_lr} \
            -o Global.epochs=${EPOCHS} \
            -o DataLoader.Train.dataset.transform_ops.3.TimmAutoAugment.prob=${ra_prob} \
            -o DataLoader.Train.dataset.transform_ops.1.RandCropImage.size=${_resolution} \
            -o DataLoader.Train.dataset.transform_ops.3.TimmAutoAugment.img_size=${_resolution}"
        eval ${cmd_train}
        status_check $? "${cmd_train}" "${STATUS_LOG}"
        cmd="find ${OUTPUT} -name epoch* | xargs rm -rf"
        eval ${cmd}
    done
}



function search_re_prob(){
    _lr=$1
    _resolution=$2
    _ra_prob=$3
    for re_prob in ${RE_PROB_LIST[*]}; do
        cmd_train="python3.7 -m paddle.distributed.launch --gpus=${GPU_IDS} tools/train.py \
            -c ${CONFIG} \
            -o Global.output_dir=${OUTPUT}/RE_${re_prob} \
            -o Optimizer.lr.learning_rate=${_lr} \
            -o Global.epochs=${EPOCHS} \
            -o DataLoader.Train.dataset.transform_ops.3.TimmAutoAugment.prob=${_ra_prob} \
            -o DataLoader.Train.dataset.transform_ops.5.RandomErasing.EPSILON=${re_prob} \
            -o DataLoader.Train.dataset.transform_ops.1.RandCropImage.size=${_resolution} \
            -o DataLoader.Train.dataset.transform_ops.3.TimmAutoAugment.img_size=${_resolution}"
        eval ${cmd_train}
        status_check $? "${cmd_train}" "${STATUS_LOG}"
        cmd="find ${OUTPUT} -name epoch* | xargs rm -rf"
        eval ${cmd}
    done
}


function search_lr_mult_list(){
    _lr=$1
    _resolution=$2
    _ra_prob=$3
    _re_prob=$4

    for lr_mult in ${LR_MULT_LIST[*]}; do
        cmd_train="python3.7 -m paddle.distributed.launch --gpus=${GPU_IDS} tools/train.py \
            -c ${CONFIG} \
            -o Global.output_dir=${OUTPUT}/LR_MULT_${lr_mult} \
            -o Optimizer.lr.learning_rate=${_lr} \
            -o Global.epochs=${EPOCHS} \
            -o DataLoader.Train.dataset.transform_ops.3.TimmAutoAugment.prob=${_ra_prob} \
            -o DataLoader.Train.dataset.transform_ops.5.RandomErasing.EPSILON=${_re_prob} \
            -o DataLoader.Train.dataset.transform_ops.1.RandCropImage.size=${_resolution} \
            -o DataLoader.Train.dataset.transform_ops.3.TimmAutoAugment.img_size=${_resolution} \
            -o Arch.lr_mult_list=${lr_mult}"
        eval ${cmd_train}
        status_check $? "${cmd_train}" "${STATUS_LOG}"
        cmd="find ${OUTPUT} -name epoch* | xargs rm -rf"
        eval ${cmd}
    done

}


function search_teacher(){
    _lr=$1
    _resolution=$2
    _ra_prob=$3
    _re_prob=$4

    for teacher in ${TEACHER_LIST[*]}; do
        cmd_train="python3.7 -m paddle.distributed.launch --gpus=${GPU_IDS} tools/train.py \
            -c ${CONFIG} \
            -o Global.output_dir=${OUTPUT}/TEACHER_${teacher} \
            -o Optimizer.lr.learning_rate=${_lr} \
            -o Global.epochs=${EPOCHS} \
            -o DataLoader.Train.dataset.transform_ops.3.TimmAutoAugment.prob=${_ra_prob} \
            -o DataLoader.Train.dataset.transform_ops.5.RandomErasing.EPSILON=${_re_prob} \
            -o DataLoader.Train.dataset.transform_ops.1.RandCropImage.size=${_resolution} \
            -o DataLoader.Train.dataset.transform_ops.3.TimmAutoAugment.img_size=${_resolution} \
            -o Arch.name=${teacher}"
        eval ${cmd_train}
        status_check $? "${cmd_train}" "${STATUS_LOG}"
        cmd="find ${OUTPUT}/* -name epoch* | xargs rm -rf"
        eval ${cmd}
    done
}


# train the model for knowledge distillation
function train_distillation_model(){
    _lr=$1
    _resolution=$2
    _ra_prob=$3
    _re_prob=$4
    _lr_mult=$5
    teacher=$6
    t_pretrained_model="${OUTPUT}/TEACHER_${teacher}/${teacher}/best_model"
    config="ppcls/configs/cls_demo/person/Distillation/PPLCNet_x1_0_distillation.yaml"
    combined_label_list="./dataset/person/train_list_for_distill.txt"

    cmd_train="python3.7 -m paddle.distributed.launch \
        --gpus=${GPU_IDS} \
        tools/train.py -c ${config} \
        -o Global.output_dir=${OUTPUT}/kd_teacher \
        -o Optimizer.lr.learning_rate=${_lr} \
        -o Global.epochs=${EPOCHS} \
        -o DataLoader.Train.dataset.transform_ops.3.TimmAutoAugment.prob=${_ra_prob} \
        -o DataLoader.Train.dataset.transform_ops.5.RandomErasing.EPSILON=${_re_prob} \
        -o DataLoader.Train.dataset.transform_ops.1.RandCropImage.size=${_resolution} \
        -o DataLoader.Train.dataset.transform_ops.3.TimmAutoAugment.img_size=${_resolution} \
        -o DataLoader.Train.dataset.cls_label_path=${combined_label_list} \
        -o Arch.models.0.Teacher.name="${teacher}" \
        -o Arch.models.0.Teacher.pretrained="${t_pretrained_model}" \
        -o Arch.models.1.Student.lr_mult_list=${_lr_mult}"
    eval ${cmd_train}
    status_check $? "${cmd_train}" "${STATUS_LOG}"
    cmd="find ${OUTPUT} -name epoch* | xargs rm -rf"
    eval ${cmd}
}

######## Train PaddleClas  ########
rm -rf ${OUTPUT}

# Train and get best lr
best_lr=0.01
if [[ ${MODE_MAP["search_lr"]} -eq 1 ]]; then
    search_lr
    best_info=$(get_best_info "LR_[0-9]")
    best_metric=$(echo $best_info | awk -F " " '{print $1}')
    best_lr=$(echo $best_info | awk -F " " '{print $2}')
    echo "The best lr is ${best_lr}, and the best metric is ${best_metric}" >> ${RESULT}
fi

# Train and get best resolution
best_resolution=192
if [[ ${MODE_MAP["search_resolution"]} -eq 1 ]]; then
    search_resolution "${best_lr}"
    best_info=$(get_best_info "RESOLUTION")
    best_metric=$(echo $best_info | awk -F " " '{print $1}')
    best_resolution=$(echo $best_info | awk -F " " '{print $2}')
    echo "The best resolution is ${best_resolution}, and the best metric is ${best_metric}" >> ${RESULT}
fi

# Train and get best ra_prob
best_ra_prob=0.0
if [[ ${MODE_MAP["search_ra_prob"]} -eq 1 ]]; then
    search_ra_prob "${best_lr}" "${best_resolution}"
    best_info=$(get_best_info "RA")
    best_metric=$(echo $best_info | awk -F " " '{print $1}')
    best_ra_prob=$(echo $best_info | awk -F " " '{print $2}')
    echo "The best ra_prob is ${best_ra_prob}, and the best metric is ${best_metric}" >> ${RESULT}
fi

# Train and get best re_prob
best_re_prob=0.1
if [[ ${MODE_MAP["search_re_prob"]} -eq 1 ]]; then
    search_re_prob "${best_lr}" "${best_resolution}" "${best_ra_prob}"
    best_info=$(get_best_info "RE")
    best_metric=$(echo $best_info | awk -F " " '{print $1}')
    best_re_prob=$(echo $best_info | awk -F " " '{print $2}')
   echo "The best re_prob is ${best_re_prob}, and the best metric is ${best_metric}" >> ${RESULT}
fi

# Train and get best lr_mult_list
best_lr_mult_list=[1.0,1.0,1.0,1.0,1.0,1.0]
if [[ ${MODE_MAP["search_lr_mult_list"]} -eq 1 ]]; then
    search_lr_mult_list "${best_lr}" "${best_resolution}" "${best_ra_prob}" "${best_re_prob}"
    best_info=$(get_best_info "LR_MULT")
    best_metric=$(echo $best_info | awk -F " " '{print $1}')
    best_lr_mult_list=$(echo $best_info | awk -F " " '{print $2}')
    echo "The best lr_mult_list is ${best_lr_mult_list}, and the best metric is ${best_metric}" >> ${RESULT}
fi

# train and get best teacher
best_teacher="ResNet101_vd"
if [[ ${MODE_MAP["search_teacher"]} -eq 1 ]]; then
    search_teacher "${best_lr}" "${best_resolution}" "${best_ra_prob}" "${best_re_prob}"
    best_info=$(get_best_info "TEACHER")
    best_metric=$(echo $best_info | awk -F " " '{print $1}')
    best_teacher=$(echo $best_info | awk -F " " '{print $2}')
    echo "The best teacher is ${best_teacher}, and the best metric is ${best_metric}" >> ${RESULT}
fi

# train the distillation model
if [[ ${MODE_MAP["train_distillation_model"]} -eq 1 ]]; then
    train_distillation_model "${best_lr}" "${best_resolution}" "${best_ra_prob}" "${best_re_prob}" "${best_lr_mult_list}" ${best_teacher}
    best_info=$(get_best_info "kd_teacher/DistillationModel")
    best_metric=$(echo $best_info | awk -F " " '{print $1}')
    echo "the distillation best metric is ${best_metric}, it is global best metric!" >> ${RESULT}
fi


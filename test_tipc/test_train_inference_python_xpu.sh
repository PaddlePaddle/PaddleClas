#!/bin/bash
source test_tipc/common_func.sh

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

function func_parser_config() {
    strs=$1
    IFS=" "
    array=(${strs})
    tmp=${array[2]}
    echo ${tmp}
}

BASEDIR=$(dirname "$0")
REPO_ROOT_PATH=$(readlinkf ${BASEDIR}/../)

FILENAME=$1

# change gpu to xpu in tipc txt configs
sed -i "s/Global.device:gpu/Global.device:xpu/g" $FILENAME
sed -i "s/Global.use_gpu/Global.use_xpu/g" $FILENAME
sed -i "s/Global.use_tensorrt:True|False/Global.use_tensorrt:False/g" $FILENAME
sed -i "s/Global.save_interval=2/Global.save_interval=1/g" $FILENAME
sed -i "s/-o Global.epochs:lite_train_lite_infer=2/-o Global.epochs:lite_train_lite_infer=1/g" $FILENAME
sed -i "s/enable_mkldnn:True|False/enable_mkldnn:False/g" $FILENAME
# python has been updated to version 3.9 for xpu backend
sed -i "s/python3.7/python3.9/g" $FILENAME
sed -i "s/python3.10/python3.9/g" $FILENAME

modelname=$(echo $FILENAME | cut -d '/' -f3)
if  [ $modelname == "PVTV2" ] || [ $modelname == "Twins" ] || [ $modelname == "SwinTransformer" ]; then
    sed -i "s/gpu_list:0|0,1/gpu_list:0,1/g" $FILENAME
fi

dataline=`cat $FILENAME`

# parser params
IFS=$'\n'
lines=(${dataline})

# replace inference config file
inference_py=$(func_parser_value "${lines[39]}")
inference_config=$(func_parser_config ${inference_py})
sed -i 's/use_gpu: True/use_xpu: True/g' "$REPO_ROOT_PATH/deploy/$inference_config"

# replace training config file
grep -n 'tools/.*yaml' $FILENAME  | cut -d ":" -f 1 \
| while read line_num ; do 
    train_cmd=$(func_parser_value "${lines[line_num-1]}")
    trainer_config=$(func_parser_config ${train_cmd})
    sed -i 's/device: gpu/device: xpu/g' "$REPO_ROOT_PATH/$trainer_config"
done

# change gpu to xpu in execution script
sed -i "s/\"gpu\"/\"xpu\"/g" test_tipc/test_train_inference_python.sh
sed -i "s/\${model_name}\/train.log/train.log/g" test_tipc/test_train_inference_python.sh
sed -i "s/\${model_name}\/\${train_model_name}/\${train_model_name}/g" test_tipc/test_train_inference_python.sh

# pass parameters to test_train_inference_python.sh
cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} $2"
echo $cmd
eval $cmd

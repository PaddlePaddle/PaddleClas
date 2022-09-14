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

# pass parameters to test_train_inference_python.sh
cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} $2"
echo $cmd
eval $cmd

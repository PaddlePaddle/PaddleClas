#!/bin/bash
source test_tipc/common_func.sh

BASIC_CONFIG="./config.txt"
CONFIG=$1

# parser tipc config
IFS=$'\n'
TIPC_CONFIG=$1
tipc_dataline=$(cat $TIPC_CONFIG)
tipc_lines=(${tipc_dataline})

runtime_device=$(func_parser_value_lite "${tipc_lines[0]}" ":")
lite_arm_work_path=$(func_parser_value_lite "${tipc_lines[1]}" ":")
lite_arm_so_path=$(func_parser_value_lite "${tipc_lines[2]}" ":")
clas_model_name=$(func_parser_value_lite "${tipc_lines[3]}" ":")
inference_cmd=$(func_parser_value_lite "${tipc_lines[4]}" ":")
num_threads_list=$(func_parser_value_lite "${tipc_lines[5]}" ":")
batch_size_list=$(func_parser_value_lite "${tipc_lines[6]}" ":")
precision_list=$(func_parser_value_lite "${tipc_lines[7]}" ":")


# Prepare config and test.sh
work_path="./deploy/lite"
cp ${CONFIG} ${work_path}
cp test_tipc/test_lite_arm_cpu_cpp.sh ${work_path}

# Prepare model
cd ${work_path}
pip3 install paddlelite==2.10
model_url="https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/${clas_model_name}_infer.tar"
wget --no-proxy ${model_url}
model_tar=$(echo ${model_url} | awk -F "/" '{print $NF}')
tar -xf ${model_tar}
paddle_lite_opt --model_dir=${clas_model_name}_infer --model_file=${clas_model_name}_infer/inference.pdmodel --param_file=${clas_model_name}_infer/inference.pdiparams --valid_targets=arm --optimize_out=${clas_model_name}
rm -rf ${clas_model_name}_infer*

# Prepare paddlelite library
paddlelite_lib_url="https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv8.clang.c++_static.with_extra.with_cv.tar.gz"
wget ${paddlelite_lib_url}
paddlelite_lib_file=$(echo ${paddlelite_lib_url} | awk -F "/" '{print $NF}')
tar -xzf ${paddlelite_lib_file}
mv ${paddlelite_lib_file%*.tar.gz} inference_lite_lib.android.armv8
rm -rf ${paddlelite_lib_file%*.tar.gz}*

# Compile and obtain executable binary file
git clone https://github.com/LDOUBLEV/AutoLog.git
make

# push executable binary, library, lite model, data, etc. to arm device
adb shell mkdir -p ${lite_arm_work_path}
adb push $(echo ${inference_cmd} | awk '{print $1}') ${lite_arm_work_path}
adb shell chmod +x ${lite_arm_work_path}/$(echo ${inference_cmd} | awk '{print $1}')
adb push ${lite_arm_so_path} ${lite_arm_work_path}
adb push ${clas_model_name}.nb ${lite_arm_work_path}
adb push ${BASIC_CONFIG} ${lite_arm_work_path}
adb push ../../ppcls/utils/imagenet1k_label_list.txt ${lite_arm_work_path}
adb push imgs/$(echo ${inference_cmd} | awk '{print $3}') ${lite_arm_work_path}

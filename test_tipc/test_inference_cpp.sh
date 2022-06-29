#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
MODE=$2

# set cuda device
GPUID=$3
if [[ ! $GPUID ]];then
   GPUID=0
fi
env="export CUDA_VISIBLE_DEVICES=${GPUID}"
set CUDA_VISIBLE_DEVICES
eval $env

dataline=$(awk 'NR==1, NR==19{print}'  $FILENAME)

# parser params
IFS=$'\n'
lines=(${dataline})
# parser cpp inference model
model_name=$(func_parser_value "${lines[1]}")
cpp_infer_type=$(func_parser_value "${lines[2]}")
cpp_infer_model_dir=$(func_parser_value "${lines[3]}")
cpp_det_infer_model_dir=$(func_parser_value "${lines[4]}")
cpp_infer_is_quant=$(func_parser_value "${lines[7]}")
# parser cpp inference
inference_cmd=$(func_parser_value "${lines[8]}")
cpp_use_gpu_list=$(func_parser_value "${lines[9]}")
cpp_use_mkldnn_list=$(func_parser_value "${lines[10]}")
cpp_cpu_threads_list=$(func_parser_value "${lines[11]}")
cpp_batch_size_list=$(func_parser_value "${lines[12]}")
cpp_use_trt_list=$(func_parser_value "${lines[13]}")
cpp_precision_list=$(func_parser_value "${lines[14]}")
cpp_image_dir_value=$(func_parser_value "${lines[15]}")
cpp_benchmark_value=$(func_parser_value "${lines[16]}")
generate_yaml_cmd=$(func_parser_value "${lines[17]}")
transform_index_cmd=$(func_parser_value "${lines[18]}")

LOG_PATH="./test_tipc/output/${model_name}/${MODE}"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_cpp.log"
# generate_yaml_cmd="python3 test_tipc/generate_cpp_yaml.py"

function func_shitu_cpp_inference(){
    IFS='|'
    _script=$1
    _model_dir=$2
    _log_path=$3
    _img_dir=$4
    _flag_quant=$5
    # inference

    for use_gpu in ${cpp_use_gpu_list[*]}; do
        if [ ${use_gpu} = "False" ] || [ ${use_gpu} = "cpu" ]; then
            for use_mkldnn in ${cpp_use_mkldnn_list[*]}; do
                if [ ${use_mkldnn} = "False" ] && [ ${_flag_quant} = "True" ]; then
                    continue
                fi
                for threads in ${cpp_cpu_threads_list[*]}; do
                    for batch_size in ${cpp_batch_size_list[*]}; do
                        precision="fp32"
                        if [ ${use_mkldnn} = "False" ] && [ ${_flag_quant} = "True" ]; then
                            precison="int8"
                        fi
                        _save_log_path="${_log_path}/cpp_infer_cpu_usemkldnn_${use_mkldnn}_threads_${threads}_precision_${precision}_batchsize_${batch_size}.log"
                        eval $transform_index_cmd
                        command="${generate_yaml_cmd} --type shitu --batch_size ${batch_size} --mkldnn ${use_mkldnn} --gpu ${use_gpu} --cpu_thread ${threads} --tensorrt False --precision ${precision} --data_dir ${_img_dir} --benchmark True --cls_model_dir ${cpp_infer_model_dir} --det_model_dir ${cpp_det_infer_model_dir} --gpu_id ${GPUID}"
                        eval $command
                        command="${_script} > ${_save_log_path} 2>&1"
                        eval $command
                        last_status=${PIPESTATUS[0]}
                        status_check $last_status "${command}" "${status_log}" "${model_name}"
                    done
                done
            done
        elif [ ${use_gpu} = "True" ] || [ ${use_gpu} = "gpu" ]; then
            for use_trt in ${cpp_use_trt_list[*]}; do
                for precision in ${cpp_precision_list[*]}; do
                    if [[ ${_flag_quant} = "False" ]] && [[ ${precision} =~ "int8" ]]; then
                        continue
                    fi
                    if [[ ${precision} =~ "fp16" || ${precision} =~ "int8" ]] && [ ${use_trt} = "False" ]; then
                        continue
                    fi
                    if [[ ${use_trt} = "False" || ${precision} =~ "int8" ]] && [ ${_flag_quant} = "True" ]; then
                        continue
                    fi
                    for batch_size in ${cpp_batch_size_list[*]}; do
                        _save_log_path="${_log_path}/cpp_infer_gpu_usetrt_${use_trt}_precision_${precision}_batchsize_${batch_size}.log"
                        eval $transform_index_cmd
                        command="${generate_yaml_cmd} --type shitu --batch_size ${batch_size} --mkldnn False --gpu ${use_gpu} --cpu_thread 1 --tensorrt ${use_trt} --precision ${precision} --data_dir ${_img_dir} --benchmark True --cls_model_dir ${cpp_infer_model_dir} --det_model_dir ${cpp_det_infer_model_dir} --gpu_id ${GPUID}"
                        eval $command
                        command="${_script} > ${_save_log_path} 2>&1"
                        eval $command
                        last_status=${PIPESTATUS[0]}
                        status_check $last_status "${command}" "${status_log}" "${model_name}"
                    done
                done
            done
        else
            echo "Does not support hardware other than CPU and GPU Currently!"
        fi
    done
}

function func_cls_cpp_inference(){
    IFS='|'
    _script=$1
    _model_dir=$2
    _log_path=$3
    _img_dir=$4
    _flag_quant=$5
    # inference

    for use_gpu in ${cpp_use_gpu_list[*]}; do
        if [ ${use_gpu} = "False" ] || [ ${use_gpu} = "cpu" ]; then
            for use_mkldnn in ${cpp_use_mkldnn_list[*]}; do
                if [ ${use_mkldnn} = "False" ] && [ ${_flag_quant} = "True" ]; then
                    continue
                fi
                for threads in ${cpp_cpu_threads_list[*]}; do
                    for batch_size in ${cpp_batch_size_list[*]}; do
                        precision="fp32"
                        if [ ${use_mkldnn} = "False" ] && [ ${_flag_quant} = "True" ]; then
                            precison="int8"
                        fi
                        _save_log_path="${_log_path}/cpp_infer_cpu_usemkldnn_${use_mkldnn}_threads_${threads}_precision_${precision}_batchsize_${batch_size}.log"

                        command="${generate_yaml_cmd} --type cls --batch_size ${batch_size} --mkldnn ${use_mkldnn} --gpu ${use_gpu} --cpu_thread ${threads} --tensorrt False --precision ${precision} --data_dir ${_img_dir} --benchmark True --cls_model_dir ${cpp_infer_model_dir} --gpu_id ${GPUID}"
                        eval $command
                        command1="${_script} > ${_save_log_path} 2>&1"
                        eval ${command1}
                        last_status=${PIPESTATUS[0]}
                        status_check $last_status "${command1}" "${status_log}" "${model_name}"
                    done
                done
            done
        elif [ ${use_gpu} = "True" ] || [ ${use_gpu} = "gpu" ]; then
            for use_trt in ${cpp_use_trt_list[*]}; do
                for precision in ${cpp_precision_list[*]}; do
                    if [[ ${_flag_quant} = "False" ]] && [[ ${precision} =~ "int8" ]]; then
                        continue
                    fi
                    if [[ ${precision} =~ "fp16" || ${precision} =~ "int8" ]] && [ ${use_trt} = "False" ]; then
                        continue
                    fi
                    if [[ ${use_trt} = "False" || ${precision} =~ "int8" ]] && [ ${_flag_quant} = "True" ]; then
                        continue
                    fi
                    for batch_size in ${cpp_batch_size_list[*]}; do
                        _save_log_path="${_log_path}/cpp_infer_gpu_usetrt_${use_trt}_precision_${precision}_batchsize_${batch_size}.log"
                        command="${generate_yaml_cmd} --type cls --batch_size ${batch_size} --mkldnn False --gpu ${use_gpu} --cpu_thread 1 --tensorrt ${use_trt} --precision ${precision} --data_dir ${_img_dir} --benchmark True --cls_model_dir ${cpp_infer_model_dir} --gpu_id ${GPUID}"
                        eval $command
                        command="${_script} > ${_save_log_path} 2>&1"
                        eval $command
                        last_status=${PIPESTATUS[0]}
                        status_check $last_status "${command}" "${status_log}" "${model_name}"
                    done
                done
            done
        else
            echo "Does not support hardware other than CPU and GPU Currently!"
        fi
    done
}


if [[ $cpp_infer_type == "cls" ]]; then
   cd deploy/cpp
elif [[ $cpp_infer_type == "shitu" ]]; then
   cd deploy/cpp_shitu
else
   echo "Only support cls and shitu"
   exit 0
fi

if [[ $cpp_infer_type == "shitu" ]]; then
    echo "################### update cmake ###################"
    wget -nc  https://github.com/Kitware/CMake/releases/download/v3.22.0/cmake-3.22.0.tar.gz
    tar xf cmake-3.22.0.tar.gz
    cd ./cmake-3.22.0
    export root_path=$PWD
    export install_path=${root_path}/cmake
    eval "./bootstrap --prefix=${install_path}"
    make -j
    make install
    export PATH=${install_path}/bin:$PATH
    cd ..
    echo "################### update cmake done ###################"

    echo "################### build faiss ###################"
    apt-get install -y libopenblas-dev
    git clone https://github.com/facebookresearch/faiss.git
    cd faiss
    export faiss_install_path=$PWD/faiss_install
    eval "cmake -B build . -DFAISS_ENABLE_PYTHON=OFF  -DCMAKE_INSTALL_PREFIX=${faiss_install_path}"
    make -C build -j faiss
    make -C build install
    cd ..
fi

echo "################### build PaddleClas demo ####################"
# pwd = /workspace/hesensen/PaddleClas/deploy/cpp_shitu
OPENCV_DIR=$(dirname $PWD)/cpp/opencv-3.4.7/opencv3/
LIB_DIR=$(dirname $PWD)/cpp/paddle_inference/

CUDA_LIB_DIR=$(dirname `find /usr -name libcudart.so`)
CUDNN_LIB_DIR=$(dirname `find /usr -name libcudnn.so`)

BUILD_DIR=build
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
if [[ $cpp_infer_type == cls ]]; then
    cmake .. \
	-DPADDLE_LIB=${LIB_DIR} \
	-DWITH_MKL=ON \
	-DWITH_GPU=ON \
	-DWITH_STATIC_LIB=OFF \
	-DWITH_TENSORRT=OFF \
	-DOPENCV_DIR=${OPENCV_DIR} \
	-DCUDNN_LIB=${CUDNN_LIB_DIR} \
	-DCUDA_LIB=${CUDA_LIB_DIR} \
	-DTENSORRT_DIR=${TENSORRT_DIR}
else
    cmake ..\
	-DPADDLE_LIB=${LIB_DIR} \
	-DWITH_MKL=ON \
	-DWITH_GPU=ON \
	-DWITH_STATIC_LIB=OFF \
	-DWITH_TENSORRT=OFF \
	-DOPENCV_DIR=${OPENCV_DIR} \
	-DCUDNN_LIB=${CUDNN_LIB_DIR} \
	-DCUDA_LIB=${CUDA_LIB_DIR} \
	-DTENSORRT_DIR=${TENSORRT_DIR} \
	-DFAISS_DIR=${faiss_install_path} \
	-DFAISS_WITH_MKL=OFF
fi
make -j
cd ../../../
# cd ../../
echo "################### build PaddleClas demo finished ###################"

echo "################### run test ###################"
export Count=0
IFS="|"
infer_quant_flag=(${cpp_infer_is_quant})
for infer_model in ${cpp_infer_model_dir[*]}; do
    #run inference
    is_quant=${infer_quant_flag[Count]}
    if [[ $cpp_infer_type == "cls" ]]; then
        func_cls_cpp_inference "${inference_cmd}" "${infer_model}" "${LOG_PATH}" "${cpp_image_dir_value}" ${is_quant}
    else
        func_shitu_cpp_inference "${inference_cmd}" "${infer_model}" "${LOG_PATH}" "${cpp_image_dir_value}" ${is_quant}
    fi
    Count=$(($Count + 1))
done

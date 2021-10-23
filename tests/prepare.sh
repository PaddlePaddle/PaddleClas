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
    if [ ${#array[*]} = 2 ]; then
        echo ${array[1]}
    else
    	IFS="|"
    	tmp="${array[1]}:${array[2]}"
        echo ${tmp}
    fi
}
model_name=$(func_parser_value "${lines[1]}")
model_url_value=$(func_parser_value "${lines[35]}")
model_url_key=$(func_parser_key "${lines[35]}")

if [ ${MODE} = "lite_train_infer" ] || [ ${MODE} = "whole_infer" ];then
    # pretrain lite train data
    cd dataset
    rm -rf ILSVRC2012
    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_little_train.tar
    tar xf whole_chain_little_train.tar
    ln -s whole_chain_little_train ILSVRC2012
    cd ILSVRC2012 
    mv train.txt train_list.txt
    mv val.txt val_list.txt
    if [ ${MODE} = "lite_train_infer" ];then
	cp -r train/* val/
    fi
    cd ../../
elif [ ${MODE} = "infer" ] || [ ${MODE} = "cpp_infer" ];then
    # download data
    cd dataset
    rm -rf ILSVRC2012
    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_infer.tar
    tar xf whole_chain_infer.tar
    ln -s whole_chain_infer ILSVRC2012
    cd ILSVRC2012 
    mv val.txt val_list.txt
    ln -s val_list.txt train_list.txt
    cd ../../
    # download inference or pretrained model
    eval "wget -nc $model_url_value"
    if [[ $model_url_key == *inference* ]]; then
	rm -rf inference
	tar xf "${model_name}_inference.tar"
    fi

elif [ ${MODE} = "whole_train_infer" ];then
    cd dataset
    rm -rf ILSVRC2012
    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_CIFAR100.tar
    tar xf whole_chain_CIFAR100.tar
    ln -s whole_chain_CIFAR100 ILSVRC2012
    cd ILSVRC2012 
    mv train.txt train_list.txt
    mv val.txt val_list.txt
    cd ../../
fi

if [ ${MODE} = "cpp_infer" ];then
    cd deploy/cpp
    echo "################### build opencv ###################"
    rm -rf 3.4.7.tar.gz opencv-3.4.7/
    wget https://github.com/opencv/opencv/archive/3.4.7.tar.gz
    tar -xf 3.4.7.tar.gz
    install_path=$(pwd)/opencv-3.4.7/opencv3
    cd opencv-3.4.7/

    rm -rf build
    mkdir build
    cd build
    cmake .. \
	-DCMAKE_INSTALL_PREFIX=${install_path} \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_SHARED_LIBS=OFF \
	-DWITH_IPP=OFF \
	-DBUILD_IPP_IW=OFF \
	-DWITH_LAPACK=OFF \
	-DWITH_EIGEN=OFF \
	-DCMAKE_INSTALL_LIBDIR=lib64 \
	-DWITH_ZLIB=ON \
	-DBUILD_ZLIB=ON \
	-DWITH_JPEG=ON \
	-DBUILD_JPEG=ON \
	-DWITH_PNG=ON \
	-DBUILD_PNG=ON \
	-DWITH_TIFF=ON \
	-DBUILD_TIFF=ON
     make -j
     make install
     cd ../../
     echo "################### build opencv finished ###################"

     echo "################### build PaddleClas demo ####################"
     OPENCV_DIR=$(pwd)/opencv-3.4.7/opencv3/
     LIB_DIR=$(pwd)/Paddle/build/paddle_inference_install_dir/
     CUDA_LIB_DIR=$(dirname `find /usr -name libcudart.so`)
     CUDNN_LIB_DIR=$(dirname `find /usr -name libcudnn.so`)

     BUILD_DIR=build
     rm -rf ${BUILD_DIR}
     mkdir ${BUILD_DIR}
     cd ${BUILD_DIR}
     cmake .. \
        -DPADDLE_LIB=${LIB_DIR} \
        -DWITH_MKL=ON \
        -DDEMO_NAME=clas_system \
        -DWITH_GPU=OFF \
        -DWITH_STATIC_LIB=OFF \
        -DWITH_TENSORRT=OFF \
        -DTENSORRT_DIR=${TENSORRT_DIR} \
        -DOPENCV_DIR=${OPENCV_DIR} \
        -DCUDNN_LIB=${CUDNN_LIB_DIR} \
        -DCUDA_LIB=${CUDA_LIB_DIR} \

     make -j
     echo "################### build PaddleClas demo finished ###################"
fi

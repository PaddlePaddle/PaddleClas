#!/bin/bash
FILENAME=$1

# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer',
#                 'whole_infer', 'klquant_whole_infer',
#                 'cpp_infer', 'serving_infer',  'lite_infer']

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

function func_get_url_file_name(){
    strs=$1
    IFS="/"
    array=(${strs})
    tmp=${array[${#array[@]}-1]}
    echo ${tmp}
}

model_name=$(func_parser_value "${lines[1]}")

if [ ${MODE} = "cpp_infer" ];then
    if [ -d "./deploy/cpp/opencv-3.4.7/opencv3/" ] && [ $(md5sum ./deploy/cpp/opencv-3.4.7.tar.gz | awk -F ' ' '{print $1}') = "faa2b5950f8bee3f03118e600c74746a" ];then
        echo "################### build opencv skipped ###################"
    else
        echo "################### build opencv ###################"
        rm -rf ./deploy/cpp/opencv-3.4.7.tar.gz ./deploy/cpp/opencv-3.4.7/
        pushd ./deploy/cpp/
        wget -nc https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/opencv-3.4.7.tar.gz
        tar -xf opencv-3.4.7.tar.gz

        cd opencv-3.4.7/
        install_path=$(pwd)/opencv3
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
        popd
        echo "################### build opencv finished ###################"
    fi
    set_OPENCV_DIR_cmd="sed -i '1s#OPENCV_DIR=.*#OPENCV_DIR=../opencv-3.4.7/opencv3/#' './deploy/cpp/tools/build.sh'"
    eval ${set_OPENCV_DIR_cmd}
    if [ -d "./deploy/cpp/paddle_inference/" ]; then
        echo "################### build paddle inference lib skipped ###################"
    else
        pushd ./deploy/cpp/
        wget https://paddle-inference-lib.bj.bcebos.com/2.1.1-gpu-cuda10.2-cudnn8.1-mkl-gcc8.2/paddle_inference.tgz
        tar -xvf paddle_inference.tgz
        echo "################### build paddle inference lib finished ###################"
    fi
    set_LIB_DIR_cmd="sed -i '2s#LIB_DIR=.*#LIB_DIR=../paddle_inference/#' './deploy/cpp/tools/build.sh'"
    # echo ${set_LIB_DIR_cmd}
    eval ${set_LIB_DIR_cmd}
    # exit
    if [ -d "./deploy/cpp/build/" ]; then
        echo "################### build cpp inference skipped ###################"
    else
        pushd ./deploy/cpp/
        bash tools/build.sh
        popd
        echo "################### build cpp inference finished ###################"
    fi

    if [ ${model_name} == "ResNet50" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_infer.tar
        tar xf ResNet50_infer.tar
        cd ../../
    elif [ ${model_name} == "ResNet50_vd" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_vd_infer.tar
        tar xf ResNet50_vd_infer.tar
        cd ../../
    elif [ ${model_name} == "MobileNetV3_large_x1_0" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_large_x1_0_infer.tar
        tar xf MobileNetV3_large_x1_0_infer.tar
        cd ../../
    elif [ ${model_name} == "SwinTransformer_tiny_patch4_window7_224" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/SwinTransformer_tiny_patch4_window7_224_infer.tar
        tar xf SwinTransformer_tiny_patch4_window7_224_infer.tar
        cd ../../
    elif [ ${model_name} == "PPLCNet_x0_25" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x0_25_infer.tar
        tar xf PPLCNet_x0_25_infer.tar
        cd ../../
    elif [ ${model_name} == "PPLCNet_x0_35" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x0_35_infer.tar
        tar xf PPLCNet_x0_35_infer.tar
        cd ../../
    elif [ ${model_name} == "PPLCNet_x0_5" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x0_5_infer.tar
        tar xf PPLCNet_x0_5_infer.tar
        cd ../../
    elif [ ${model_name} == "PPLCNet_x0_75" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x0_75_infer.tar
        tar xf PPLCNet_x0_75_infer.tar
        cd ../../
    elif [ ${model_name} == "PPLCNet_x1_0" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x1_0_infer.tar
        tar xf PPLCNet_x1_0_infer.tar
        cd ../../
    elif [ ${model_name} == "PPLCNet_x1_5" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x1_5_infer.tar
        tar xf PPLCNet_x1_5_infer.tar
        cd ../../
    elif [ ${model_name} == "PPLCNet_x2_0" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x2_0_infer.tar
        tar xf PPLCNet_x2_0_infer.tar
        cd ../../
    elif [ ${model_name} == "PPLCNet_x2_5" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x2_5_infer.tar
        tar xf PPLCNet_x2_5_infer.tar
        cd ../../
    elif [ ${model_name} == "PP-ShiTu_general_rec" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/general_PPLCNet_x2_5_lite_v1.0_infer.tar
        tar xf general_PPLCNet_x2_5_lite_v1.0_infer.tar
        cd ../../
    elif [ ${model_name} == "PP-ShiTu_mainbody_det" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
        tar xf picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
        cd ../../
    elif [ ${model_name} == "PPLCNetV2_base" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNetV2_base_infer.tar
        tar xf PPLCNetV2_base_infer.tar
        cd ../../
    elif [ ${model_name} == "PPHGNet_tiny" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNet_tiny_infer.tar
        tar xf PPHGNet_tiny_infer.tar
        cd ../../
    elif [ ${model_name} == "PPHGNet_small" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNet_small_infer.tar
        tar xf PPHGNet_small_infer.tar
        cd ../../
    else
        echo "Not added into TIPC yet."
    fi
fi

model_name=$(func_parser_value "${lines[1]}")
model_url_value=$(func_parser_value "${lines[35]}")
model_url_key=$(func_parser_key "${lines[35]}")

if [[ $FILENAME == *GeneralRecognition* ]];then
   cd dataset
   rm -rf Aliproduct
   rm -rf train_reg_all_data.txt
   rm -rf demo_train
   wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/tipc_shitu_demo_data.tar
   tar -xf tipc_shitu_demo_data.tar
   ln -s tipc_shitu_demo_data Aliproduct
   ln -s tipc_shitu_demo_data/demo_train.txt train_reg_all_data.txt
   ln -s tipc_shitu_demo_data/demo_train demo_train
   cd tipc_shitu_demo_data
   ln -s demo_test.txt val_list.txt
   cd ../../
   eval "wget -nc $model_url_value"
   mv general_PPLCNet_x2_5_pretrained_v1.0.pdparams GeneralRecognition_PPLCNet_x2_5_pretrained.pdparams
   exit 0
fi

if [[ $FILENAME == *use_dali* ]];then
    python_name=$(func_parser_value "${lines[2]}")
    ${python_name} -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-nightly-cuda102
fi

if [ ${MODE} = "lite_train_lite_infer" ] || [ ${MODE} = "lite_train_whole_infer" ];then
    # pretrain lite train data
    cd dataset
    rm -rf ILSVRC2012
    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_little_train.tar
    tar xf whole_chain_little_train.tar
    ln -s whole_chain_little_train ILSVRC2012
    cd ILSVRC2012
    mv train.txt train_list.txt
    mv val.txt val_list.txt
    cp -r train/* val/
    cd ../../
elif [ ${MODE} = "whole_infer" ] || [ ${MODE} = "klquant_whole_infer" ];then
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
    if [[ $model_name == "SwinTransformer_large_patch4_window7_224" || $model_name == "SwinTransformer_large_patch4_window12_384" ]];then
	cmd="mv ${model_name}_22kto1k_pretrained.pdparams ${model_name}_pretrained.pdparams"
	eval $cmd
    fi

elif [ ${MODE} = "whole_train_whole_infer" ];then
    cd dataset
    rm -rf ILSVRC2012
    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_CIFAR100.tar
    tar xf whole_chain_CIFAR100.tar
    ln -s whole_chain_CIFAR100 ILSVRC2012
    cd ILSVRC2012
    mv train.txt train_list.txt
    mv test.txt val_list.txt
    cd ../../
fi

if [ ${MODE} = "serving_infer" ];then
    # prepare serving env
    python_name=$(func_parser_value "${lines[2]}")
    ${python_name} -m pip install install paddle-serving-server-gpu==0.6.1.post101
    ${python_name} -m pip install paddle_serving_client==0.6.1
    ${python_name} -m pip install paddle-serving-app==0.6.1
    unset http_proxy
    unset https_proxy
    cd ./deploy/paddleserving
    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_vd_infer.tar && tar xf ResNet50_vd_infer.tar
fi

if [ ${MODE} = "paddle2onnx_infer" ];then
    # prepare paddle2onnx env
    python_name=$(func_parser_value "${lines[2]}")
    ${python_name} -m pip install install paddle2onnx
    ${python_name} -m pip install onnxruntime
    if [ ${model_name} == "ResNet50" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_infer.tar
        tar xf ResNet50_infer.tar
        cd ../../
    elif [ ${model_name} == "ResNet50_vd" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_vd_infer.tar
        tar xf ResNet50_vd_infer.tar
        cd ../../
    elif [ ${model_name} == "MobileNetV3_large_x1_0" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_large_x1_0_infer.tar
        tar xf MobileNetV3_large_x1_0_infer.tar
        cd ../../
    elif [ ${model_name} == "SwinTransformer_tiny_patch4_window7_224" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/SwinTransformer_tiny_patch4_window7_224_infer.tar
        tar xf SwinTransformer_tiny_patch4_window7_224_infer.tar
        cd ../../
    elif [ ${model_name} == "PPLCNet_x0_25" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x0_25_infer.tar
        tar xf PPLCNet_x0_25_infer.tar
        cd ../../
    elif [ ${model_name} == "PPLCNet_x0_35" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x0_35_infer.tar
        tar xf PPLCNet_x0_35_infer.tar
        cd ../../
    elif [ ${model_name} == "PPLCNet_x0_5" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x0_5_infer.tar
        tar xf PPLCNet_x0_5_infer.tar
        cd ../../
    elif [ ${model_name} == "PPLCNet_x0_75" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x0_75_infer.tar
        tar xf PPLCNet_x0_75_infer.tar
        cd ../../
    elif [ ${model_name} == "PPLCNet_x1_0" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x1_0_infer.tar
        tar xf PPLCNet_x1_0_infer.tar
        cd ../../
    elif [ ${model_name} == "PPLCNet_x1_5" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x1_5_infer.tar
        tar xf PPLCNet_x1_5_infer.tar
        cd ../../
    elif [ ${model_name} == "PPLCNet_x2_0" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x2_0_infer.tar
        tar xf PPLCNet_x2_0_infer.tar
        cd ../../
    elif [ ${model_name} == "PPLCNet_x2_5" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x2_5_infer.tar
        tar xf PPLCNet_x2_5_infer.tar
        cd ../../
    elif [ ${model_name} == "PP-ShiTu_general_rec" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/general_PPLCNet_x2_5_lite_v1.0_infer.tar
        tar xf general_PPLCNet_x2_5_lite_v1.0_infer.tar
        cd ../../
    elif [ ${model_name} == "PP-ShiTu_mainbody_det" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
        tar xf picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
        cd ../../
    elif [ ${model_name} == "PPLCNetV2_base" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNetV2_base_infer.tar
        tar xf PPLCNetV2_base_infer.tar
        cd ../../
    elif [ ${model_name} == "PPHGNet_tiny" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNet_tiny_infer.tar
        tar xf PPHGNet_tiny_infer.tar
        cd ../../
    elif [ ${model_name} == "PPHGNet_small" ]; then
        # wget model
        cd deploy
        mkdir models
        cd models
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNet_small_infer.tar
        tar xf PPHGNet_small_infer.tar
        cd ../../
    else
        echo "Not added into TIPC yet."
    fi
fi

if [ ${MODE} = "benchmark_train" ];then
    pip install -r requirements.txt
    cd dataset
    rm -rf ILSVRC2012
    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/ImageNet1k/ILSVRC2012_val.tar
    tar xf ILSVRC2012_val.tar
    ln -s ILSVRC2012_val ILSVRC2012
    cd ILSVRC2012
    ln -s val_list.txt  train_list.txt
    cd ../../
fi

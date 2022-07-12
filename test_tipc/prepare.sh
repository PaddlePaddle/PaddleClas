#!/bin/bash
FILENAME=$1

# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer',
#                 'whole_infer', 'cpp_infer', 'serving_infer',  'lite_infer']

MODE=$2

dataline=$(cat ${FILENAME})
# parser params
IFS=$'\n'
lines=(${dataline})

function func_parser_key() {
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[0]}
    echo ${tmp}
}

function func_parser_value() {
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

function func_get_url_file_name() {
    strs=$1
    IFS="/"
    array=(${strs})
    tmp=${array[${#array[@]} - 1]}
    echo ${tmp}
}

model_name=$(func_parser_value "${lines[1]}")

if [[ ${MODE} = "cpp_infer" ]]; then
    if [ -d "./deploy/cpp/opencv-3.4.7/opencv3/" ] && [ $(md5sum ./deploy/cpp/opencv-3.4.7.tar.gz | awk -F ' ' '{print $1}') = "faa2b5950f8bee3f03118e600c74746a" ]; then
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
    if [[ ! -d "./deploy/cpp/paddle_inference/" ]]; then
        pushd ./deploy/cpp/
        PADDLEInfer=$3
        if [ "" = "$PADDLEInfer" ];then
            wget -nc https://paddle-inference-lib.bj.bcebos.com/2.2.2/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddle_inference.tgz --no-check-certificate
        else
            wget -nc ${PADDLEInfer} --no-check-certificate
        fi
        tar xf paddle_inference.tgz
        popd
    fi
    if [[ $FILENAME == *infer_cpp_linux_gpu_cpu.txt ]]; then
        cpp_type=$(func_parser_value "${lines[2]}")
        cls_inference_model_dir=$(func_parser_value "${lines[3]}")
        det_inference_model_dir=$(func_parser_value "${lines[4]}")
        cls_inference_url=$(func_parser_value "${lines[5]}")
        det_inference_url=$(func_parser_value "${lines[6]}")

        if [[ $cpp_type == "cls" ]]; then
            eval "wget -nc $cls_inference_url"
            tar_name=$(func_get_url_file_name "$cls_inference_url")
            model_dir=${tar_name%.*}
            eval "tar xf ${tar_name}"

            # move '_int8' suffix in pact models
            if [[ ${tar_name} =~ "pact_infer" ]]; then
                cd ${cls_inference_model_dir}
                mv inference_int8.pdiparams inference.pdiparams
                mv inference_int8.pdmodel inference.pdmodel
                cd ..
            fi

            cd dataset
            rm -rf ILSVRC2012
            wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_infer.tar
            tar xf whole_chain_infer.tar
            ln -s whole_chain_infer ILSVRC2012
            cd ..
        elif [[ $cpp_type == "shitu" ]]; then
            eval "wget -nc $cls_inference_url"
            tar_name=$(func_get_url_file_name "$cls_inference_url")
            model_dir=${tar_name%.*}
            eval "tar xf ${tar_name}"

            eval "wget -nc $det_inference_url"
            tar_name=$(func_get_url_file_name "$det_inference_url")
            model_dir=${tar_name%.*}
            eval "tar xf ${tar_name}"

            cd dataset
            wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/drink_dataset_v1.0.tar
            tar -xf drink_dataset_v1.0.tar
        else
            echo "Wrong cpp type in config file in line 3. only support cls, shitu"
        fi
        exit 0
    else
        echo "use wrong config file"
        exit 1
    fi
fi

model_name=$(func_parser_value "${lines[1]}")
model_url_value=$(func_parser_value "${lines[35]}")
model_url_key=$(func_parser_key "${lines[35]}")

if [[ $model_name == *ShiTu* ]]; then
    cd dataset
    rm -rf Aliproduct
    rm -rf train_reg_all_data.txt
    rm -rf demo_train
    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/tipc_shitu_demo_data.tar --no-check-certificate
    tar -xf tipc_shitu_demo_data.tar
    ln -s tipc_shitu_demo_data Aliproduct
    ln -s tipc_shitu_demo_data/demo_train.txt train_reg_all_data.txt
    ln -s tipc_shitu_demo_data/demo_train demo_train
    cd tipc_shitu_demo_data
    ln -s demo_test.txt val_list.txt
    cd ../../
    eval "wget -nc $model_url_value --no-check-certificate"
    mv general_PPLCNet_x2_5_pretrained_v1.0.pdparams GeneralRecognition_PPLCNet_x2_5_pretrained.pdparams
    exit 0
fi

if [[ $FILENAME == *use_dali* ]]; then
    python_name=$(func_parser_value "${lines[2]}")
    ${python_name} -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-nightly-cuda102
fi

if [[ ${MODE} = "lite_train_lite_infer" ]] || [[ ${MODE} = "lite_train_whole_infer" ]]; then
    # pretrain lite train data
    cd dataset
    rm -rf ILSVRC2012
    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_little_train.tar --no-check-certificate
    tar xf whole_chain_little_train.tar
    ln -s whole_chain_little_train ILSVRC2012
    cd ILSVRC2012
    mv train.txt train_list.txt
    mv val.txt val_list.txt
    cp -r train/* val/
    cd ../../
    if [[ ${FILENAME} =~ "pact_infer" ]]; then
        # download pretrained model for PACT training
        pretrpretrained_model_url=$(func_parser_value "${lines[35]}")
        mkdir pretrained_model
        cd pretrained_model
        wget -nc ${pretrpretrained_model_url} --no-check-certificate
        cd ..
    fi
elif [[ ${MODE} = "whole_infer" ]]; then
    # download data
    if [[ ${model_name} =~ "GeneralRecognition" ]]; then
        cd dataset
        rm -rf Aliproduct
        rm -rf train_reg_all_data.txt
        rm -rf demo_train
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/tipc_shitu_demo_data.tar --no-check-certificate
        tar -xf tipc_shitu_demo_data.tar
        ln -s tipc_shitu_demo_data Aliproduct
        ln -s tipc_shitu_demo_data/demo_train.txt train_reg_all_data.txt
        ln -s tipc_shitu_demo_data/demo_train demo_train
        cd tipc_shitu_demo_data
        ln -s demo_test.txt val_list.txt
        cd ../../
    else
        cd dataset
        rm -rf ILSVRC2012
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_infer.tar
        tar xf whole_chain_infer.tar
        ln -s whole_chain_infer ILSVRC2012
        cd ILSVRC2012
        mv val.txt val_list.txt
        ln -s val_list.txt train_list.txt
        cd ../../
    fi
    # download inference or pretrained model
    eval "wget -nc $model_url_value"
    if [[ ${model_url_value} =~ ".tar" ]]; then
        tar_name=$(func_get_url_file_name "${model_url_value}")
        echo $tar_name
        rm -rf {tar_name}
        tar xf ${tar_name}
    fi
    if [[ $model_name == "SwinTransformer_large_patch4_window7_224" || $model_name == "SwinTransformer_large_patch4_window12_384" ]]; then
        cmd="mv ${model_name}_22kto1k_pretrained.pdparams ${model_name}_pretrained.pdparams"
        eval $cmd
    fi

elif [[ ${MODE} = "whole_train_whole_infer" ]]; then
    cd dataset
    rm -rf ILSVRC2012
    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_CIFAR100.tar
    tar xf whole_chain_CIFAR100.tar
    ln -s whole_chain_CIFAR100 ILSVRC2012
    cd ILSVRC2012
    mv train.txt train_list.txt
    mv test.txt val_list.txt
    cd ../../
    if [[ ${FILENAME} =~ "pact_infer" ]]; then
        # download pretrained model for PACT training
        pretrpretrained_model_url=$(func_parser_value "${lines[35]}")
        mkdir pretrained_model
        cd pretrained_model
        wget -nc ${pretrpretrained_model_url} --no-check-certificate
        cd ..
    fi
fi

if [[ ${MODE} = "serving_infer" ]]; then
    # prepare serving env
    python_name=$(func_parser_value "${lines[2]}")
    if [[ ${model_name} = "PPShiTu" ]]; then
        cls_inference_model_url=$(func_parser_value "${lines[3]}")
        cls_tar_name=$(func_get_url_file_name "${cls_inference_model_url}")
        det_inference_model_url=$(func_parser_value "${lines[4]}")
        det_tar_name=$(func_get_url_file_name "${det_inference_model_url}")
        cd ./deploy
        wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/drink_dataset_v1.0.tar --no-check-certificate
        tar -xf drink_dataset_v1.0.tar
        mkdir models
        cd models
        wget -nc ${cls_inference_model_url} && tar xf ${cls_tar_name}
        wget -nc ${det_inference_model_url} && tar xf ${det_tar_name}
        cd ..
    else
        cls_inference_model_url=$(func_parser_value "${lines[3]}")
        cls_tar_name=$(func_get_url_file_name "${cls_inference_model_url}")
        cd ./deploy/paddleserving
        wget -nc ${cls_inference_model_url}
        tar xf ${cls_tar_name}

        # move '_int8' suffix in pact models
        if [[ ${cls_tar_name} =~ "pact_infer" ]]; then
            cls_inference_model_dir=${cls_tar_name%%.tar}
            cd ${cls_inference_model_dir}
            mv inference_int8.pdiparams inference.pdiparams
            mv inference_int8.pdmodel inference.pdmodel
            cd ..
        fi

        cd ../../
    fi
    unset http_proxy
    unset https_proxy
fi

if [[ ${MODE} = "paddle2onnx_infer" ]]; then
    # prepare paddle2onnx env
    python_name=$(func_parser_value "${lines[2]}")
    inference_model_url=$(func_parser_value "${lines[10]}")
    tar_name=${inference_model_url##*/}

    ${python_name} -m pip install paddle2onnx
    ${python_name} -m pip install onnxruntime
    cd deploy
    mkdir models
    cd models
    wget -nc ${inference_model_url}
    tar xf ${tar_name}
    cd ../../
fi

if [[ ${MODE} = "benchmark_train" ]]; then
    pip install -r requirements.txt
    cd dataset
    rm -rf ILSVRC2012
    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/ImageNet1k/ILSVRC2012_val.tar
    tar xf ILSVRC2012_val.tar
    ln -s ILSVRC2012_val ILSVRC2012
    cd ILSVRC2012
    ln -s val_list.txt train_list.txt
    cd ../../
fi

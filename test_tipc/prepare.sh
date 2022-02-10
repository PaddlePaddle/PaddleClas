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
   if [[ $FILENAME == *infer_cpp_linux_gpu_cpu.txt ]];then
	cpp_type=$(func_parser_value "${lines[2]}")
	cls_inference_model_dir=$(func_parser_value "${lines[3]}")
	det_inference_model_dir=$(func_parser_value "${lines[4]}")
	cls_inference_url=$(func_parser_value "${lines[5]}")
	det_inference_url=$(func_parser_value "${lines[6]}")

	if [[ $cpp_type == "cls" ]];then
	    eval "wget -nc $cls_inference_url"
	    tar xf "${model_name}_inference.tar"
	    eval "mv inference $cls_inference_model_dir"
	    cd dataset
    	    rm -rf ILSVRC2012
    	    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_infer.tar
    	    tar xf whole_chain_infer.tar
    	    ln -s whole_chain_infer ILSVRC2012
	    cd ..
	elif [[ $cpp_type == "shitu" ]];then
	    eval "wget -nc $cls_inference_url"
	    tar_name=$(func_get_url_file_name "$cls_inference_url")
	    model_dir=${tar_name%.*}
	    eval "tar xf ${tar_name}"
	    eval "mv ${model_dir} ${cls_inference_model_dir}"
	    
	    eval "wget -nc $det_inference_url"
	    tar_name=$(func_get_url_file_name "$det_inference_url") 
	    model_dir=${tar_name%.*}
	    eval "tar xf ${tar_name}"
	    eval "mv ${model_dir} ${det_inference_model_dir}"
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

    # wget model
    cd deploy && mkdir models && cd models
    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_vd_infer.tar  && tar xf ResNet50_vd_infer.tar
    cd ../../
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

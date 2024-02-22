#!/bin/bash

# 本脚本用于测试PaddleClas系列模型的自动压缩功能
## 运行脚本前，请确保处于以下环境：
## CUDA11.2+TensorRT8.0.3.4+Paddle2.5.2

## MobileNetV3_small_x1_0
## 启动自动化压缩训练
CUDA_VISIBLE_DEVICES=0 python run.py --save_dir ./models/MobileNetV3_small_x1_0_qat/ --compression_config_path ./configs/MobileNetV3_small_x1_0/qat_dis.yaml --reader_config_path ./configs/MobileNetV3_small_x1_0/data_reader.yaml
## GPU指标测试
### 量化前，预期指标：Top-1 Acc:68.01%;time:2.4ms
python test_ppclas.py --model_path ./models/MobileNetV3_small_x1_0_infer/ --use_gpu=True --use_trt=True
### 量化后，预期指标：Top-1 Acc:66.94%;time:1.6ms
python test_ppclas.py --model_path ./models/MobileNetV3_small_x1_0_qat/ --use_gpu=True --use_trt=True --use_int8=True
## CPU指标测试
### 量化前，预期指标：Top-1 Acc:68.01%;time:27.2ms
python test_ppclas.py --model_path ./models/MobileNetV3_small_x1_0_infer/ --cpu_num_threads=10 --use_mkldnn=True
### 量化后，预期指标：Top-1 Acc:66.93%;time:27.5ms
python test_ppclas.py --model_path ./models/MobileNetV3_small_x1_0_qat/ --cpu_num_threads=10 --use_mkldnn=True --use_int8=True


## ResNet50_vd
## 启动自动化压缩训练
CUDA_VISIBLE_DEVICES=0 python run.py --save_dir ./models/ResNet50_vd_qat/ --compression_config_path ./configs/ResNet50/qat_dis.yaml --reader_config_path ./configs/ResNet50/data_reader.yaml
## GPU指标测试
### 量化前，预期指标：Top-1 Acc:79.05%;time:5.9ms
python test_ppclas.py --model_path ./models/ResNet50_vd_infer/ --use_gpu=True --use_trt=True
### 量化后，预期指标：Top-1 Acc:78.62%;time:2.8ms
python test_ppclas.py --model_path ./models/ResNet50_vd_qat/ --use_gpu=True --use_trt=True --use_int8=True
## CPU指标测试
### 量化前，预期指标：Top-1 Acc:79.05%;time:100.4ms
python test_ppclas.py --model_path ./models/ResNet50_vd_infer/ --cpu_num_threads=10 --use_mkldnn=True
### 量化后，预期指标：Top-1 Acc:78.64%;time:101.6ms
python test_ppclas.py --model_path ./models/ResNet50_vd_qat/ --cpu_num_threads=10 --use_mkldnn=True --use_int8=True


## PPHGNet_small
## 启动自动化压缩训练
CUDA_VISIBLE_DEVICES=0 python run.py --save_dir ./models/PPHGNet_small_qat/ --compression_config_path ./configs/PPHGNet_small/qat_dis.yaml --reader_config_path ./configs/PPHGNet_small/data_reader.yaml
## GPU指标测试
### 量化前，预期指标：Top-1 Acc:81.33%;time:6.5ms
python test_ppclas.py --model_path ./models/PPHGNet_small_infer/ --use_gpu=True --use_trt=True
### 量化后，预期指标：Top-1 Acc:81.25%;time:4.0ms
python test_ppclas.py --model_path ./models/PPHGNet_small_qat/ --use_gpu=True --use_trt=True --use_int8=True
## CPU指标测试
### 量化前，预期指标：Top-1 Acc:81.33%;time:151.9ms
python test_ppclas.py --model_path ./models/PPHGNet_small_infer/ --cpu_num_threads=10 --use_mkldnn=True
### 量化后，预期指标：Top-1 Acc:81.26%;time:154.9ms
python test_ppclas.py --model_path ./models/PPHGNet_small_qat/ --cpu_num_threads=10 --use_mkldnn=True --use_int8=True


## SwinTransformer_base_patch4_window7_224
## 启动自动化压缩训练
CUDA_VISIBLE_DEVICES=0 python run.py --save_dir ./models/SwinTransformer_base_patch4_window7_224_qat/ --compression_config_path ./configs/SwinTransformer_base/qat_dis.yaml --reader_config_path ./configs/SwinTransformer_base/data_reader.yaml
## GPU指标测试
### 量化前，预期指标：Top-1 Acc:83.26%;time:18.5ms
python test_ppclas.py --model_path ./models/SwinTransformer_base_patch4_window7_224_infer/ --use_gpu=True --use_trt=True
### 量化后，预期指标：Top-1 Acc:83.26%;time:9.2ms
python test_ppclas.py --model_path ./models/SwinTransformer_base_patch4_window7_224_qat/ --use_gpu=True --use_trt=True --use_int8=True
## CPU指标测试
### 量化前，预期指标：Top-1 Acc:83.26%;time:4792.4ms
python test_ppclas.py --model_path ./models/SwinTransformer_base_patch4_window7_224_infer/ --cpu_num_threads=10 --use_mkldnn=True
### 量化后，预期指标：Top-1 Acc:82.12%;time:4727.5ms
python test_ppclas.py --model_path ./models/SwinTransformer_base_patch4_window7_224_qat/ --cpu_num_threads=10 --use_mkldnn=True --use_int8=True


## PPLCNet_x1_0
## 启动自动化压缩训练
CUDA_VISIBLE_DEVICES=0 python run.py --save_dir ./models/PPLCNet_x1_0_qat/ --compression_config_path ./configs/PPLCNet_x1_0/qat_dis.yaml --reader_config_path ./configs/PPLCNet_x1_0/data_reader.yaml
## GPU指标测试
### 量化前，预期指标：Top-1 Acc:71.05%;time:2.3ms
python test_ppclas.py --model_path ./models/PPLCNet_x1_0_infer/ --use_gpu=True --use_trt=True
### 量化后，预期指标：Top-1 Acc:70.70%;time:1.6ms
python test_ppclas.py --model_path ./models/PPLCNet_x1_0_qat/ --use_gpu=True --use_trt=True --use_int8=True
## CPU指标测试
### 量化前，预期指标：Top-1 Acc:71.05%;time:45.7ms
python test_ppclas.py --model_path ./models/PPLCNet_x1_0_infer/ --cpu_num_threads=10 --use_mkldnn=True
### 量化后，预期指标：Top-1 Acc:70.70%;time:43.2ms
python test_ppclas.py --model_path ./models/PPLCNet_x1_0_qat/ --cpu_num_threads=10 --use_mkldnn=True --use_int8=True


## CLIP_vit_base_patch16_224
## 启动自动化压缩训练
CUDA_VISIBLE_DEVICES=0 python run.py --save_dir ./models/CLIP_vit_base_patch16_224_qat/ --compression_config_path ./configs/CLIP_vit_base_patch16_224/qat_dis.yaml --reader_config_path ./configs/CLIP_vit_base_patch16_224/data_reader.yaml
## GPU指标测试
### 量化前，预期指标：Top-1 Acc:85.36%;time:13.9ms
python test_ppclas.py --model_path ./models/CLIP_vit_base_patch16_224_infer/ --use_gpu=True --use_trt=True --min_subgraph_size=5
### 量化后，预期指标：Top-1 Acc:85.36%;time:8.3ms
python test_ppclas.py --model_path ./models/CLIP_vit_base_patch16_224_qat/ --use_gpu=True --use_trt=True --use_int8=True --min_subgraph_size=5
## CPU指标测试
### 量化前，预期指标：Top-1 Acc:85.36%;time:437.6ms
python test_ppclas.py --model_path ./models/CLIP_vit_base_patch16_224_infer/ --cpu_num_threads=10 --use_mkldnn=True
### 量化后，预期指标：Top-1 Acc:85.33%;time:405.7ms
python test_ppclas.py --model_path ./models/CLIP_vit_base_patch16_224_qat/ --cpu_num_threads=10 --use_mkldnn=True --use_int8=True

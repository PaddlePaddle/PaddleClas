#!/usr/bin/env python
# coding: utf-8

#                                      基于移动轻量级网络的杂草的分类
#    本文以ResNet50和MobileNetV3模型为基础，采用数据增强、迁移学习来进行快速训练，在训练中通过设置加权Softmax损失函数的权重，最后再利用精度高的服务器端模型指导和优化移动端模型性能，从而得到一个轻量模型。
# 
# 
# 

# 一 项目背景
#    针对目前杂草识别分类存在的主要问题：一是杂草识别模型的研究大多集中在服务器端模型，而该类模型存在规模较大，占用较多计算资源，计算速度较慢等问题；二是杂草识别准确率仍有待进一步提升。
#     ![](https://ai-studio-static-online.cdn.bcebos.com/06f74c8e5a99412797b6cc64f197239f428057ff82c546e7871825d3207b84af)
# 
#     
#     
#     

# In[2]:


#导入一些图像处理的包
get_ipython().run_line_magic('cd', '/home/aistudio')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, shutil, cv2, random
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# 先把paddleclas安装上再说
# 安装paddleclas以及相关三方包(好像studio自带的已经够用了，无需安装了)
get_ipython().system('git clone https://gitee.com/paddlepaddle/PaddleClas.git -b release/2.2')
# 我这里安装相关包时，花了30几分钟还有错误提示，不管他即可


# 二 数据集
# # # 训练数据集
# # train_dataset_path = ./PaddleClas/dataset/dataset/trainImageSet
# 
# # # 评估数据集
# # eval_dataset_path = ./PaddleClas/dataset/dataset/evalImageSet
# 
# 本文采用DeepWeeds数据集。该数据集是Alex Olsen等人于2018年12月提供的一个标准杂草数据集。该数据集DeepWeeds包含17509张8类杂草物种和负类目标。该数据集图像分辨率是256×256，所有图片格式为JPG，其中Chinee apple 1125张、Lantana 1064张、Parkinsonia 1031张、Parthenium 1031张、Prickly acacia 1062张、Rubber vine 1009张、Siambic weed 1074张、Snake weed 1074张 、不包含目标杂草的负类照片 9106张。本文对DeepWeeds数据集按照4：1的比例划分为训练集和验证集。

# In[ ]:


get_ipython().run_line_magic('cd', 'PaddleClas/')


# 下载预训练参数

# In[ ]:


get_ipython().run_line_magic('cd', 'pretrained')
#!wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_pretrained.pdparams
# !wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_pretrained.pdparams
##!wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_ssld_pretrained.pdparams
# !wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x1_0_pretrained.pdparams
# !wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x1_0_pretrained.pdparams
get_ipython().system('wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x1_0_ssld_pretrained.pdparams')


# In[ ]:


get_ipython().run_line_magic('cd', 'PaddleClas')



# 三  开始模型训练

# In[ ]:


get_ipython().system('python tools/train.py -c ./Ai-quick/quick_start/MobileNetV3_large_x1_0_finetune.yaml   ')


# #MobileNetV3_large_x1_0_ssld

# In[ ]:


get_ipython().system('python tools/train.py -c ./Ai-quick/quick_start/MobileNetV3_large_x1_0_ssld_finetune.yaml    ')


# #MobileNetV3_large_x1_0_ssld 加权

# In[ ]:


get_ipython().system('python tools/train.py -c ./Ai-quick/quick_start/MobileNetV3_large_x1_0_ssld_finetune.yaml')


# In[ ]:


get_ipython().run_line_magic('cd', 'PaddleClas/')


# 开始调用resnet50模型
# 

# In[ ]:


get_ipython().system('python tools/train.py -c ./Ai-quick/quick_start/ResNet50.yaml')


# In[ ]:


# 开始训练
get_ipython().system('python tools/train.py -c ./configs/ResNet/ResNet50.yaml')


# In[ ]:


get_ipython().system('python tools/train.py -c ./Ai-quick/quick_start/ResNet50_vd_finetune.yaml #    --vdl_dir ./output  ')


# 更改知识蒸馏学习率  

# In[ ]:


get_ipython().system('python tools/train.py -c ./Ai-quick/quick_start/ResNet50_vd_ssld_finetune_1.yaml')


#  ppcls/modeling/loss.py中修改  softmax权重

# # 开始调用resnet50_vd_ssld   （带预训练模型）
# 

# In[ ]:


get_ipython().system('python tools/train.py -c ./Ai-quick/quick_start/ResNet50_vd_ssld_finetune.yaml')


# In[ ]:


##微调学习率  0.002


# In[ ]:


get_ipython().system('python tools/train.py -c ./Ai-quick/quick_start/ResNet50_vd_ssld_0.002.yaml')


# In[ ]:


get_ipython().system('python tools/train.py -c ./Ai-quick/quick_start/ResNet50_vd_ssld_0.000001.yaml')


# **开始调用resnet50_vd_ssld  加权·**    获取训练参数

# In[ ]:


get_ipython().system('python tools/train.py -c ./Ai-quick/quick_start/ResNet50_vd_ssld_finetune.yaml')


# 数据增广的尝试-RandomErasing

# In[ ]:


get_ipython().system('python tools/train.py -c ./Ai-quick/quick_start/ResNet50_vd_ssld_random_erasing_finetune.yaml')


# 调用resnet50_mobilenet知识蒸馏

# In[ ]:


get_ipython().system('python -m paddle.distributed.launch     --selected_gpus="0"     tools/train.py         -c ./Ai-quick/quick_start/R50_vd_distill_MV3_large_x1_0.yaml')


# **调用resnet50_mobilenet_ssld知识蒸馏**
# 

# In[ ]:


调用resnet50_mobilenet_ssld知识蒸馏  加权处理 


# In[1]:


cd PaddleClas/


# In[ ]:


get_ipython().system('python tools/train.py -c ./Ai-quick/quick_start/R50_vd_distill_MV3_large_x1_0_ssld_100.yaml')


# 四 模型评估

# In[ ]:





# In[ ]:


get_ipython().system('python tools/eval.py     -c ./Ai-quick/quick_start/ResNet50_vd_ssld_finetune.yaml     -o pretrained_model="./output/ResNet50_vd/best_model/ppcls"    -o load_static_weights=False')


# #  resnet_vd模型评估

# In[ ]:


#%cd PaddleClas/
get_ipython().system('python tools/eval.py     -c ./Ai-quick/quick_start/ResNet50_vd_ssld_finetune.yaml     -o pretrained_model="../work/output/ResNet50_0.0125/ResNet50/best_model/ppcls"     -o load_static_weights=False')


# mobilenet_v3-large + 加权

# In[ ]:


get_ipython().run_line_magic('pwd', '')


# In[ ]:


get_ipython().system('python tools/eval.py     -c ./Ai-quick/quick_start/MobileNetV3_large_x1_0_finetune.yaml     -o pretrained_model="../work/output/MobileNetV3_large_x1_0/best_model/ppcls"     -o load_static_weights=False')


# 蒸馏 resnet_vd   mobilenet_v3-large + 加权  评估

# In[ ]:


get_ipython().system('python tools/eval.py     # -c ./Ai-quick/eval.yaml \\')
    -c ./Ai-quick/quick_start/R50_vd_distill_MV3_large_x1_0.yaml     -o pretrained_model="../work/output/ResNet50_vd_distill_MobileNetV3_large_x1_0_ssld_add/ResNet50_vd_distill_MobileNetV3_large_x1_0/best_model/ppcls"     -o load_static_weights=False


# In[ ]:


get_ipython().run_line_magic('cd', 'PaddleClas/')
get_ipython().system('python tools/eval.py     -c ./Ai-quick/quick_start/MobileNetV3_large_x1_0_finetune.yaml     -o pretrained_model="../work/output/ResNet50_vd_distill_MobileNetV3_large_x1_0/best_model/ppcls_student"     -o load_static_weights=False')


# In[ ]:


get_ipython().system('cp ../random_erasing.py ppcls/data/imaug/random_erasing.py')


# In[ ]:


get_ipython().system('python tools/train.py -c ./ResNet50.yaml')


# ### 五、模型推理
# 首先，对训练好的模型进行转换：
# ```
# python tools/export_model.py \
#     --model=模型名字 \
#     --pretrained_model=预训练模型路径 \
#     --output_path=预测模型保存路径
# ```
# 之后，通过推理引擎进行推理：
# ```
# python tools/infer/predict.py \
#     -m model文件路径 \
#     -p params文件路径 \
#     -i 图片路径 \
#     --use_gpu=1 \
#     --use_tensorrt=True
# ```
# 
# 更多的参数说明可以参考[https://github.com/PaddlePaddle/PaddleClas/blob/master/tools/infer/predict.py](https://github.com/PaddlePaddle/PaddleClas/blob/master/tools/infer/predict.py)中的`parse_args`函数。
# 
# 更多关于服务器端与端侧的预测部署方案请参考：[https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/index_cn.html](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/index_cn.html)

# mobilenet 模型转化和评估

# In[ ]:


get_ipython().system('python tools/export_model.py     --model=MobileNetV3_large_x1_0     --pretrained_model=../work/output/MobileNetV3_large_x1_0/best_model/ppcls     --output_path=inference/MobileNetV3_large_x1_0     --class_dim 9')


# In[ ]:


# 可以预测整个目录
get_ipython().system('python tools/infer/predict.py     --model_file ./inference/MobileNetV3_large_x1_0/inference.pdmodel     --params_file ./inference/MobileNetV3_large_x1_0/inference.pdiparams     --image_file ./dataset/dataset/evalImageSet     # --args.model ./inference/ResNet50_vd_distill_MobileNetV3_large_x1_0/ \\')
    # --args.batch_size=5 \
    --use_gpu=True


# 对resnet50进行推理模型转换

# In[ ]:


get_ipython().system('python tools/export_model.py     --model=ResNet50     --pretrained_model=../work/output/ResNet50/best_model/ppcls     --output_path=inference/ResNet50     --class_dim 9')


# In[ ]:


get_ipython().run_line_magic('cd', 'PaddleClas/')
get_ipython().run_line_magic('pwd', '')


# resnet50_vd_ssld+w模型转换

# In[ ]:


get_ipython().system('python tools/export_model.py     --model=ResNet50_vd     --pretrained_model=../work/output/ResNet50_vd_ssld_add/best_model/ppcls    --output_path=inference/ResNet50_vd_ssld_add    --class_dim 9')


# 对MobileNetv3_large_x1进行推理模型转换

# In[ ]:


# 注意要写入类别数
get_ipython().system('python tools/export_model.py     --model=MobileNetV3_large_x1_0     --pretrained_model=../work/output/MobileNetV3_large_x1_0/best_model/ppcls     --output_path=inference/MobileNetv3_large_x1     --class_dim 9')


# **对Resnet-Mobilenet-SSLD进行推理模型转换**

# In[ ]:


get_ipython().system('python tools/export_model.py     --model= ResNet50_vd_distill_MobileNetV3_large_x1_0_ssld_add     --pretrained_model= ../work/output/ResNet50_vd_distill_MobileNetV3_large_x1_0_ssld_add/ResNet50_vd_distill_MobileNetV3_large_x1_0/best_model/ppcls_student     --output_path= inference/ResNet50_vd_distill_MobileNetV3_large_x1_0_ssld_add     --class_dim 9')


# In[ ]:


# 可以预测整个目录
get_ipython().system('python tools/infer/predict.py     --model_file ./inference/ResNet50_vd_distill_MobileNetV3_large_x1_0/inference.pdmodel     --params_file ./inference/ResNet50_vd_distill_MobileNetV3_large_x1_0/inference.pdiparams     --image_file ./dataset/dataset/evalImageSet     # --args.model ./inference/ResNet50_vd_distill_MobileNetV3_large_x1_0/ \\')
    # --args.batch_size=5 \
    --use_gpu=True


# ResNet50_vd_distill_MobileNetV3_large_x1_0模型推理转换

# In[ ]:


# 注意要写入类别数
get_ipython().system('python tools/export_model.py     --model= MobileNetV3_large_x1_0     --pretrained_model=../work/output/ResNet50_vd_distill_MobileNetV3_large_x1_0/best_model/ppcls_student     --output_path=inference/ResNet50_vd_distill_MobileNetV3_large_x1_0     --class_dim 9')


# ResNet50_vd_distill_MobileNetV3_large_x1_0_ssld模型推理转换   转化为静态模型

# In[ ]:


work/output/ResNet50_vd_distill_MobileNetV3_large_x1_0/best_model


# In[ ]:


get_ipython().run_line_magic('cd', 'PaddleClas')


# In[ ]:


# 注意要写入类别数
get_ipython().system('python tools/export_model.py     --model=MobileNetV3_large_x1_0     --pretrained_model=../work/output/ResNet50_vd_distill_MobileNetV3_large_x1_0_ssld_add/ResNet50_vd_distill_MobileNetV3_large_x1_0/best_model/ppcls_student     --output_path=inference/ResNet50_vd_distill_MobileNetV3_large_x1_0_ssld_1     --class_dim 9')


# In[ ]:


# 可以预测整个目录
get_ipython().system('python tools/infer/predict.py     --model_file ./inference/ResNet50_vd_distill_MobileNetV3_large_x1_0_ssld_1/inference.pdmodel     --params_file ./inference/ResNet50_vd_distill_MobileNetV3_large_x1_0_ssld_1/inference.pdiparams     --image_file dataset/dataset/evalImageSet     --use_gpu=True')
    # --use_tensorrt=True


# In[ ]:


# 可以预测整个目录
get_ipython().system('python tools/infer/predict.py     --model_file inference/ResNet50/inference.pdmodel     --params_file inference/ResNet50/inference.pdiparams     --image_file dataset/dataset/evalImageSet     --use_gpu=True')
    # --use_tensorrt=True


# In[ ]:


# 可以预测整个目录
get_ipython().system('python tools/infer/predict.py     --model_file inference/ResNet50_vd_ssld_add/inference.pdmodel     --params_file inference/ResNet50_vd_ssld_add/inference.pdiparams     --image_file dataset/dataset/evalImageSet     --use_gpu=True')
    # --use_tensorrt=True


# In[ ]:


get_ipython().run_line_magic('cd', 'inference/')


# In[ ]:


get_ipython().system('unzip -oq /home/aistudio/PaddleClas/inference/MobileNetv3_large_x1.zip')


# In[ ]:


# 可以预测整个目录
get_ipython().system('python tools/infer/predict.py     --model_file inference/MobileNetv3_large_x1/inference.pdmodel     --params_file inference/MobileNetv3_large_x1/inference.pdiparams     --image_file dataset/dataset/evalImageSet     --use_gpu=True')
    # --use_tensorrt=True


# In[ ]:


# 可以预测整个目录
get_ipython().system('python tools/infer/predict.py     --model_file inference/ResNet50_vd_distill_MobileNetV3_large_x1_0/inference.pdmodel     --params_file inference/ResNet50_vd_distill_MobileNetV3_large_x1_0/inference.pdiparams     --image_file dataset/dataset/evalImageSet     --use_gpu=True')
    # --use_tensorrt=True


# In[ ]:


# 可以预测整个目录
get_ipython().system('python tools/infer/predict.py     --model_file inference/ResNet50_vd_distill_MobileNetV3_large_x1_0/inference.pdmodel     --params_file inference/ResNet50_vd_distill_MobileNetV3_large_x1_0/inference.pdiparams     --image_file dataset/dataset/evalImageSet     --use_gpu=True')
    # --use_tensorrt=True


# ### 六、输出预测结果
# 做一些小幅改造，让预测结果以`sample_submit.csv`的格式保存，便于提交。

# In[ ]:


get_ipython().system('cp ../predict.py tools/infer/submit.py')


# In[ ]:


# 可以预测整个目录
get_ipython().system('python tools/infer/submit.py     --model_file inference/inference.pdmodel     --params_file inference/inference.pdiparams     --image_file ./dataset/dataset/evalImageSet     --use_gpu=True')


# ResNet50预测

# In[ ]:


get_ipython().system('python tools/infer/submit.py     --model_file inference/ResNet50/inference.pdmodel     --params_file inference/ResNet50/inference.pdiparams     --image_file ./dataset/dataset/evalImageSet     # --image_file ./dataset/dataset/5/20170913-104812-3.jpg \\')
    --use_gpu=True


# In[ ]:


get_ipython().system('cp ../predict.py tools/infer/submit.py')


# /MobileNetV3_large_x1_0预测

# In[ ]:


get_ipython().system('python tools/infer/submit.py     --model_file inference/MobileNetv3_large_x1/inference.pdmodel     --params_file inference/MobileNetv3_large_x1/inference.pdiparams     --image_file ./dataset/dataset/evalImageSet     --use_gpu=True')


# resnet50_vd_ssld+w预测

# In[ ]:


get_ipython().system('python tools/infer/submit.py     --model_file inference/inference.pdmodel     --params_file inference/inference.pdiparams     --image_file ./dataset/dataset/evalImageSet     --use_gpu=True')


# 进行预测/ResNet50_vd_distill_MobileNetV3_large_x1_0

# In[ ]:


get_ipython().run_line_magic('cd', 'PaddleClas')


# In[ ]:


get_ipython().system('python tools/infer/submit.py     --model_file inference/ResNet50_vd_distill_MobileNetV3_large_x1_0/inference.pdmodel     --params_file inference/ResNet50_vd_distill_MobileNetV3_large_x1_0/inference.pdiparams     --image_file ./dataset/dataset/evalImageSet     --use_gpu=True')


# In[ ]:


get_ipython().system('python tools/infer/submit.py     --model_file ./inference/ResNet50_vd_distill_MobileNetV3_large_x1_0_ssld_1/inference.pdmodel     --params_file ./inference/ResNet50_vd_distill_MobileNetV3_large_x1_0_ssld_1/inference.pdiparams     --image_file ./dataset/dataset/evalImageSet     --use_gpu=True')


# 推理单张图片

# In[ ]:


get_ipython().run_line_magic('cd', 'PaddleClas/')
get_ipython().system('python tools/infer/predict.py     --model_file ./inference/ResNet50_vd_distill_MobileNetV3_large_x1_0_ssld_1/inference.pdmodel     --params_file ./inference/ResNet50_vd_distill_MobileNetV3_large_x1_0_ssld_1/inference.pdiparams     -i=./dataset/dataset/8/20170128-101909-0.jpg     --use_gpu=True')


# 七 效果展示
#    通过代码命令，代码调用paddleclass中的模型，并对模型中的参数进行更改，调整结构。试验结果表明，本文轻量模型相比移动端模型MobileNetV3模型，模型大小变化不大的情况下，识别准确率提升了1.2%；相比服务器端模型ResNet50，准确率提升0.78%，平均每张推理时间减少7.8%，模型大小减少80%，本研究可为杂草精准施药的实施应用提供理论基础和技术支持。![](https://ai-studio-static-online.cdn.bcebos.com/84a0b6d806154da3a3aa2c65bad9a88e0d7ec651787f4c5f870877ebc9be37a0)
#   ![](https://ai-studio-static-online.cdn.bcebos.com/bda025813a474a808b7830eacb62038a5f090e9ea3534966bb84380a0a9e3caa)
#   
# 

# 八 总结与升华
# 	保存模型时，自己命名不注意，保存混乱，今后多注意保存命名的设置。
#    图像的可视化自己也弄了很久，当时是项目试运行，动静态模型显示方法不同，同学们要多注意，务必确保和自己的版本相符。
#    通过项目的学习，对深度学习有了一个更加整体和深入的了解。

# 陈启   19级 三峡大学研究生  aistudio账号cq18727163960   欢迎交流

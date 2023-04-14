# PaddleClas服务化部署示例

PaddleClas 服务化部署示例是利用FastDeploy Serving搭建的服务化部署示例。FastDeploy Serving是基于Triton Inference Server框架封装的适用于高并发、高吞吐量请求的服务化部署框架，是一套可用于实际生产的完备且性能卓越的服务化部署框架.

## 1. 部署环境准备
在服务化部署前，需确认服务化镜像的软硬件环境要求和镜像拉取命令，请参考[FastDeploy服务化部署](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/README_CN.md)

## 2. 启动服务

```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/classification/paddleclas/serving

# 如果您希望从PaddleClas下载示例代码，请运行
git clone https://github.com/PaddlePaddle/PaddleClas.git
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
git checkout develop
cd PaddleClas/deploy/fastdeploy/serving

# 下载ResNet50_vd模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# 将配置文件放入预处理目录
mv ResNet50_vd_infer/inference_cls.yaml models/preprocess/1/inference_cls.yaml

# 将模型放入 models/runtime/1目录下, 并重命名为model.pdmodel和model.pdiparams
mv ResNet50_vd_infer/inference.pdmodel models/runtime/1/model.pdmodel
mv ResNet50_vd_infer/inference.pdiparams models/runtime/1/model.pdiparams

# 拉取fastdeploy镜像(x.y.z为镜像版本号，需参照serving文档替换为数字)
# GPU镜像
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10
# CPU镜像
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-cpu-only-21.10

# 运行容器.容器名字为 fd_serving, 并挂载当前目录为容器的 /serving 目录
nvidia-docker run -it --net=host --name fd_serving -v `pwd`/:/serving registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10  bash

# 启动服务(不设置CUDA_VISIBLE_DEVICES环境变量，会拥有所有GPU卡的调度权限)
CUDA_VISIBLE_DEVICES=0 fastdeployserver --model-repository=/serving/models --backend-config=python,shm-default-byte-size=10485760
```
>> **注意**:

>> 拉取其他硬件上的镜像请看[服务化部署主文档](../../../../../serving/README_CN.md)

>> 执行fastdeployserver启动服务出现"Address already in use", 请使用`--grpc-port`指定端口号来启动服务，同时更改客户端示例中的请求端口号.

>> 其他启动参数可以使用 fastdeployserver --help 查看

服务启动成功后， 会有以下输出:
```
......
I0928 04:51:15.784517 206 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I0928 04:51:15.785177 206 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I0928 04:51:15.826578 206 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```


## 3. 客户端请求

在物理机器中执行以下命令，发送grpc请求并输出结果
```
#下载测试图片
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

#安装客户端依赖
python3 -m pip install tritonclient\[all\]

# 发送请求
python3 paddlecls_grpc_client.py
```

发送请求成功后，会返回json格式的检测结果并打印输出:
```
output_name: CLAS_RESULT
{'label_ids': [153], 'scores': [0.6862289905548096]}
```

## 4. 配置修改

当前默认配置在GPU上运行TensorRT引擎， 如果要在CPU或其他推理引擎上运行。 需要修改`models/runtime/config.pbtxt`中配置，详情请参考[配置文档](../../../../../serving/docs/zh_CN/model_configuration.md)

## 5. 使用VisualDL进行可视化部署

可以[使用 VisualDL 进行 Serving 可视化部署](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/vdl_management.md)，上述启动服务、配置修改以及客户端请求的操作都可以基于VisualDL进行。

通过VisualDL的可视化界面对PaddleClas进行服务化部署只需要如下三步：
```text
1. 载入模型库：PaddleClas/deploy/fastdeploy/serving/models
2. 下载模型资源文件：点击runtime模型，点击版本号1添加预训练模型，选择图像分类模型ResNet50_vd进行下载。
3. 启动服务：点击启动服务按钮，输入启动参数。
```
 <p align="center">
  <img src="https://user-images.githubusercontent.com/22424850/211708702-828d8ad8-4e85-457f-9c62-12f53fc81853.gif" width="100%"/>
</p>

## 6. 常见问题
- [如何编写客户端 HTTP/GRPC 请求](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/client.md)
- [如何编译服务化部署镜像](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/compile.md)
- [服务化部署原理及动态Batch介绍](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/demo.md)
- [模型仓库介绍](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/model_repository.md)

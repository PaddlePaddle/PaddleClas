# PaddleClas rv1126 开发板 C++ 部署示例
本目录下提供的 `infer.cc`，可以帮助用户快速完成 PaddleClas 量化模型在 rv1126 上的部署推理加速。

## 1. 部署环境准备
在部署前，需确认以下两个步骤
- 1. 在部署前，需自行编译基于rv的预测库，参考文档[rv1126部署环境编译](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#自行编译安装)

## 2. 量化模型准备
1. 需要特别注意的是，在 rv1126 上部署的模型需要是量化后的模型. 用户可以直接使用由 FastDeploy 提供的量化模型进行部署。
2. 用户也可以使用 FastDeploy 提供的[一键模型自动化压缩工具](https://github.com/PaddlePaddle/FastDeploy/tree/develop/tools/common_tools/auto_compression/)，自行进行模型量化, 并使用产出的量化模型进行部署。(注意: 推理量化后的分类模型仍然需要FP32模型文件夹下的inference_cls.yaml文件, 自行量化的模型文件夹内不包含此 yaml 文件, 用户从 FP32 模型文件夹下复制此 yaml 文件到量化后的模型文件夹内即可.)

## 3. 在 RV1126 上部署量化后的 ResNet50_Vd 分类模型
请按照以下步骤完成在 RV1126 上部署 ResNet50_Vd 量化模型：
1. 交叉编译编译 FastDeploy 库，具体请参考：[交叉编译 FastDeploy](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/rv1126.md#基于-paddlelite-的-fastdeploy-交叉编译库编译)

2. 将编译后的库拷贝到当前目录，可使用如下命令：
```bash
cp -r FastDeploy/build/fastdeploy-timvx/ PaddleClas/deploy/fastdeploy/rockchip/rv1126/cpp/
```

3. 在当前路径下载部署所需的模型和示例图片：
```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/classification/paddleclas/rockchip/rv1126/cpp/

# 如果您希望从PaddleClas下载示例代码，请运行
git clone https://github.com/PaddlePaddle/PaddleClas.git
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
git checkout develop
cd PaddleClas/deploy/fastdeploy/rockchip/rv1126/cpp/

mkdir models && mkdir images
wget https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar
tar -xvf resnet50_vd_ptq.tar
cp -r resnet50_vd_ptq models
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg
cp -r ILSVRC2012_val_00000010.jpeg images
```

4. 编译部署示例，可使入如下命令：
```bash
cd PaddleClas/deploy/fastdeploy/rockchip/rv1126/cpp/
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=${PWD}/../fastdeploy-timvx/toolchain.cmake -DFASTDEPLOY_INSTALL_DIR=${PWD}/../fastdeploy-timvx -DTARGET_ABI=armhf ..
make -j8
make install
# 成功编译之后，会生成 install 文件夹，里面有一个运行 demo 和部署所需的库
```

5. 基于 adb 工具部署 ResNet50 分类模型到 Rockchip RV1126，可使用如下命令：
```bash
# 进入 install 目录
cd PaddleClas/deploy/fastdeploy/rockchip/rv1126/cpp/build/install/
# 如下命令表示：bash run_with_adb.sh 需要运行的demo 模型路径 图片路径 设备的DEVICE_ID
bash run_with_adb.sh infer_demo resnet50_vd_ptq ILSVRC2012_val_00000010.jpeg $DEVICE_ID
```

部署成功后运行结果如下：

<img width="640" src="https://user-images.githubusercontent.com/30516196/200767389-26519e50-9e4f-4fe1-8d52-260718f73476.png">

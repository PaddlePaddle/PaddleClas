# PaddleClas SOPHGO部署示例

## 1. 说明  
PaddleClas支持通过FastDeploy在SOPHGO上部署相关模型.

## 2. 支持模型列表

目前FastDeploy支持的如下模型的部署[ResNet系列模型](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/ResNet_and_vd.md)

## 3.准备ResNet部署模型以及转换模型

SOPHGO-TPU部署模型前需要将Paddle模型转换成bmodel模型，具体步骤如下:
- Paddle动态图模型转换为ONNX模型，请参考[Paddle2ONNX模型转换](https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/model_zoo/classification)
- ONNX模型转换bmodel模型的过程，请参考[TPU-MLIR](https://github.com/sophgo/tpu-mlir)。

下面以[ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz)为例子,教大家如何转换Paddle模型到SOPHGO-TPU模型。

### 3.1 导出ONNX模型

#### 3.1.1 下载Paddle ResNet50_vd静态图模型并解压
```shell
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar xvf ResNet50_vd_infer.tgz
```

#### 3.1.2  静态图转ONNX模型，注意，这里的save_file请和压缩包名对齐
```shell
paddle2onnx --model_dir ResNet50_vd_infer \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ResNet50_vd_infer.onnx \
            --enable_dev_version True
```

### 3.2 导出bmodel模型

以转化BM1684x的bmodel模型为例子，我们需要下载[TPU-MLIR](https://github.com/sophgo/tpu-mlir)工程，安装过程具体参见[TPU-MLIR文档](https://github.com/sophgo/tpu-mlir/blob/master/README.md)。

#### 3.2.1	安装
``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234是一个示例，也可以设置其他名字
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest

source ./envsetup.sh
./build.sh
```

#### 3.2.2	ONNX模型转换为bmodel模型
``` shell
mkdir ResNet50_vd_infer && cd ResNet50_vd_infer

# 在该文件中放入测试图片，同时将上一步转换好的ResNet50_vd_infer.onnx放入该文件夹中
cp -rf ${REGRESSION_PATH}/dataset/COCO2017 .
cp -rf ${REGRESSION_PATH}/image .
# 放入onnx模型文件ResNet50_vd_infer.onnx

mkdir workspace && cd workspace

# 将ONNX模型转换为mlir模型，其中参数--output_names可以通过NETRON查看
model_transform.py \
    --model_name ResNet50_vd_infer \
    --model_def ../ResNet50_vd_infer.onnx \
    --input_shapes [[1,3,224,224]] \
    --mean 0.0,0.0,0.0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --output_names save_infer_model/scale_0.tmp_1 \
    --test_input ../image/dog.jpg \
    --test_result ResNet50_vd_infer_top_outputs.npz \
    --mlir ResNet50_vd_infer.mlir

# 将mlir模型转换为BM1684x的F32 bmodel模型
model_deploy.py \
  --mlir ResNet50_vd_infer.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input ResNet50_vd_infer_in_f32.npz \
  --test_reference ResNet50_vd_infer_top_outputs.npz \
  --model ResNet50_vd_infer_1684x_f32.bmodel
```
最终获得可以在BM1684x上能够运行的bmodel模型ResNet50_vd_infer_1684x_f32.bmodel。如果需要进一步对模型进行加速，可以将ONNX模型转换为INT8 bmodel，具体步骤参见[TPU-MLIR文档](https://github.com/sophgo/tpu-mlir/blob/master/README.md)。

## 4. 其他链接
- [Python部署](python)
- [C++部署](cpp)

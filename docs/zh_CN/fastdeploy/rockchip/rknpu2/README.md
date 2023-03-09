
# PaddleClas 模型在RKNPU2上部署方案-FastDeploy

## 1. 说明  
PaddleClas支持通过FastDeploy在RKNPU2上部署相关模型.

## 2. 转换模型
下面以 ResNet50_vd为例子，教大家如何转换分类模型到RKNN模型.
目前FastDeploy只在PaddleClas测试过ResNet50_vd模型, 欢迎用户尝试其他的PaddleClas模型进行部署.

### 2.1 导出ONNX模型
```bash
# 安装 paddle2onnx
pip install paddle2onnx

# 下载ResNet50_vd模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz

# 静态图转ONNX模型，注意，这里的save_file请和压缩包名对齐
paddle2onnx --model_dir ResNet50_vd_infer  \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams  \
            --save_file ResNet50_vd_infer/ResNet50_vd_infer.onnx  \
            --enable_dev_version True  \
            --opset_version 10  \
            --enable_onnx_checker True

# 固定shape，注意这里的inputs得对应netron.app展示的 inputs 的 name，有可能是image 或者 x
python -m paddle2onnx.optimize --input_model ResNet50_vd_infer/ResNet50_vd_infer.onnx \
                               --output_model ResNet50_vd_infer/ResNet50_vd_infer.onnx \
                               --input_shape_dict "{'inputs':[1,3,224,224]}"
```  

### 2.2 编写模型导出配置文件
以转化RK3588的RKNN模型为例子，我们需要编辑./rknpu2_tools/config/ResNet50_vd_infer_rknn.yaml，来转换ONNX模型到RKNN模型。

如果你需要在NPU上执行normalize操作，请根据你的模型配置normalize参数，例如:
```yaml
model_path: ./ResNet50_vd_infer/ResNet50_vd_infer.onnx
output_folder: ./ResNet50_vd_infer
mean:
  -
    - 123.675
    - 116.28
    - 103.53
std:
  -
    - 58.395
    - 57.12
    - 57.375
outputs_nodes:
do_quantization: False
dataset: "./ResNet50_vd_infer/dataset.txt"
```

**在CPU上做normalize**可以参考以下yaml：
```yaml
model_path: ./ResNet50_vd_infer/ResNet50_vd_infer.onnx
output_folder: ./ResNet50_vd_infer
mean:
  -
    - 0
    - 0
    - 0
std:
  -
    - 1
    - 1
    - 1
outputs_nodes:
do_quantization: False
dataset: "./ResNet50_vd_infer/dataset.txt"
```
这里我们选择在NPU上执行normalize操作.


### 2.3 ONNX模型转RKNN模型
```shell
python ./rknpu2_tools/export.py \
        --config_path ./rknpu2_tools/config/ResNet50_vd_infer_rknn.yaml \
        --target_platform rk3588
```

## 3. 其他链接
- [Cpp部署](./cpp)
- [Python部署](./python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)

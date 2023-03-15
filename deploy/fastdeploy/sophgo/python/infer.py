import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path of model.")
    parser.add_argument(
        "--config_file", required=True, help="Path of config file.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
    parser.add_argument(
        "--topk", type=int, default=1, help="Return topk results.")

    return parser.parse_args()


args = parse_arguments()

# 配置runtime，加载模型
runtime_option = fd.RuntimeOption()
runtime_option.use_sophgo()

model_file = args.model
params_file = ""
config_file = args.config_file

model = fd.vision.classification.PaddleClasModel(
    model_file,
    params_file,
    config_file,
    runtime_option=runtime_option,
    model_format=fd.ModelFormat.SOPHGO)

# 预测图片分类结果
im = cv2.imread(args.image)
result = model.predict(im, args.topk)
print(result)

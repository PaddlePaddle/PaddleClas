import fastdeploy as fd
import cv2
import os
from subprocess import run


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--auto",
        required=True,
        help="Auto download, convert, compile and infer if True")
    parser.add_argument("--model", required=True, help="Path of bmodel")
    parser.add_argument(
        "--config_file", required=True, help="Path of config file")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
    parser.add_argument(
        "--topk", type=int, default=1, help="Return topk results.")

    return parser.parse_args()


def download():
    cmd_str = 'wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz'
    jpg_str = 'wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg'
    tar_str = 'tar xvf ResNet50_vd_infer.tgz'
    if not os.path.exists('ResNet50_vd_infer.tgz'):
        run(cmd_str, shell=True)
    if not os.path.exists('ILSVRC2012_val_00000010.jpeg'):
        run(jpg_str, shell=True)
    run(tar_str, shell=True)


def paddle2onnx():
    cmd_str = 'paddle2onnx --model_dir ResNet50_vd_infer \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ResNet50_vd_infer.onnx \
            --enable_dev_version True'

    print(cmd_str)
    run(cmd_str, shell=True)


def mlir_prepare():
    mlir_path = os.getenv("MODEL_ZOO_PATH")
    mlir_path = mlir_path[:-13]
    cmd_list = [
        'mkdir ResNet50', 'cp -rf ' + os.path.join(
            mlir_path, 'regression/dataset/COCO2017/') + ' ./ResNet50',
        'cp -rf ' + os.path.join(mlir_path,
                                 'regression/image/') + ' ./ResNet50',
        'cp ResNet50_vd_infer.onnx ./ResNet50/', 'mkdir ./ResNet50/workspace'
    ]
    for str in cmd_list:
        print(str)
        run(str, shell=True)


def onnx2mlir():
    cmd_str = 'model_transform.py \
        --model_name ResNet50_vd_infer \
        --model_def ../ResNet50_vd_infer.onnx \
        --input_shapes [[1,3,224,224]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb \
        --output_names save_infer_model/scale_0.tmp_1 \
        --test_input ../image/dog.jpg \
        --test_result ./ResNet50_vd_infer_top_outputs.npz \
        --mlir ./ResNet50_vd_infer.mlir'

    print(cmd_str)
    os.chdir('./ResNet50/workspace/')
    run(cmd_str, shell=True)
    os.chdir('../../')


def mlir2bmodel():
    cmd_str = 'model_deploy.py \
        --mlir ./ResNet50_vd_infer.mlir \
        --quantize F32 \
        --chip bm1684x \
        --test_input ./ResNet50_vd_infer_in_f32.npz \
        --test_reference ./ResNet50_vd_infer_top_outputs.npz \
        --model ./ResNet50_vd_infer_1684x_f32.bmodel'

    print(cmd_str)
    os.chdir('./ResNet50/workspace')
    run(cmd_str, shell=True)
    os.chdir('../../')


args = parse_arguments()

if (args.auto):
    download()
    paddle2onnx()
    mlir_prepare()
    onnx2mlir()
    mlir2bmodel()

# config runtime and load the model
runtime_option = fd.RuntimeOption()
runtime_option.use_sophgo()

model_file = './ResNet50/workspace/ResNet50_vd_infer_1684x_f32.bmodel' if args.auto else args.model
params_file = ""
config_file = './ResNet50_vd_infer/inference_cls.yaml' if args.auto else args.config_file
image_file = './ILSVRC2012_val_00000010.jpeg' if args.auto else args.image
model = fd.vision.classification.PaddleClasModel(
    model_file,
    params_file,
    config_file,
    runtime_option=runtime_option,
    model_format=fd.ModelFormat.SOPHGO)

# predict the results of image classification
im = cv2.imread(image_file)
result = model.predict(im, args.topk)
print(result)

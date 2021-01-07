# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
__dir__ = os.path.dirname(__file__)
sys.path.append(os.path.join(__dir__, ''))

import cv2
import numpy as np
import tarfile
import requests
from tqdm import tqdm
import tools.infer.utils as utils
import shutil
__all__ = ['PaddleClas']
BASE_DIR = os.path.expanduser("~/.paddleclas/")
BASE_INFERENCE_MODEL_DIR = os.path.join(BASE_DIR, 'inference_model')
BASE_IMAGES_DIR = os.path.join(BASE_DIR, 'images')

model_names = {
    'Xception71', 'SE_ResNeXt101_32x4d', 'ShuffleNetV2_x0_5', 'ResNet34',
    'ShuffleNetV2_x2_0', 'ResNeXt101_32x4d', 'HRNet_W48_C_ssld',
    'ResNeSt50_fast_1s1x64d', 'MobileNetV2_x2_0', 'MobileNetV3_large_x1_0',
    'Fix_ResNeXt101_32x48d_wsl', 'MobileNetV2_ssld', 'ResNeXt101_vd_64x4d',
    'ResNet34_vd_ssld', 'MobileNetV3_small_x1_0', 'VGG11',
    'ResNeXt50_vd_32x4d', 'MobileNetV3_large_x1_25',
    'MobileNetV3_large_x1_0_ssld', 'MobileNetV2_x0_75',
    'MobileNetV3_small_x0_35', 'MobileNetV1_x0_75', 'MobileNetV1_ssld',
    'ResNeXt50_32x4d', 'GhostNet_x1_3_ssld', 'Res2Net101_vd_26w_4s',
    'ResNet152', 'Xception65', 'EfficientNetB0', 'ResNet152_vd', 'HRNet_W18_C',
    'Res2Net50_14w_8s', 'ShuffleNetV2_x0_25', 'HRNet_W64_C',
    'Res2Net50_vd_26w_4s_ssld', 'HRNet_W18_C_ssld', 'ResNet18_vd',
    'ResNeXt101_32x16d_wsl', 'SE_ResNeXt50_32x4d', 'SqueezeNet1_1',
    'SENet154_vd', 'SqueezeNet1_0', 'GhostNet_x1_0', 'ResNet50_vc', 'DPN98',
    'HRNet_W48_C', 'DenseNet264', 'SE_ResNet34_vd', 'HRNet_W44_C',
    'MobileNetV3_small_x1_25', 'MobileNetV1_x0_5', 'ResNet200_vd', 'VGG13',
    'EfficientNetB3', 'EfficientNetB2', 'ShuffleNetV2_x0_33',
    'MobileNetV3_small_x0_75', 'ResNeXt152_vd_32x4d', 'ResNeXt101_32x32d_wsl',
    'ResNet18', 'MobileNetV3_large_x0_35', 'Res2Net50_26w_4s',
    'MobileNetV2_x0_5', 'EfficientNetB0_small', 'ResNet101_vd_ssld',
    'EfficientNetB6', 'EfficientNetB1', 'EfficientNetB7', 'ResNeSt50',
    'ShuffleNetV2_x1_0', 'MobileNetV3_small_x1_0_ssld', 'InceptionV4',
    'GhostNet_x0_5', 'SE_HRNet_W64_C_ssld', 'ResNet50_ACNet_deploy',
    'Xception41', 'ResNet50', 'Res2Net200_vd_26w_4s_ssld',
    'Xception41_deeplab', 'SE_ResNet18_vd', 'SE_ResNeXt50_vd_32x4d',
    'HRNet_W30_C', 'HRNet_W40_C', 'VGG19', 'Res2Net200_vd_26w_4s',
    'ResNeXt101_32x8d_wsl', 'ResNet50_vd', 'ResNeXt152_64x4d', 'DarkNet53',
    'ResNet50_vd_ssld', 'ResNeXt101_64x4d', 'MobileNetV1_x0_25',
    'Xception65_deeplab', 'AlexNet', 'ResNet101', 'DenseNet121',
    'ResNet50_vd_v2', 'Res2Net50_vd_26w_4s', 'ResNeXt101_32x48d_wsl',
    'MobileNetV3_large_x0_5', 'MobileNetV2_x0_25', 'DPN92', 'ResNet101_vd',
    'MobileNetV2_x1_5', 'DPN131', 'ResNeXt50_vd_64x4d', 'ShuffleNetV2_x1_5',
    'ResNet34_vd', 'MobileNetV1', 'ResNeXt152_vd_64x4d', 'DPN107', 'VGG16',
    'ResNeXt50_64x4d', 'RegNetX_4GF', 'DenseNet161', 'GhostNet_x1_3',
    'HRNet_W32_C', 'Fix_ResNet50_vd_ssld_v2', 'Res2Net101_vd_26w_4s_ssld',
    'DenseNet201', 'DPN68', 'EfficientNetB4', 'ResNeXt152_32x4d',
    'InceptionV3', 'ShuffleNetV2_swish', 'GoogLeNet', 'ResNet50_vd_ssld_v2',
    'SE_ResNet50_vd', 'MobileNetV2', 'ResNeXt101_vd_32x4d',
    'MobileNetV3_large_x0_75', 'MobileNetV3_small_x0_5', 'DenseNet169',
    'EfficientNetB5'
}


def download_with_progressbar(url, save_path):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes == 0 or progress_bar.n != total_size_in_bytes:
        raise Exception("Something went wrong while downloading models")


def maybe_download(model_storage_directory, url):
    # using custom model
    tar_file_name_list = [
        'inference.pdiparams', 'inference.pdiparams.info', 'inference.pdmodel'
    ]
    if not os.path.exists(
            os.path.join(model_storage_directory, 'inference.pdiparams')
    ) or not os.path.exists(
            os.path.join(model_storage_directory, 'inference.pdmodel')):
        tmp_path = os.path.join(model_storage_directory, url.split('/')[-1])
        print('download {} to {}'.format(url, tmp_path))
        os.makedirs(model_storage_directory, exist_ok=True)
        download_with_progressbar(url, tmp_path)
        with tarfile.open(tmp_path, 'r') as tarObj:
            for member in tarObj.getmembers():
                filename = None
                for tar_file_name in tar_file_name_list:
                    if tar_file_name in member.name:
                        filename = tar_file_name
                if filename is None:
                    continue
                file = tarObj.extractfile(member)
                with open(
                        os.path.join(model_storage_directory, filename),
                        'wb') as f:
                    f.write(file.read())
        os.remove(tmp_path)


def save_prelabel_results(class_id, input_filepath, output_idr):
    output_dir = os.path.join(output_idr, str(class_id))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    shutil.copy(input_filepath, output_dir)


def load_label_name_dict(path):
    result = {}
    if not os.path.exists(path):
        print(
            'Warning: If want to use your own label_dict, please input legal path!\nOtherwise label_names will be empty!'
        )
    else:
        for line in open(path, 'r'):
            partition = line.split('\n')[0].partition(' ')
            try:
                result[int(partition[0])] = str(partition[-1])
            except:
                result = {}
                break
    return result


def parse_args(mMain=True, add_help=True):
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    if mMain == True:

        # general params
        parser = argparse.ArgumentParser(add_help=add_help)
        parser.add_argument("--model_name", type=str)
        parser.add_argument("-i", "--image_file", type=str)
        parser.add_argument("--use_gpu", type=str2bool, default=False)

        # params for preprocess
        parser.add_argument("--resize_short", type=int, default=256)
        parser.add_argument("--resize", type=int, default=224)
        parser.add_argument("--normalize", type=str2bool, default=True)
        parser.add_argument("-b", "--batch_size", type=int, default=1)

        # params for predict
        parser.add_argument(
            "--model_file", type=str, default='')  ## inference.pdmodel
        parser.add_argument(
            "--params_file", type=str, default='')  ## inference.pdiparams
        parser.add_argument("--ir_optim", type=str2bool, default=True)
        parser.add_argument("--use_fp16", type=str2bool, default=False)
        parser.add_argument("--use_tensorrt", type=str2bool, default=False)
        parser.add_argument("--gpu_mem", type=int, default=8000)
        parser.add_argument("--enable_profile", type=str2bool, default=False)
        parser.add_argument("--top_k", type=int, default=1)
        parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
        parser.add_argument("--enable_benchmark", type=str2bool, default=False)
        parser.add_argument("--cpu_num_threads", type=int, default=10)
        parser.add_argument("--hubserving", type=str2bool, default=False)

        # parameters for pre-label the images
        parser.add_argument("--label_name_path", type=str, default='')
        parser.add_argument(
            "--pre_label_image",
            type=str2bool,
            default=False,
            help="Whether to pre-label the images using the loaded weights")
        parser.add_argument("--pre_label_out_idr", type=str, default=None)

        return parser.parse_args()
    else:
        return argparse.Namespace(
            model_name='',
            image_file='',
            use_gpu=False,
            use_fp16=False,
            use_tensorrt=False,
            resize_short=256,
            resize=224,
            normalize=True,
            batch_size=1,
            model_file='',
            params_file='',
            ir_optim=True,
            gpu_mem=8000,
            enable_profile=False,
            top_k=1,
            enable_mkldnn=False,
            enable_benchmark=False,
            cpu_num_threads=10,
            hubserving=False,
            label_name_path='',
            pre_label_image=False,
            pre_label_out_idr=None)


class PaddleClas(object):
    print('Inference models that Paddle provides are listed as follows:\n\n{}'.
          format(model_names), '\n')

    def __init__(self, **kwargs):

        process_params = parse_args(mMain=False, add_help=False)
        process_params.__dict__.update(**kwargs)

        if not os.path.exists(process_params.model_file):
            if process_params.model_name is None:
                raise Exception(
                    'Please input model name that you want to use!')
            if process_params.model_name in model_names:
                url = 'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/{}_infer.tar'.format(
                    process_params.model_name)

                if not os.path.exists(
                        os.path.join(BASE_INFERENCE_MODEL_DIR,
                                     process_params.model_name)):
                    os.makedirs(
                        os.path.join(BASE_INFERENCE_MODEL_DIR,
                                     process_params.model_name))
                download_path = os.path.join(BASE_INFERENCE_MODEL_DIR,
                                             process_params.model_name)
                maybe_download(model_storage_directory=download_path, url=url)
                process_params.model_file = os.path.join(download_path,
                                                         'inference.pdmodel')
                process_params.params_file = os.path.join(
                    download_path, 'inference.pdiparams')
                process_params.label_name_path = os.path.join(
                    __dir__, 'ppcls/utils/imagenet1k_label_list.txt')
            else:
                raise Exception(
                    'If you want to use your own model, Please input model_file as model path!'
                )
        else:
            print('Using user-specified model and params!')

        print("process params are as follows: \n{}".format(process_params))
        self.label_name_dict = load_label_name_dict(
            process_params.label_name_path)

        self.args = process_params
        self.predictor = utils.create_paddle_predictor(process_params)

    def predict(self, img):
        """
        predict label of img with paddleclas
        Args:
            img: input image for clas, support single image , internet url, folder path containing series of images
        Returns:
            dict：{image_name: "", class_id: [], scores: [], label_names: []}，if label name path == None，label_names will be empty.
        """
        assert isinstance(img, (str, np.ndarray))

        input_names = self.predictor.get_input_names()
        input_tensor = self.predictor.get_input_handle(input_names[0])

        output_names = self.predictor.get_output_names()
        output_tensor = self.predictor.get_output_handle(output_names[0])
        if isinstance(img, str):
            # download internet image
            if img.startswith('http'):
                if not os.path.exists(BASE_IMAGES_DIR):
                    os.makedirs(BASE_IMAGES_DIR)
                image_path = os.path.join(BASE_IMAGES_DIR, 'tmp.jpg')
                download_with_progressbar(img, image_path)
                print("Current using image from Internet:{}, renamed as: {}".
                      format(img, image_path))
                img = image_path
            image_list = utils.get_image_list(img)
        else:
            if isinstance(img, np.ndarray):
                image_list = [img]
            else:
                print('Please input legal image!')

        total_result = []
        for filename in image_list:
            if isinstance(filename, str):
                image = cv2.imread(filename)[:, :, ::-1]
                assert image is not None, "Error in loading image: {}".format(
                    filename)
                inputs = utils.preprocess(image, self.args)
                inputs = np.expand_dims(
                    inputs, axis=0).repeat(
                        1, axis=0).copy()
            else:
                inputs = filename

            input_tensor.copy_from_cpu(inputs)

            self.predictor.run()

            outputs = output_tensor.copy_to_cpu()
            classes, scores = utils.postprocess(outputs, self.args)
            label_names = []
            if len(self.label_name_dict) != 0:
                label_names = [self.label_name_dict[c] for c in classes]
            result = {
                "filename": filename if isinstance(filename, str) else 'image',
                "class_ids": classes.tolist(),
                "scores": scores.tolist(),
                "label_names": label_names,
            }
            total_result.append(result)
            if self.args.pre_label_image:
                save_prelabel_results(classes[0], filename,
                                      self.args.pre_label_out_idr)
                print("\tSaving prelabel results in {}".format(
                    os.path.join(self.args.pre_label_out_idr, str(classes[
                        0]))))
        return total_result


def main():
    # for cmd
    args = parse_args(mMain=True)
    clas_engine = PaddleClas(**(args.__dict__))
    print('{}{}{}'.format('*' * 10, args.image_file, '*' * 10))
    result = clas_engine.predict(args.image_file)
    if result is not None:
        print(result)


if __name__ == '__main__':
    main()

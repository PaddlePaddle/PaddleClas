# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import time
import sys
import argparse
import numpy as np
import cv2
import yaml

import paddle
from paddle.inference import create_predictor
from paddle.io import DataLoader
from imagenet_reader import ImageNetDataset


def argsparser():
    """
    argsparser func
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path",
        type=str,
        default="./MobileNetV1_infer",
        help="model directory")
    parser.add_argument(
        "--model_filename",
        type=str,
        default="inference.pdmodel",
        help="model file name")
    parser.add_argument(
        "--params_filename",
        type=str,
        default="inference.pdiparams",
        help="params file name")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument(
        "--data_path", type=str, default="./dataset/ILSVRC2012/")
    parser.add_argument(
        "--use_gpu", type=bool, default=False, help="Whether to use gpu")
    parser.add_argument(
        "--use_trt", type=bool, default=False, help="Whether to use tensorrt")
    parser.add_argument(
        "--use_mkldnn", type=bool, default=False, help="Whether to use mkldnn")
    parser.add_argument(
        "--cpu_num_threads",
        type=int,
        default=10,
        help="Number of cpu threads")
    parser.add_argument(
        "--use_fp16", type=bool, default=False, help="Whether to use fp16")
    parser.add_argument(
        "--use_int8", type=bool, default=False, help="Whether to use int8")
    parser.add_argument("--gpu_mem", type=int, default=8000, help="GPU memory")
    parser.add_argument("--ir_optim", type=bool, default=True)
    parser.add_argument(
        "--use_dynamic_shape",
        type=bool,
        default=True,
        help="Whether use dynamic shape or not.")
    parser.add_argument(
        "--min_subgraph_size", 
        type=int, 
        default=30, 
        help="Minimum Subgraph Size for TensorRT Acceleration"
    )
    return parser


def eval_reader(data_dir, batch_size, crop_size, resize_size, args, config=None):
    """
    eval reader func
    """
    # 这样加载数据的方式慢很多
    # device = 'gpu' if args.use_gpu else 'cpu'
    # use_dali = False
    # eval_dataloader = build_dataloader(
    #                 config["DataLoader"], "Eval", device,
    #                 use_dali)
    # return eval_dataloader

    val_reader = ImageNetDataset(
        mode="val",
        data_dir=data_dir,
        crop_size=crop_size,
        resize_size=resize_size)

    val_loader = DataLoader(
        val_reader,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=12)
    return val_loader


class Predictor(object):
    """
    Paddle Inference Predictor class
    """

    def __init__(self):
        # HALF precission predict only work when using tensorrt
        if args.use_fp16 is True:
            assert args.use_trt is True

        self.rerun_flag = False
        self.paddle_predictor = self._create_paddle_predictor()
        input_names = self.paddle_predictor.get_input_names()
        self.input_tensor = self.paddle_predictor.get_input_handle(input_names[
            0])

        output_names = self.paddle_predictor.get_output_names()
        self.output_tensor = self.paddle_predictor.get_output_handle(
            output_names[0])

    def _create_paddle_predictor(self):
        inference_model_dir = args.model_path
        model_file = os.path.join(inference_model_dir, args.model_filename)
        params_file = os.path.join(inference_model_dir, args.params_filename)
        config = paddle.inference.Config(model_file, params_file)
        precision = paddle.inference.Config.Precision.Float32
        if args.use_int8:
            precision = paddle.inference.Config.Precision.Int8
        elif args.use_fp16:
            precision = paddle.inference.Config.Precision.Half

        if args.use_gpu:
            config.enable_use_gpu(args.gpu_mem, 0)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(args.cpu_num_threads)
            if args.use_mkldnn:
                config.enable_mkldnn()
                if args.use_int8:
                    config.enable_mkldnn_int8({
                        "conv2d", "depthwise_conv2d"
                    })

        config.switch_ir_optim(args.ir_optim)  # default true
        if args.use_trt:
            config.enable_tensorrt_engine(
                precision_mode=precision,
                max_batch_size=args.batch_size,
                workspace_size=1 << 30,
                min_subgraph_size=args.min_subgraph_size,
                use_static=True,
                use_calib_mode=False, )

            if args.use_dynamic_shape:
                dynamic_shape_file = os.path.join(inference_model_dir,
                                                  "dynamic_shape.txt")
                if os.path.exists(dynamic_shape_file):
                    config.enable_tuned_tensorrt_dynamic_shape(
                        dynamic_shape_file, True)
                    print("trt set dynamic shape done!")
                else:
                    config.collect_shape_range_info(dynamic_shape_file)
                    print("Start collect dynamic shape...")
                    self.rerun_flag = True

        config.enable_memory_optim()
        # When performing inference on the CPU for the CLIP_vit_base_patch16_224 and 
        # SwinTransformer_base_patch4_window7_224 models, it is necessary to remove the following two passes 
        # in order to ensure accurate results.
        # config.delete_pass("fc_mkldnn_pass")
        # config.delete_pass("fc_act_mkldnn_fuse_pass")
        
        predictor = create_predictor(config)

        return predictor

    def eval(self): 
        """
        eval func
        """
        if os.path.exists(args.data_path):
            val_loader = eval_reader(
                args.data_path,
                batch_size=args.batch_size,
                crop_size=args.img_size,
                resize_size=args.resize_size,
                args=args)
        else:
            image = np.ones((args.batch_size, 3, args.img_size,
                             args.img_size)).astype(np.float32)
            label = [[None]] * args.batch_size
            val_loader = [[image, label]]
        results = []
        input_names = self.paddle_predictor.get_input_names()
        input_tensor = self.paddle_predictor.get_input_handle(input_names[0])
        output_names = self.paddle_predictor.get_output_names()
        output_tensor = self.paddle_predictor.get_output_handle(output_names[
            0])
        predict_time = 0.0
        time_min = float("inf")
        time_max = float("-inf")
        sample_nums = len(val_loader)
        for batch_id, (image, label) in enumerate(val_loader):
            image = np.array(image)

            input_tensor.copy_from_cpu(image)
            start_time = time.time()
            self.paddle_predictor.run()
            batch_output = output_tensor.copy_to_cpu()
            end_time = time.time()
            timed = end_time - start_time
            time_min = min(time_min, timed)
            time_max = max(time_max, timed)
            predict_time += timed
            if self.rerun_flag:
                return
            sort_array = batch_output.argsort(axis=1)
            top_1_pred = sort_array[:, -1:][:, ::-1]
            if label is None:
                results.append(top_1_pred)
                break
            label = np.array(label)
            top_1 = np.mean(label == top_1_pred)
            top_5_pred = sort_array[:, -5:][:, ::-1]
            acc_num = 0
            for i, _ in enumerate(label):
                if label[i][0] in top_5_pred[i]:
                    acc_num += 1
            top_5 = float(acc_num) / len(label)
            results.append([top_1, top_5])
            if batch_id % 100 == 0:
                print("Eval iter:", batch_id)
                sys.stdout.flush()
        result = np.mean(np.array(results), axis=0)
        fp_message = "FP16" if args.use_fp16 else "FP32"
        fp_message = "INT8" if args.use_int8 else fp_message
        print_msg = "Paddle"
        if args.use_trt:
            print_msg = "using TensorRT"
        elif args.use_mkldnn:
            print_msg = "using MKLDNN"
        time_avg = predict_time / sample_nums
        print(
            "[Benchmark]{}\t{}\tbatch size: {}.Inference time(ms): min={}, max={}, avg={}".
            format(
                print_msg,
                fp_message,
                args.batch_size,
                round(time_min * 1000, 2),
                round(time_max * 1000, 1),
                round(time_avg * 1000, 1), ))
        print("[Benchmark] Evaluation acc result: {}".format(result[0]))
        sys.stdout.flush()


if __name__ == "__main__": 
    parser = argsparser()
    args = parser.parse_args()
    predictor = Predictor()
    predictor.eval()
    if predictor.rerun_flag:
        print(
            "***** Collect dynamic shape done, Please rerun the program to get correct results. *****"
        )

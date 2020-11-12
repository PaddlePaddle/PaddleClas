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
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import argparse

from ppcls.modeling import architectures
from ppcls.utils.check import enable_static_mode
import paddle
import paddle.fluid as fluid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-p", "--pretrained_model", type=str)
    parser.add_argument("-o", "--output_path", type=str)
    parser.add_argument("--class_dim", type=int, default=1000)
    parser.add_argument("--img_size", type=int, default=224)

    return parser.parse_args()


def create_input(img_size=224):
    image = fluid.data(
        name='image', shape=[None, 3, img_size, img_size], dtype='float32')
    return image


def create_model(args, model, input, class_dim=1000):
    if args.model == "GoogLeNet":
        out, _, _ = model.net(input=input, class_dim=class_dim)
    else:
        out = model.net(input=input, class_dim=class_dim)
        out = fluid.layers.softmax(out)
    return out


def main():
    args = parse_args()

    model = architectures.__dict__[args.model]()

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    startup_prog = fluid.Program()
    infer_prog = fluid.Program()

    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            image = create_input(args.img_size)
            out = create_model(args, model, image, class_dim=args.class_dim)

    infer_prog = infer_prog.clone(for_test=True)
    fluid.load(
        program=infer_prog, model_path=args.pretrained_model, executor=exe)

    fluid.io.save_inference_model(
        dirname=args.output_path,
        feeded_var_names=[image.name],
        main_program=infer_prog,
        target_vars=out,
        executor=exe,
        model_filename='model',
        params_filename='params')


if __name__ == "__main__":
    enable_static_mode()
    main()

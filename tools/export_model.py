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

import argparse

from ppcls.modeling import architectures
from ppcls.utils.save_load import load_dygraph_pretrain
import paddle
import paddle.nn.functional as F
from paddle.jit import to_static


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-p", "--pretrained_model", type=str)
    parser.add_argument("-o", "--output_path", type=str)
    parser.add_argument("--class_dim", type=int, default=1000)
    # parser.add_argument("--img_size", type=int, default=224)

    return parser.parse_args()


class Net(paddle.nn.Layer):
    def __init__(self, net, to_static, class_dim):
        super(Net, self).__init__()
        self.pre_net = net(class_dim=class_dim)
        self.to_static = to_static

    # Please modify the 'shape' according to actual needs
    @to_static(input_spec=[
        paddle.static.InputSpec(
            shape=[None, 3, 224, 224], dtype='float32')
    ])
    def forward(self, inputs):
        x = self.pre_net(inputs)
        x = F.softmax(x)
        return x


def main():
    args = parse_args()

    paddle.disable_static()
    net = architectures.__dict__[args.model]

    model = Net(net, to_static, args.class_dim)

    # Please set 'load_static_weights' to 'True' or 'False' according to the 'pretrained_model'
    load_dygraph_pretrain(
        model, path=args.pretrained_model, load_static_weights=True)
    paddle.jit.save(model, args.output_path)


if __name__ == "__main__":
    main()

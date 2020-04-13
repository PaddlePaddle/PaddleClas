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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import paddle.fluid as fluid

from ppcls.modeling import get_architectures
from ppcls.modeling import similar_architectures
from ppcls.utils import logger


def check_version():
    """
    Log error and exit when the installed version of paddlepaddle is
    not satisfied.
    """
    err = "PaddlePaddle version 1.7 or higher is required, " \
          "or a suitable develop version is satisfied as well. \n" \
          "Please make sure the version is good with your code." \

    try:
        fluid.require_version('1.7.0')
    except Exception as e:
        logger.error(err)
        sys.exit(1)


def check_gpu():
    """
    Log error and exit when using paddlepaddle cpu version.
    """
    err = "You are using paddlepaddle cpu version! Please try to " \
          "install paddlepaddle-gpu to run model on GPU."

    try:
        assert fluid.is_compiled_with_cuda()
    except AssertionError:
        logger.error(err)
        sys.exit(1)


def check_architecture(architecture):
    """
    check architecture and recommend similar architectures
    """
    assert isinstance(architecture, str), \
            ("the type of architecture({}) should be str". format(architecture))
    similar_names = similar_architectures(architecture, get_architectures())
    model_list = ', '.join(similar_names)
    err = "{} is not exist! Maybe you want: [{}]" \
          "".format(architecture, model_list)

    try:
        assert architecture in similar_names
    except AssertionError:
        logger.error(err)
        sys.exit(1)


def check_mix(architecture, use_mix=False):
    """
    check mix parameter
    """
    err = "Cannot use mix processing in GoogLeNet, " \
          "please set use_mix = False."
    try:
        if architecture == "GoogLeNet": assert use_mix == False
    except AssertionError:
        logger.error(err)
        sys.exit(1)


def check_classes_num(classes_num):
    """
    check classes_num
    """
    err = "classes_num({}) should be a positive integer" \
           "and larger than 1".format(classes_num)
    try:
        assert isinstance(classes_num, int)
        assert classes_num > 1
    except AssertionError:
        logger.error(err)
        sys.exit(1)


def check_data_dir(path):
    """
    check cata_dir
    """
    err = "Data path is not exist, please given a right path" \
          "".format(path)
    try:
        assert os.isdir(path)
    except AssertionError:
        logger.error(err)
        sys.exit(1)


def check_function_params(config, key):
    """
    check specify config
    """
    k_config = config.get(key)
    assert k_config is not None, \
            ('{} is required in config'.format(key))

    assert k_config.get('function'), \
            ('function is required {} config'.format(key))
    params = k_config.get('params')
    assert params is not None, \
            ('params is required in {} config'.format(key))
    assert isinstance(params, dict), \
            ('the params in {} config should be a dict'.format(key))

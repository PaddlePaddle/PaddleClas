# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from io import open
from setuptools import setup

with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()


def readme():
    with open(
            'docs/en/inference_deployment/whl_deploy_en.md',
            encoding="utf-8-sig") as f:
        README = f.read()
    return README


setup(
    name='paddleclas',
    packages=['paddleclas'],
    package_dir={'paddleclas': ''},
    include_package_data=True,
    entry_points={
        "console_scripts": ["paddleclas= paddleclas.paddleclas:main"]
    },
    version='0.0.0',
    install_requires=requirements,
    license='Apache License 2.0',
    description='A treasure chest for visual recognition powered by PaddlePaddle.',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/PaddlePaddle/PaddleClas',
    download_url='https://github.com/PaddlePaddle/PaddleClas.git',
    keywords=[
        'image-classification', 'image-recognition', 'pretrained-models',
        'knowledge-distillation', 'product-recognition', 'autoaugment',
        'cutmix', 'randaugment', 'gridmask', 'deit', 'repvgg',
        'swin-transformer', 'image-retrieval-system'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7', 'Topic :: Utilities'
    ], )

#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import functools
import shutil
import sys

def main():
"""
Usage: when training with flag use_ema, and evaluating EMA model, should clean the saved model at first.
       To generate clean model:
    
       python ema_clean.py ema_model_dir cleaned_model_dir
"""
    cleaned_model_dir = sys.argv[1]
    ema_model_dir = sys.argv[2]
    if not os.path.exists(cleaned_model_dir):
        os.makedirs(cleaned_model_dir)

    items = os.listdir(ema_model_dir)
    for item in items:
        if item.find('ema') > -1:
            item_clean = item.replace('_ema_0', '')
            shutil.copyfile(os.path.join(ema_model_dir, item),
                            os.path.join(cleaned_model_dir, item_clean))
        elif item.find('mean') > -1 or item.find('variance') > -1:
            shutil.copyfile(os.path.join(ema_model_dir, item),
                            os.path.join(cleaned_model_dir, item))

if __name__ == '__main__':
    main()

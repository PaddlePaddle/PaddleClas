# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

def anti_fleet(log):
    """
    Because of the fucking Fleet, logs will print multi-times.
    So we only display one of them and ignore the others.    
    """
    def wrapper(fmt, *args):
        if int(os.getenv("PADDLE_TRAINER_ID", 0)) == 0:
            log(fmt, *args)
    return wrapper
 
@anti_fleet
def info(fmt, *args):
    _logger.info(fmt, *args)


@anti_fleet
def warning(fmt, *args):
    _logger.warning(fmt, *args)


@anti_fleet
def error(fmt, *args):
    _logger.error(fmt, *args)


def advertise():
    """
    Show the advertising message like the following:

    ===========================================================
    ==        PaddleClas is powered by PaddlePaddle !        ==
    ===========================================================
    ==                                                       ==
    ==   For more info please go to the following website.   ==
    ==                                                       ==
    ==       https://github.com/PaddlePaddle/PaddleClas      ==
    ===========================================================

    """
    copyright = "PaddleClas is powered by PaddlePaddle !"
    ad = "For more info please go to the following website."
    website = "https://github.com/PaddlePaddle/PaddleClas"
    AD_LEN = 6 + len(max([copyright, ad, website], key=len))

    info("\n{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n{6}\n{7}\n".format(
        "=" * (AD_LEN + 4),
        "=={}==".format(copyright.center(AD_LEN)),
        "=" * (AD_LEN + 4),
        "=={}==".format(' ' * AD_LEN),
        "=={}==".format(ad.center(AD_LEN)),
        "=={}==".format(' ' * AD_LEN),
        "=={}==".format(website.center(AD_LEN)),
        "=" * (AD_LEN + 4) ))

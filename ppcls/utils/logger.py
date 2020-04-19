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

import os
import logging
logging.basicConfig()
import random

DEBUG = logging.DEBUG  #10
INFO = logging.INFO  #20
WARN = logging.WARN  #30
ERROR = logging.ERROR  #40


class Logger(object):
    """
    Logger
    """

    def __init__(self, level=DEBUG):
        self.init(level)

    def init(self, level=DEBUG):
        """
        init
        """
        self._logger = logging.getLogger()
        self._logger.setLevel(level)

    def info(self, fmt, *args):
        """info"""
        self._logger.info(fmt, *args)

    def warning(self, fmt, *args):
        """warning"""
        self._logger.warning(fmt, *args)

    def error(self, fmt, *args):
        """error"""
        self._logger.error(fmt, *args)


_logger = Logger()


def init(level=DEBUG):
    """init for external"""
    _logger.init(level)


def info(fmt, *args):
    """info"""
    _logger.info(fmt, *args)


def warning(fmt, *args):
    """warn"""
    _logger.warning(fmt, *args)


def error(fmt, *args):
    """error"""
    _logger.error(fmt, *args)

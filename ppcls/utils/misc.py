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

__all__ = ['AverageMeter']


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self, name='', fmt='f', need_avg=False):
        self.name = name
        self.fmt = fmt
        self.need_avg = need_avg
        self.reset()

    def reset(self):
        """ reset """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ update """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def total(self):
        return '{self.name}_sum: {self.sum:{self.fmt}}'.format(self=self)

    @property
    def total_minute(self):
        return '{self.name}_sum: {s:{self.fmt}} min'.format(
            s=self.sum / 60, self=self)

    @property
    def mean(self):
        return '{self.name}_avg: {self.avg:{self.fmt}}'.format(
            self=self) if self.need_avg else ''

    @property
    def value(self):
        return '{self.name}: {self.val:{self.fmt}}'.format(self=self)

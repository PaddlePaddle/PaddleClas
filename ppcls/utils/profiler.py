# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import sys
import paddle

_is_profiler_finished = False
_profiler_step_id = 0


def _parse_profiler_options(profiler_options=None):
    if isinstance(profiler_options, str):
        try:
            profiler_options = eval(profiler_options)
        except NameError:
            # If profiler_option is a single string, we treat it as the profile_path.
            profiler_options = {'profile_path': profiler_options}

    assert isinstance(profiler_options, dict)

    default_options = {
        'batch_range': [10, 20],
        'state': 'All',
        'sorted_key': 'total',
        'tracer_option': 'Default',
        'profile_path': '/tmp/profile'
    }

    for key, value in default_options.items():
        if key not in profiler_options.keys():
            profiler_options[key] = value

    return profiler_options


def add_profiler_step(profiler_options=None):
    if profiler_options is None:
        return

    global _is_profiler_finished
    global _profiler_step_id

    if not _is_profiler_finished:
        profiler_options = _parse_profiler_options(profiler_options)

        if _profiler_step_id == profiler_options['batch_range'][0]:
            paddle.utils.profiler.start_profiler(
                profiler_options['state'], profiler_options['tracer_option'])
        elif _profiler_step_id == profiler_options['batch_range'][1]:
            paddle.utils.profiler.stop_profiler(
                profiler_options['sorted_key'],
                profiler_options['profile_path'])
            _is_profiler_finished = True

    _profiler_step_id += 1

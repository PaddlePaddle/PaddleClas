#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import functools
from typing import Any, Callable, Tuple

import paddle.distributed as dist


def get_dist_info() -> Tuple[int, int]:
    """get current rank and world_size.

    Returns:
        Tuple[int, int]: rank, world_size
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return rank, world_size


def main_only(func: Callable) -> Callable:
    """The decorated function will only run on the main process.

    Args:
        func (Callable): decorated function.

    Returns:
        Callable: decorator function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper

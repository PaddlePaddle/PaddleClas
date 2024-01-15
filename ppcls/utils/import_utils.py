# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
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

import importlib.util

def is_package_available(package_name: str) -> bool:
    """check if the package is avaliable
    Args:
        package_name (str): the installed package name
    Returns:
        bool: the existence of installed package
    """
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None and package_spec.has_location


def is_paddleclas_ops_available() -> bool:
    """check if `paddleclas_ops` ia avaliable
    Returns:
        bool: if `paddleclas_ops` is avaliable
    """
    return is_package_available("paddleclas_ops")
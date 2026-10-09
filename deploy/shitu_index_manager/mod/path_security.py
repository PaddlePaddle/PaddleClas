# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import os


class IndexPathPolicy:
    """Restrict filesystem access to a server-selected gallery directory."""

    def __init__(self, allowed_root):
        self.allowed_root = os.path.realpath(allowed_root)
        if not os.path.isdir(self.allowed_root):
            raise ValueError("The allowed index directory must exist")

    def within(self, root, path):
        root = os.path.realpath(root)
        try:
            resolved = os.path.realpath(path)
            if os.path.commonpath([root, resolved]) != root:
                raise ValueError("Path is outside the allowed index directory")
        except (TypeError, OSError):
            raise ValueError("Invalid index path") from None
        return resolved

    def gallery_root(self, root):
        if not root:
            root = self.allowed_root
        resolved = self.within(self.allowed_root, root)
        if not os.path.isdir(resolved):
            raise ValueError("The index directory must exist")
        return resolved

    def file_path(self, root, path):
        root = self.gallery_root(root)
        if not path:
            raise ValueError("An index file path is required")
        return self.within(root, os.path.join(root, path))

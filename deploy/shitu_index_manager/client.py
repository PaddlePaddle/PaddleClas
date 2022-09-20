# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import mod.mainwindow
"""
完整的index库如下:
root_path/            # 库存储目录
|-- image_list.txt     # 图像列表，每行：image_path label。由前端生成及修改。后端只读
|-- features.pkl       # 建库之后，保存的embedding向量，后端生成，前端无需操作
|-- images             # 图像存储目录，由前端生成及增删查等操作。后端只读
|   |-- md5.jpg
|   |-- md5.jpg
|   |-- ……
|-- index              # 真正的生成的index库存储目录，后端生成及操作，前端无需操作。
|   |-- vector.index   # faiss生成的索引库
|   |-- id_map.pkl     # 索引文件
"""


def FrontInterface(server_ip=None, server_port=None):
    front = QtWidgets.QApplication([])
    main_window = mod.mainwindow.MainWindow(ip=server_ip, port=server_port)
    main_window.showMaximized()
    sys.exit(front.exec_())


if __name__ == '__main__':
    server_ip = None
    server_port = None
    if len(sys.argv) == 2 and len(sys.argv[1].split(' ')) == 2:
        [server_ip, server_port] = sys.argv[1].split(' ')
    FrontInterface(server_ip, server_port)

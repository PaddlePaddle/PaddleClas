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
import subprocess
import psutil
import time
import secrets
import socket
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

if __name__ == "__main__":
    if not (len(sys.argv) == 3 or len(sys.argv) == 5):
        print("start example:")
        print("   python index_manager.py -c xxx.yaml")
        print("   python index_manager.py -c xxx.yaml -p port")
    yaml_path = sys.argv[2]
    if len(sys.argv) == 5:
        port = sys.argv[4]
    else:
        port = 8000
    assert (int(port) > 1024
            and int(port) < 65536), "The port should be bigger than 1024 and \
            smaller than 65536"

    try:
        ip = socket.gethostbyname(socket.gethostname())
    except:
        ip = "127.0.0.1"
    service_dir = os.path.dirname(os.path.abspath(__file__))
    server_cmd = [
        sys.executable,
        os.path.join(service_dir, "server.py"),
        "-c",
        yaml_path,
        "-o",
        "ip=" + ip,
        "-o",
        "port=" + str(port),
    ]
    process_env = os.environ.copy()
    if not process_env.get("PADDLECLAS_INDEX_TOKEN"):
        process_env["PADDLECLAS_INDEX_TOKEN"] = secrets.token_urlsafe(32)
    process_env.setdefault("PADDLECLAS_INDEX_ROOT", os.getcwd())
    server_proc = subprocess.Popen(server_cmd, env=process_env)
    client_proc = subprocess.Popen(
        [
            sys.executable,
            os.path.join(service_dir, "client.py"),
            "{} {}".format(ip, port),
        ],
        env=process_env,
    )
    try:
        while psutil.Process(client_proc.pid).status() == "running":
            time.sleep(0.5)
    except:
        pass

    client_proc.terminate()
    server_proc.terminate()

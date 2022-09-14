import json
import os
import urllib3
import urllib.parse


class IndexHttpClient():
    """索引库客户端，使用 urllib3 连接，使用 urllib.parse 进行 url 编码"""
    def __init__(self, host: str, port: int):
        self.__host = host
        self.__port = port
        self.__http = urllib3.PoolManager()
        self.__headers = {"Content-type": "application/json"}

    def url(self):
        return "http://{}:{}".format(self.__host, self.__port)

    def new_index(self,
                  image_list_path: str,
                  index_root_path: str,
                  index_method="HNSW32",
                  force=False):
        """新建 重建 库"""
        if index_method not in ["HNSW32", "FLAT", "IVF"]:
            raise Exception(
                "index_method 必须是 HNSW32, FLAT, IVF，实际值为：{}".format(
                    index_method))
        params = {"image_list_path":image_list_path, \
            "index_root_path":index_root_path, \
            "index_method":index_method, \
            "force":force}
        return self.__post(self.url() + "/new_index?", params)

    def open_index(self, index_root_path: str, image_list_path: str):
        """打开库"""
        params = {
            "index_root_path": index_root_path,
            "image_list_path": image_list_path
        }
        return self.__post(self.url() + "/open_index?", params)

    def update_index(self, image_list_path: str, index_root_path: str):
        """更新索引库"""
        params = {"image_list_path":image_list_path, \
            "index_root_path":index_root_path}
        return self.__post(self.url() + "/update_index?", params)

    def __post(self, url: str, params: dict):
        """发送 url 并接收数据"""
        http = self.__http
        encode_params = urllib.parse.urlencode(params)
        get_url = url + encode_params
        req = http.request("GET", get_url, headers=self.__headers)
        result = json.loads(req.data)
        if isinstance(result, str):
            result = eval(result)
        msg = result["error_message"]
        if msg != None and len(msg) == 0:
            msg = None
        return msg

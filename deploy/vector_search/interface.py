# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import ctypes
import paddle
import numpy.ctypeslib as ctl
import numpy as np
import os
import sys
import json
import platform

from ctypes import *
from numpy.ctypeslib import ndpointer

__dir__ = os.path.dirname(os.path.abspath(__file__))
winmode = None
if platform.system() == "Windows":
    lib_filename = "index.dll"
    if sys.version_info.minor >= 8:
        winmode = 0x8
else:
    lib_filename = "index.so"
so_path = os.path.join(__dir__, lib_filename)
try:
    if winmode is not None:
        lib = ctypes.CDLL(so_path, winmode=winmode)
    else:
        lib = ctypes.CDLL(so_path)
except Exception as ex:
    readme_path = os.path.join(__dir__, "README.md")
    print(
        f"Error happened when load lib {so_path} with msg {ex},\nplease refer to {readme_path} to rebuild your library."
    )
    exit(-1)


class IndexContext(Structure):
    _fields_ = [("graph", c_void_p), ("data", c_void_p)]


# for mobius IP index
build_mobius_index = lib.build_mobius_index
build_mobius_index.restype = None
build_mobius_index.argtypes = [
    ctl.ndpointer(
        np.float32, flags='aligned, c_contiguous'), ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_double, ctypes.c_char_p
]

search_mobius_index = lib.search_mobius_index
search_mobius_index.restype = None
search_mobius_index.argtypes = [
    ctl.ndpointer(
        np.float32, flags='aligned, c_contiguous'), ctypes.c_int, ctypes.c_int,
    ctypes.c_int, POINTER(IndexContext), ctl.ndpointer(
        np.uint64, flags='aligned, c_contiguous'), ctl.ndpointer(
            np.float64, flags='aligned, c_contiguous')
]

load_mobius_index_prefix = lib.load_mobius_index_prefix
load_mobius_index_prefix.restype = None
load_mobius_index_prefix.argtypes = [
    ctypes.c_int, ctypes.c_int, POINTER(IndexContext), ctypes.c_char_p
]

save_mobius_index_prefix = lib.save_mobius_index_prefix
save_mobius_index_prefix.restype = None
save_mobius_index_prefix.argtypes = [POINTER(IndexContext), ctypes.c_char_p]

# for L2 index
build_l2_index = lib.build_l2_index
build_l2_index.restype = None
build_l2_index.argtypes = [
    ctl.ndpointer(
        np.float32, flags='aligned, c_contiguous'), ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_char_p
]

search_l2_index = lib.search_l2_index
search_l2_index.restype = None
search_l2_index.argtypes = [
    ctl.ndpointer(
        np.float32, flags='aligned, c_contiguous'), ctypes.c_int, ctypes.c_int,
    ctypes.c_int, POINTER(IndexContext), ctl.ndpointer(
        np.uint64, flags='aligned, c_contiguous'), ctl.ndpointer(
            np.float64, flags='aligned, c_contiguous')
]

load_l2_index_prefix = lib.load_l2_index_prefix
load_l2_index_prefix.restype = None
load_l2_index_prefix.argtypes = [
    ctypes.c_int, ctypes.c_int, POINTER(IndexContext), ctypes.c_char_p
]

save_l2_index_prefix = lib.save_l2_index_prefix
save_l2_index_prefix.restype = None
save_l2_index_prefix.argtypes = [POINTER(IndexContext), ctypes.c_char_p]

release_context = lib.release_context
release_context.restype = None
release_context.argtypes = [POINTER(IndexContext)]


class Graph_Index(object):
    """
        graph index
    """

    def __init__(self, dist_type="IP"):
        self.dim = 0
        self.total_num = 0
        self.dist_type = dist_type
        self.mobius_pow = 2.0
        self.index_context = IndexContext(0, 0)
        self.gallery_doc_dict = {}
        self.with_attr = False
        assert dist_type in ["IP", "L2"], "Only support IP and L2 distance ..."

    def build(self,
              gallery_vectors,
              gallery_docs=[],
              pq_size=100,
              index_path='graph_index/',
              append_index=False):
        """
        build index 
        """
        if paddle.is_tensor(gallery_vectors):
            gallery_vectors = gallery_vectors.numpy()
        assert gallery_vectors.ndim == 2, "Input vector must be 2D ..."

        self.total_num = gallery_vectors.shape[0]
        self.dim = gallery_vectors.shape[1]

        assert (len(gallery_docs) == self.total_num
                if len(gallery_docs) > 0 else True)

        print("training index -> num: {}, dim: {}, dist_type: {}".format(
            self.total_num, self.dim, self.dist_type))

        if not os.path.exists(index_path):
            os.makedirs(index_path)

        if self.dist_type == "IP":
            build_mobius_index(
                gallery_vectors, self.total_num, self.dim, pq_size,
                self.mobius_pow,
                create_string_buffer((index_path + "/index").encode('utf-8')))
            load_mobius_index_prefix(
                self.total_num, self.dim,
                ctypes.byref(self.index_context),
                create_string_buffer((index_path + "/index").encode('utf-8')))
        else:
            build_l2_index(
                gallery_vectors, self.total_num, self.dim, pq_size,
                create_string_buffer((index_path + "/index").encode('utf-8')))
            load_l2_index_prefix(
                self.total_num, self.dim,
                ctypes.byref(self.index_context),
                create_string_buffer((index_path + "/index").encode('utf-8')))

        self.gallery_doc_dict = {}
        if len(gallery_docs) > 0:
            self.with_attr = True
            for i in range(gallery_vectors.shape[0]):
                self.gallery_doc_dict[str(i)] = gallery_docs[i]

        self.gallery_doc_dict["total_num"] = self.total_num
        self.gallery_doc_dict["dim"] = self.dim
        self.gallery_doc_dict["dist_type"] = self.dist_type
        self.gallery_doc_dict["with_attr"] = self.with_attr

        output_path = os.path.join(index_path, "info.json")
        if append_index is True and os.path.exists(output_path):
            with open(output_path, "r") as fin:
                lines = fin.readlines()[0]
                ori_gallery_doc_dict = json.loads(lines)
            assert ori_gallery_doc_dict["dist_type"] == self.gallery_doc_dict[
                "dist_type"]
            assert ori_gallery_doc_dict["dim"] == self.gallery_doc_dict["dim"]
            assert ori_gallery_doc_dict["with_attr"] == self.gallery_doc_dict[
                "with_attr"]
            offset = ori_gallery_doc_dict["total_num"]
            for i in range(0, self.gallery_doc_dict["total_num"]):
                ori_gallery_doc_dict[str(i + offset)] = self.gallery_doc_dict[
                    str(i)]

            ori_gallery_doc_dict["total_num"] += self.gallery_doc_dict[
                "total_num"]
            self.gallery_doc_dict = ori_gallery_doc_dict
        with open(output_path, "w") as f:
            json.dump(self.gallery_doc_dict, f)

        print("finished creating index ...")

    def search(self, query, return_k=10, search_budget=100):
        """
        search
        """
        ret_id = np.zeros(return_k, dtype=np.uint64)
        ret_score = np.zeros(return_k, dtype=np.float64)

        if paddle.is_tensor(query):
            query = query.numpy()
        if self.dist_type == "IP":
            search_mobius_index(query, self.dim, search_budget, return_k,
                                ctypes.byref(self.index_context), ret_id,
                                ret_score)
        else:
            search_l2_index(query, self.dim, search_budget, return_k,
                            ctypes.byref(self.index_context), ret_id,
                            ret_score)

        ret_id = ret_id.tolist()
        ret_doc = []
        if self.with_attr:
            for i in range(return_k):
                ret_doc.append(self.gallery_doc_dict[str(ret_id[i])])
            return ret_score, ret_doc
        else:
            return ret_score, ret_id

    def dump(self, index_path):

        if not os.path.exists(index_path):
            os.makedirs(index_path)

        if self.dist_type == "IP":
            save_mobius_index_prefix(
                ctypes.byref(self.index_context),
                create_string_buffer((index_path + "/index").encode('utf-8')))
        else:
            save_l2_index_prefix(
                ctypes.byref(self.index_context),
                create_string_buffer((index_path + "/index").encode('utf-8')))

        with open(index_path + "/info.json", "w") as f:
            json.dump(self.gallery_doc_dict, f)

    def load(self, index_path):
        self.gallery_doc_dict = {}

        with open(index_path + "/info.json", "r") as f:
            self.gallery_doc_dict = json.load(f)

        self.total_num = self.gallery_doc_dict["total_num"]
        self.dim = self.gallery_doc_dict["dim"]
        self.dist_type = self.gallery_doc_dict["dist_type"]
        self.with_attr = self.gallery_doc_dict["with_attr"]

        if self.dist_type == "IP":
            load_mobius_index_prefix(
                self.total_num, self.dim,
                ctypes.byref(self.index_context),
                create_string_buffer((index_path + "/index").encode('utf-8')))
        else:
            load_l2_index_prefix(
                self.total_num, self.dim,
                ctypes.byref(self.index_context),
                create_string_buffer((index_path + "/index").encode('utf-8')))

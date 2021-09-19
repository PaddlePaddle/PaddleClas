# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle_serving_app.reader import Sequential, URL2Image, Resize, CenterCrop, RGB2BGR, Transpose, Div, Normalize, Base64ToImage
try:
    from paddle_serving_server_gpu.web_service import WebService, Op
except ImportError:
    from paddle_serving_server.web_service import WebService, Op
import logging
import numpy as np
import base64, cv2
import os
import faiss
import pickle

class RecogOp(Op):
    def init_op(self):
        self.seq = Sequential([
            Resize(256), CenterCrop(224), RGB2BGR(), Transpose((2, 0, 1)),
            Div(255), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                                True)
        ])

        #load index; and return top1
        index_dir = "../../recognition_demo_data_v1.1/gallery_product/index"
        assert os.path.exists(os.path.join(
            index_dir, "vector.index")), "vector.index not found ..."
        assert os.path.exists(os.path.join(
            index_dir, "id_map.pkl")), "id_map.pkl not found ... "
        
        self.Searcher = faiss.read_index(
            os.path.join(index_dir, "vector.index"))
                
        with open(os.path.join(index_dir, "id_map.pkl"), "rb") as fd:
            self.id_map = pickle.load(fd)
        print("init done!!!!!!")

    def preprocess(self, input_dicts, data_id, log_id):
        print("1111111111")
        (_, input_dict), = input_dicts.items()
        batch_size = len(input_dict.keys())
        imgs = []
        print("222222222")
        for key in input_dict.keys():
            data = base64.b64decode(input_dict[key].encode('utf8'))
            data = np.fromstring(data, np.uint8)
            im = cv2.imdecode(data, cv2.IMREAD_COLOR)
            img = self.seq(im)
            imgs.append(img[np.newaxis, :].copy())
        input_imgs = np.concatenate(imgs, axis=0)
        return {"x": input_imgs}, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, log_id):
        """
        get feature and do retrieval
        """
        print("333333333")
        score_list = fetch_dict["features"]
        print(score_list.shape)   #it is an array, print the shape (1, 512)
        
        #search
        scores, docs = self.Searcher.search(score_list,  1)
        print(scores.shape)  # 1 * 1
        print(docs.shape)    # 1 * 1
            
        # just top-1 result will be returned for the final
        result = {}
        result["label"] = self.id_map[docs[0][0]].split()[1]
        
        #add result
        return result, None, ""

        
class ProductRecognitionService(WebService):
    def get_pipeline_response(self, read_op):
        image_op = RecogOp(name="recog", input_ops=[read_op])
        return image_op

uci_service = ProductRecognitionService(name="productrecog")
uci_service.prepare_pipeline_config("config.yml")
uci_service.run_service()

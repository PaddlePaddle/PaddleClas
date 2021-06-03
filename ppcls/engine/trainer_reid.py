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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

import numpy as np
import paddle
from .trainer import Trainer
from ppcls.utils import logger
from ppcls.data import build_dataloader


class TrainerReID(Trainer):
    def __init__(self, config, mode="train"):
        super().__init__(config, mode)

        self.gallery_dataloader = build_dataloader(self.config["DataLoader"],
                                                   "Gallery", self.device)

        self.query_dataloader = build_dataloader(self.config["DataLoader"],
                                                 "Query", self.device)

    @paddle.no_grad()
    def eval(self, epoch_id=0):
        output_info = dict()
        self.model.eval()
        print_batch_step = self.config["Global"]["print_batch_step"]

        # step1. build gallery
        gallery_feas, gallery_img_id, gallery_camera_id = self._cal_feature(
            name='gallery')
        query_feas, query_img_id, query_camera_id = self._cal_feature(
            name='query')

        # step2. do evaluation
        if "num_split" in self.config["Global"]:
            num_split = self.config["Global"]["num_split"]
        else:
            num_split = 1
        fea_blocks = paddle.split(query_feas, num_or_sections=1)

        total_similarities_matrix = None

        for block_fea in fea_blocks:
            similarities_matrix = paddle.matmul(
                block_fea, gallery_feas, transpose_y=True)
            if total_similarities_matrix is None:
                total_similarities_matrix = similarities_matrix
            else:
                total_similarities_matrix = paddle.concat(
                    [total_similarities_matrix, similarities_matrix])

        #  distmat = (1 - total_similarities_matrix).numpy()
        q_pids = query_img_id.numpy().reshape((query_img_id.shape[0]))
        g_pids = gallery_img_id.numpy().reshape((gallery_img_id.shape[0]))
        if query_camera_id is not None and gallery_camera_id is not None:
            q_camids = query_camera_id.numpy().reshape(
                (query_camera_id.shape[0]))
            g_camids = gallery_camera_id.numpy().reshape(
                (gallery_camera_id.shape[0]))
        max_rank = 50

        num_q, num_g = total_similarities_matrix.shape
        if num_g < max_rank:
            max_rank = num_g
            print('Note: number of gallery samples is quite small, got {}'.
                  format(num_g))

        #  indices = np.argsort(distmat, axis=1)
        indices = paddle.argsort(
            total_similarities_matrix, axis=1, descending=True).numpy()

        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        all_INP = []
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            if query_camera_id is not None and gallery_camera_id is not None:
                remove = (g_pids[order] == q_pid) & (
                    g_camids[order] == q_camid)
            else:
                remove = g_pids[order] == q_pid
            keep = np.invert(remove)

            # compute cmc curve
            raw_cmc = matches[q_idx][
                keep]  # binary vector, positions with value 1 are correct matches
            if not np.any(raw_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = raw_cmc.cumsum()

            pos_idx = np.where(raw_cmc == 1)
            max_pos_idx = np.max(pos_idx)
            inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
            all_INP.append(inp)

            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = raw_cmc.sum()
            tmp_cmc = raw_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)
        assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        logger.info(
            "[Eval][Epoch {}]: mAP: {:.5f}, mINP: {:.5f},rank_1: {:.5f}, rank_5: {:.5f}"
            .format(epoch_id, mAP, mINP, all_cmc[0], all_cmc[4]))
        return mAP

    def _cal_feature(self, name='gallery'):
        all_feas = None
        all_image_id = None
        all_camera_id = None
        if name == 'gallery':
            dataloader = self.gallery_dataloader
        elif name == 'query':
            dataloader = self.query_dataloader
        else:
            raise RuntimeError("Only support gallery or query dataset")

        has_cam_id = False
        for idx, batch in enumerate(dataloader(
        )):  # load is very time-consuming
            batch = [paddle.to_tensor(x) for x in batch]
            batch[1] = batch[1].reshape([-1, 1])
            if len(batch) == 3:
                has_cam_id = True
                batch[2] = batch[2].reshape([-1, 1])
            out = self.model(batch[0], batch[1])
            batch_feas = out["features"]

            # do norm
            feas_norm = paddle.sqrt(
                paddle.sum(paddle.square(batch_feas), axis=1, keepdim=True))
            batch_feas = paddle.divide(batch_feas, feas_norm)

            batch_feas = batch_feas
            batch_image_labels = batch[1]
            if has_cam_id:
                batch_camera_labels = batch[2]

            if all_feas is None:
                all_feas = batch_feas
                if has_cam_id:
                    all_camera_id = batch[2]
                all_image_id = batch[1]
            else:
                all_feas = paddle.concat([all_feas, batch_feas])
                all_image_id = paddle.concat([all_image_id, batch[1]])
                if has_cam_id:
                    all_camera_id = paddle.concat([all_camera_id, batch[2]])

        if paddle.distributed.get_world_size() > 1:
            feat_list = []
            img_id_list = []
            cam_id_list = []
            paddle.distributed.all_gather(feat_list, all_feas)
            paddle.distributed.all_gather(img_id_list, all_image_id)
            all_feas = paddle.concat(feat_list, axis=0)
            all_image_id = paddle.concat(img_id_list, axis=0)
            if has_cam_id:
                paddle.distributed.all_gather(cam_id_list, all_camera_id)
                all_camera_id = paddle.concat(cam_id_list, axis=0)

        logger.info("Build {} done, all feat shape: {}, begin to eval..".
                    format(name, all_feas.shape))
        return all_feas, all_image_id, all_camera_id

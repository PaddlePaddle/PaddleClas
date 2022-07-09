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

import platform
from typing import Optional

import numpy as np
import paddle
from ppcls.utils import logger


def retrieval_eval(engine, epoch_id=0):
    engine.model.eval()
    # step1. build gallery
    if engine.gallery_query_dataloader is not None:
        gallery_feas, gallery_img_id, gallery_unique_id = cal_feature(
            engine, name='gallery_query')
        query_feas, query_img_id, query_query_id = gallery_feas, gallery_img_id, gallery_unique_id
    else:
        gallery_feas, gallery_img_id, gallery_unique_id = cal_feature(
            engine, name='gallery')
        query_feas, query_img_id, query_query_id = cal_feature(
            engine, name='query')

    # step2. do evaluation
    sim_block_size = engine.config["Global"].get("sim_block_size", 64)
    sections = [sim_block_size] * (len(query_feas) // sim_block_size)
    if len(query_feas) % sim_block_size:
        sections.append(len(query_feas) % sim_block_size)
    fea_blocks = paddle.split(query_feas, num_or_sections=sections)
    if query_query_id is not None:
        query_id_blocks = paddle.split(
            query_query_id, num_or_sections=sections)
    image_id_blocks = paddle.split(query_img_id, num_or_sections=sections)
    metric_key = None

    if engine.eval_loss_func is None:
        metric_dict = {metric_key: 0.}
    else:
        reranking_flag = engine.config['Global'].get('re_ranking', False)
        logger.info(f"re_ranking={reranking_flag}")
        metric_dict = dict()
        if reranking_flag:
            # set the order from small to large
            for i in range(len(engine.eval_metric_func.metric_func_list)):
                if hasattr(engine.eval_metric_func.metric_func_list[i], 'descending') \
                        and engine.eval_metric_func.metric_func_list[i].descending is True:
                    engine.eval_metric_func.metric_func_list[
                        i].descending = False
                    logger.warning(
                        f"re_ranking=True,{engine.eval_metric_func.metric_func_list[i].__class__.__name__}.descending has been set to False"
                    )

            # compute distance matrix(The smaller the value, the more similar)
            distmat = re_ranking(
                query_feas, gallery_feas, k1=20, k2=6, lambda_value=0.3)

            # compute keep mask
            query_id_mask = (query_query_id != gallery_unique_id.t())
            image_id_mask = (query_img_id != gallery_img_id.t())
            keep_mask = paddle.logical_or(query_id_mask, image_id_mask)

            # set inf(1e9) distance to those exist in gallery
            distmat = distmat * keep_mask.astype("float32")
            inf_mat = (paddle.logical_not(keep_mask).astype("float32")) * 1e20
            distmat = distmat + inf_mat

            # compute metric
            metric_tmp = engine.eval_metric_func(distmat, query_img_id,
                                                 gallery_img_id, keep_mask)
            for key in metric_tmp:
                metric_dict[key] = metric_tmp[key]
        else:
            for block_idx, block_fea in enumerate(fea_blocks):
                similarity_matrix = paddle.matmul(
                    block_fea, gallery_feas, transpose_y=True)  # [n,m]
                if query_query_id is not None:
                    query_id_block = query_id_blocks[block_idx]
                    query_id_mask = (query_id_block != gallery_unique_id.t())

                    image_id_block = image_id_blocks[block_idx]
                    image_id_mask = (image_id_block != gallery_img_id.t())

                    keep_mask = paddle.logical_or(query_id_mask, image_id_mask)
                    similarity_matrix = similarity_matrix * keep_mask.astype(
                        "float32")
                else:
                    keep_mask = None

                metric_tmp = engine.eval_metric_func(
                    similarity_matrix, image_id_blocks[block_idx],
                    gallery_img_id, keep_mask)

                for key in metric_tmp:
                    if key not in metric_dict:
                        metric_dict[key] = metric_tmp[key] * block_fea.shape[
                            0] / len(query_feas)
                    else:
                        metric_dict[key] += metric_tmp[key] * block_fea.shape[
                            0] / len(query_feas)

    metric_info_list = []
    for key in metric_dict:
        if metric_key is None:
            metric_key = key
        metric_info_list.append("{}: {:.5f}".format(key, metric_dict[key]))
    metric_msg = ", ".join(metric_info_list)
    logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

    return metric_dict[metric_key]


def cal_feature(engine, name='gallery'):
    has_unique_id = False
    all_unique_id = None

    if name == 'gallery':
        dataloader = engine.gallery_dataloader
    elif name == 'query':
        dataloader = engine.query_dataloader
    elif name == 'gallery_query':
        dataloader = engine.gallery_query_dataloader
    else:
        raise RuntimeError("Only support gallery or query dataset")

    batch_feas_list = []
    img_id_list = []
    unique_id_list = []
    max_iter = len(dataloader) - 1 if platform.system() == "Windows" else len(
        dataloader)
    for idx, batch in enumerate(dataloader):  # load is very time-consuming
        if idx >= max_iter:
            break
        if idx % engine.config["Global"]["print_batch_step"] == 0:
            logger.info(
                f"{name} feature calculation process: [{idx}/{len(dataloader)}]"
            )
        if engine.use_dali:
            batch = [
                paddle.to_tensor(batch[0]['data']),
                paddle.to_tensor(batch[0]['label'])
            ]
        batch = [paddle.to_tensor(x) for x in batch]
        batch[1] = batch[1].reshape([-1, 1]).astype("int64")
        if len(batch) == 3:
            has_unique_id = True
            batch[2] = batch[2].reshape([-1, 1]).astype("int64")
        if engine.amp and engine.amp_eval:
            with paddle.amp.auto_cast(
                    custom_black_list={
                        "flatten_contiguous_range", "greater_than"
                    },
                    level=engine.amp_level):
                out = engine.model(batch[0], batch[1])
        else:
            out = engine.model(batch[0], batch[1])
        if "Student" in out:
            out = out["Student"]

        # get features
        if engine.config["Global"].get("retrieval_feature_from",
                                       "features") == "features":
            # use neck's output as features
            batch_feas = out["features"]
        else:
            # use backbone's output as features
            batch_feas = out["backbone"]

        # do norm
        if engine.config["Global"].get("feature_normalize", True):
            feas_norm = paddle.sqrt(
                paddle.sum(paddle.square(batch_feas), axis=1, keepdim=True))
            batch_feas = paddle.divide(batch_feas, feas_norm)

        # do binarize
        if engine.config["Global"].get("feature_binarize") == "round":
            batch_feas = paddle.round(batch_feas).astype("float32") * 2.0 - 1.0

        if engine.config["Global"].get("feature_binarize") == "sign":
            batch_feas = paddle.sign(batch_feas).astype("float32")

        if paddle.distributed.get_world_size() > 1:
            batch_feas_gather = []
            img_id_gather = []
            unique_id_gather = []
            paddle.distributed.all_gather(batch_feas_gather, batch_feas)
            paddle.distributed.all_gather(img_id_gather, batch[1])
            batch_feas_list.append(paddle.concat(batch_feas_gather))
            img_id_list.append(paddle.concat(img_id_gather))
            if has_unique_id:
                paddle.distributed.all_gather(unique_id_gather, batch[2])
                unique_id_list.append(paddle.concat(unique_id_gather))
        else:
            batch_feas_list.append(batch_feas)
            img_id_list.append(batch[1])
            if has_unique_id:
                unique_id_list.append(batch[2])

    if engine.use_dali:
        dataloader.reset()

    all_feas = paddle.concat(batch_feas_list)
    all_img_id = paddle.concat(img_id_list)
    if has_unique_id:
        all_unique_id = paddle.concat(unique_id_list)

    # just for DistributedBatchSampler issue: repeat sampling
    total_samples = len(
        dataloader.dataset) if not engine.use_dali else dataloader.size
    all_feas = all_feas[:total_samples]
    all_img_id = all_img_id[:total_samples]
    if has_unique_id:
        all_unique_id = all_unique_id[:total_samples]

    logger.info("Build {} done, all feat shape: {}, begin to eval..".format(
        name, all_feas.shape))
    return all_feas, all_img_id, all_unique_id


def re_ranking(query_feas: paddle.Tensor,
               gallery_feas: paddle.Tensor,
               k1: int=20,
               k2: int=6,
               lambda_value: int=0.5,
               local_distmat: Optional[np.ndarray]=None,
               only_local: bool=False) -> paddle.Tensor:
    """re-ranking, most computed with numpy

    code heavily based on
    https://github.com/michuanhaohao/reid-strong-baseline/blob/3da7e6f03164a92e696cb6da059b1cd771b0346d/utils/reid_metric.py

    Args:
        query_feas (paddle.Tensor): query features, [num_query, num_features]
        gallery_feas (paddle.Tensor): gallery features, [num_gallery, num_features]
        k1 (int, optional): k1. Defaults to 20.
        k2 (int, optional): k2. Defaults to 6.
        lambda_value (int, optional): lambda. Defaults to 0.5.
        local_distmat (Optional[np.ndarray], optional): local_distmat. Defaults to None.
        only_local (bool, optional): only_local. Defaults to False.

    Returns:
        paddle.Tensor: final_dist matrix after re-ranking, [num_query, num_gallery]
    """
    query_num = query_feas.shape[0]
    all_num = query_num + gallery_feas.shape[0]
    if only_local:
        original_dist = local_distmat
    else:
        feat = paddle.concat([query_feas, gallery_feas])
        logger.info('using GPU to compute original distance')

        # L2 distance
        distmat = paddle.pow(feat, 2).sum(axis=1, keepdim=True).expand([all_num, all_num]) + \
            paddle.pow(feat, 2).sum(axis=1, keepdim=True).expand([all_num, all_num]).t()
        distmat = distmat.addmm(x=feat, y=feat.t(), alpha=-2.0, beta=1.0)

        original_dist = distmat.cpu().numpy()
        del feat
        if local_distmat is not None:
            original_dist = original_dist + local_distmat

    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)
    logger.info('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(
                np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(
                candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[
                fi_candidate]
            if len(
                    np.intersect1d(candidate_k_reciprocal_index,
                                   k_reciprocal_index)) > 2 / 3 * len(
                                       candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value
                                 ) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    final_dist = paddle.to_tensor(final_dist)
    return final_dist

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

from collections import defaultdict

import numpy as np
import paddle
import scipy

from ppcls.utils import all_gather, logger


def retrieval_eval(engine, epoch_id=0):
    engine.model.eval()
    # step1. prepare query and gallery features
    if engine.gallery_query_dataloader is not None:
        gallery_feat, gallery_label, gallery_camera = compute_feature(
            engine, "gallery_query")
        query_feat, query_label, query_camera = gallery_feat, gallery_label, gallery_camera
    else:
        gallery_feat, gallery_label, gallery_camera = compute_feature(
            engine, "gallery")
        query_feat, query_label, query_camera = compute_feature(engine,
                                                                "query")

    # step2. split features into feature blocks for saving memory
    num_query = len(query_feat)
    block_size = engine.config["Global"].get("sim_block_size", 64)
    sections = [block_size] * (num_query // block_size)
    if num_query % block_size > 0:
        sections.append(num_query % block_size)

    query_feat_blocks = paddle.split(query_feat, sections)
    query_label_blocks = paddle.split(query_label, sections)
    query_camera_blocks = paddle.split(
        query_camera, sections) if query_camera is not None else None
    metric_key = None

    # step3. compute metric
    if engine.eval_loss_func is None:
        metric_dict = {metric_key: 0.0}
    else:
        use_reranking = engine.config["Global"].get("re_ranking", False)
        logger.info(f"re_ranking={use_reranking}")
        if use_reranking:
            # compute distance matrix
            distmat = compute_re_ranking_dist(
                query_feat, gallery_feat, engine.config["Global"].get(
                    "feature_normalize", True), 20, 6, 0.3)
            # exclude illegal distance
            if query_camera is not None:
                camera_mask = query_camera != gallery_camera.t()
                label_mask = query_label != gallery_label.t()
                keep_mask = label_mask | camera_mask
                distmat = keep_mask.astype(query_feat.dtype) * distmat + (
                    ~keep_mask).astype(query_feat.dtype) * (distmat.max() + 1)
            else:
                keep_mask = None
            # compute metric with all samples
            metric_dict = engine.eval_metric_func(-distmat, query_label,
                                                  gallery_label, keep_mask)
        else:
            metric_dict = defaultdict(float)
            for block_idx, block_feat in enumerate(query_feat_blocks):
                # compute distance matrix
                distmat = paddle.matmul(
                    block_feat, gallery_feat, transpose_y=True)
                # exclude illegal distance
                if query_camera is not None:
                    camera_mask = query_camera_blocks[
                        block_idx] != gallery_camera.t()
                    label_mask = query_label_blocks[
                        block_idx] != gallery_label.t()
                    keep_mask = label_mask | camera_mask
                    distmat = keep_mask.astype(query_feat.dtype) * distmat
                else:
                    keep_mask = None
                # compute metric by block
                metric_block = engine.eval_metric_func(
                    distmat, query_label_blocks[block_idx], gallery_label,
                    keep_mask)
                # accumulate metric
                for key in metric_block:
                    metric_dict[key] += metric_block[key] * block_feat.shape[
                        0] / num_query

    metric_info_list = []
    for key, value in metric_dict.items():
        metric_info_list.append(f"{key}: {value:.5f}")
        if metric_key is None:
            metric_key = key
    metric_msg = ", ".join(metric_info_list)
    logger.info(f"[Eval][Epoch {epoch_id}][Avg]{metric_msg}")

    return metric_dict[metric_key]


def compute_feature(engine, name="gallery"):
    if name == "gallery":
        dataloader = engine.gallery_dataloader
    elif name == "query":
        dataloader = engine.query_dataloader
    elif name == "gallery_query":
        dataloader = engine.gallery_query_dataloader
    else:
        raise ValueError(
            f"Only support gallery or query or gallery_query dataset, but got {name}"
        )

    all_feat = []
    all_label = []
    all_camera = []
    has_camera = False
    for idx, batch in enumerate(dataloader):  # load is very time-consuming
        if idx % engine.config["Global"]["print_batch_step"] == 0:
            logger.info(
                f"{name} feature calculation process: [{idx}/{len(dataloader)}]"
            )

        batch = [paddle.to_tensor(x) for x in batch]
        batch[1] = batch[1].reshape([-1, 1]).astype("int64")
        if len(batch) >= 3:
            has_camera = True
            batch[2] = batch[2].reshape([-1, 1]).astype("int64")
        with engine.auto_cast(is_eval=True):
            out = engine.model(batch[0])
        if "Student" in out:
            out = out["Student"]

        # get features
        if engine.config["Global"].get("retrieval_feature_from",
                                       "features") == "features":
            # use output from neck as feature
            batch_feat = out["features"]
        else:
            # use output from backbone as feature
            batch_feat = out["backbone"]

        # do norm(optional)
        if engine.config["Global"].get("feature_normalize", True):
            batch_feat = paddle.nn.functional.normalize(batch_feat, p=2)

        # do binarize(optional)
        if engine.config["Global"].get("feature_binarize") == "round":
            batch_feat = paddle.round(batch_feat).astype("float32") * 2.0 - 1.0
        elif engine.config["Global"].get("feature_binarize") == "sign":
            batch_feat = paddle.sign(batch_feat).astype("float32")

        if paddle.distributed.get_world_size() > 1:
            all_feat.append(all_gather(batch_feat))
            all_label.append(all_gather(batch[1]))
            if has_camera:
                all_camera.append(all_gather(batch[2]))
        else:
            all_feat.append(batch_feat)
            all_label.append(batch[1])
            if has_camera:
                all_camera.append(batch[2])

    if engine.use_dali:
        dataloader.reset()

    all_feat = paddle.concat(all_feat)
    all_label = paddle.concat(all_label)
    if has_camera:
        all_camera = paddle.concat(all_camera)
    else:
        all_camera = None
    # discard redundant padding sample(s) at the end
    total_samples = dataloader.size if engine.use_dali else len(
        dataloader.dataset)
    all_feat = all_feat[:total_samples]
    all_label = all_label[:total_samples]
    if has_camera:
        all_camera = all_camera[:total_samples]

    logger.info(f"Build {name} done, all feat shape: {all_feat.shape}")
    return all_feat, all_label, all_camera


def k_reciprocal_neighbor(rank: np.ndarray, p: int, k: int) -> np.ndarray:
    """Implementation of k-reciprocal nearest neighbors, i.e. R(p, k)

    Args:
        rank (np.ndarray): Rank mat with shape of [N, N].
        p (int): Probe index.
        k (int): Parameter k for k-reciprocal nearest neighbors algorithm.

    Returns:
        np.ndarray: K-reciprocal nearest neighbors of probe p with shape of [M, ].
    """
    # use k+1 for excluding probe index itself
    forward_k_neigh_index = rank[p, :k + 1]
    backward_k_neigh_index = rank[forward_k_neigh_index, :k + 1]
    candidate = np.where(backward_k_neigh_index == p)[0]
    return forward_k_neigh_index[candidate]


def compute_re_ranking_dist(query_feat: paddle.Tensor,
                            gallery_feat: paddle.Tensor,
                            feature_normed: bool=True,
                            k1: int=20,
                            k2: int=6,
                            lamb: float=0.5) -> paddle.Tensor:
    """
    Re-ranking Person Re-identification with k-reciprocal Encoding
    Reference: https://arxiv.org/abs/1701.08398
    Code refernence: https://github.com/michuanhaohao/reid-strong-baseline/blob/master/utils/re_ranking.py

    Args:
        query_feat (paddle.Tensor): Query features with shape of [num_query, feature_dim].
        gallery_feat (paddle.Tensor):  Gallery features with shape of [num_gallery, feature_dim].
        feature_normed (bool, optional):  Whether input features are normalized.
        k1 (int, optional): Parameter for K-reciprocal nearest neighbors. Defaults to 20.
        k2 (int, optional): Parameter for K-nearest neighbors. Defaults to 6.
        lamb (float, optional): Penalty factor. Defaults to 0.5.

    Returns:
        paddle.Tensor: (1 - lamb) x Dj + lamb x D, with shape of [num_query, num_gallery].
    """
    num_query = query_feat.shape[0]
    num_gallery = gallery_feat.shape[0]
    num_all = num_query + num_gallery
    feat = paddle.concat([query_feat, gallery_feat], 0)
    logger.info("Using GPU to compute original distance matrix")
    # use L2 distance
    if feature_normed:
        original_dist = 2 - 2 * paddle.matmul(feat, feat, transpose_y=True)
    else:
        original_dist = paddle.pow(feat, 2).sum(axis=1, keepdim=True).expand([num_all, num_all]) + \
            paddle.pow(feat, 2).sum(axis=1, keepdim=True).expand([num_all, num_all]).t()
        original_dist = original_dist.addmm(feat, feat.t(), -2.0, 1.0)
    original_dist = original_dist.numpy()
    del feat

    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argpartition(original_dist, range(1, k1 + 1))
    logger.info("Start re-ranking...")

    for p in range(num_all):
        # compute R(p,k1)
        p_k_reciprocal_ind = k_reciprocal_neighbor(initial_rank, p, k1)

        # compute R*(p,k1)=R(p,k1)∪R(q,k1/2)
        # s.t. |R(p,k1)∩R(q,k1/2)|>=2/3|R(q,k1/2)|, ∀q∈R(p,k1)
        p_k_reciprocal_exp_ind = p_k_reciprocal_ind
        for _, q in enumerate(p_k_reciprocal_ind):
            q_k_reciprocal_ind = k_reciprocal_neighbor(initial_rank, q,
                                                       int(np.around(k1 / 2)))
            if len(
                    np.intersect1d(
                        p_k_reciprocal_ind,
                        q_k_reciprocal_ind,
                        assume_unique=True)) > 2 / 3 * len(q_k_reciprocal_ind):
                p_k_reciprocal_exp_ind = np.append(p_k_reciprocal_exp_ind,
                                                   q_k_reciprocal_ind)
        p_k_reciprocal_exp_ind = np.unique(p_k_reciprocal_exp_ind)
        # reweight distance using gaussian kernel
        weight = np.exp(-original_dist[p, p_k_reciprocal_exp_ind])
        V[p, p_k_reciprocal_exp_ind] = weight / np.sum(weight)

    # local query expansion
    original_dist = original_dist[:num_query, ]
    if k2 > 1:
        try:
            # use sparse tensor to speed up query expansion
            indices = (np.repeat(np.arange(num_all), k2),
                       initial_rank[:, :k2].reshape([-1, ]))
            values = np.array(
                [1 / k2 for _ in range(num_all * k2)], dtype="float16")
            V = scipy.sparse.coo_matrix(
                (values, indices), V.shape,
                dtype="float16") @V.astype("float16")
        except Exception as e:
            logger.info(
                f"Failed to do local query expansion with sparse tensor for reason: \n{e}\n"
                f"now use for-loop instead")
            # use vanilla for-loop
            V_qe = np.zeros_like(V, dtype=np.float16)
            for i in range(num_all):
                V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
            V = V_qe
            del V_qe
    del initial_rank

    # cache k-reciprocal sets which contains gj
    invIndex = []
    for gj in range(num_all):
        invIndex.append(np.nonzero(V[:, gj])[0])

    # compute jaccard distance
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)
    for p in range(num_query):
        sum_min = np.zeros(shape=[1, num_all], dtype=np.float16)
        gj_ind = np.nonzero(V[p, :])[0]
        gj_ind_inv = [invIndex[gj] for gj in gj_ind]
        for j, gj in enumerate(gj_ind):
            gi = gj_ind_inv[j]
            sum_min[0, gi] += np.minimum(V[p, gj], V[gi, gj])
        jaccard_dist[p] = 1 - sum_min / (2 - sum_min)

    # fuse jaccard distance with original distance
    final_dist = (1 - lamb) * jaccard_dist + lamb * original_dist
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:num_query, num_query:]
    final_dist = paddle.to_tensor(final_dist)
    return final_dist

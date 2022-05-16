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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import numpy as np
import platform
import paddle
import sklearn
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

from ppcls.utils.misc import AverageMeter
from ppcls.utils import logger


def fuse_features_with_norm(stacked_embeddings, stacked_norms):
    assert stacked_embeddings.ndim == 3  # (n_features_to_fuse, batch_size, channel)
    assert stacked_norms.ndim == 3  # (n_features_to_fuse, batch_size, 1)
    pre_norm_embeddings = stacked_embeddings * stacked_norms
    fused = pre_norm_embeddings.sum(axis=0)
    norm = paddle.norm(fused, 2, 1, True)
    fused = paddle.divide(fused, norm)
    return fused, norm


def adaface_eval(engine, epoch_id=0):
    output_info = dict()
    time_info = {
        "batch_cost": AverageMeter(
            "batch_cost", '.5f', postfix=" s,"),
        "reader_cost": AverageMeter(
            "reader_cost", ".5f", postfix=" s,"),
    }
    print_batch_step = engine.config["Global"]["print_batch_step"]

    metric_key = None
    tic = time.time()
    unique_dict = {}
    for iter_id, batch in enumerate(engine.eval_dataloader):
        images, labels, dataname, image_index = batch
        if iter_id == 5:
            for key in time_info:
                time_info[key].reset()
        time_info["reader_cost"].update(time.time() - tic)
        batch_size = images.shape[0]
        batch[0] = paddle.to_tensor(images)
        embeddings = engine.model(images, labels)['features']
        norms = paddle.divide(embeddings, paddle.norm(embeddings, 2, 1, True))
        embeddings = paddle.divide(embeddings, norms)
        fliped_images = paddle.flip(images, axis=[3])
        flipped_embeddings = engine.model(fliped_images, labels)['features']
        flipped_norms = paddle.divide(
            flipped_embeddings, paddle.norm(flipped_embeddings, 2, 1, True))
        flipped_embeddings = paddle.divide(flipped_embeddings, flipped_norms)
        stacked_embeddings = paddle.stack(
            [embeddings, flipped_embeddings], axis=0)
        stacked_norms = paddle.stack([norms, flipped_norms], axis=0)
        embeddings, norms = fuse_features_with_norm(stacked_embeddings,
                                                    stacked_norms)

        for out, nor, label, data, idx in zip(embeddings, norms, labels,
                                              dataname, image_index):
            unique_dict[int(idx.numpy())] = {
                'output': out,
                'norm': nor,
                'target': label,
                'dataname': data
            }
            #  calc metric
        time_info["batch_cost"].update(time.time() - tic)
        if iter_id % print_batch_step == 0:
            time_msg = "s, ".join([
                "{}: {:.5f}".format(key, time_info[key].avg)
                for key in time_info
            ])

            ips_msg = "ips: {:.5f} images/sec".format(
                batch_size / time_info["batch_cost"].avg)

            metric_msg = ", ".join([
                "{}: {:.5f}".format(key, output_info[key].val)
                for key in output_info
            ])
            logger.info("[Eval][Epoch {}][Iter: {}/{}]{}, {}, {}".format(
                epoch_id, iter_id,
                len(engine.eval_dataloader), metric_msg, time_msg, ips_msg))

        tic = time.time()

    unique_keys = sorted(unique_dict.keys())
    all_output_tensor = paddle.stack(
        [unique_dict[key]['output'] for key in unique_keys], axis=0)
    all_norm_tensor = paddle.stack(
        [unique_dict[key]['norm'] for key in unique_keys], axis=0)
    all_target_tensor = paddle.stack(
        [unique_dict[key]['target'] for key in unique_keys], axis=0)
    all_dataname_tensor = paddle.stack(
        [unique_dict[key]['dataname'] for key in unique_keys], axis=0)

    eval_result = cal_metric(all_output_tensor, all_norm_tensor,
                             all_target_tensor, all_dataname_tensor)

    metric_msg = ", ".join([
        "{}: {:.5f}".format(key, output_info[key].avg) for key in output_info
    ])
    face_msg = ", ".join([
        "{}: {:.5f}".format(key, eval_result[key])
        for key in eval_result.keys()
    ])
    logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg + ", " +
                                                 face_msg))

    # return 1st metric in the dict
    return eval_result['all_test_acc']


def cal_metric(all_output_tensor, all_norm_tensor, all_target_tensor,
               all_dataname_tensor):
    all_target_tensor = all_target_tensor.reshape([-1])
    all_dataname_tensor = all_dataname_tensor.reshape([-1])
    dataname_to_idx = {
        "agedb_30": 0,
        "cfp_fp": 1,
        "lfw": 2,
        "cplfw": 3,
        "calfw": 4
    }
    idx_to_dataname = {val: key for key, val in dataname_to_idx.items()}
    test_logs = {}
    # _, indices = paddle.unique(all_dataname_tensor, return_index=True, return_inverse=False, return_counts=False)
    for dataname_idx in all_dataname_tensor.unique():
        dataname = idx_to_dataname[dataname_idx.item()]
        # per dataset evaluation
        embeddings = all_output_tensor[all_dataname_tensor ==
                                       dataname_idx].numpy()
        labels = all_target_tensor[all_dataname_tensor == dataname_idx].numpy()
        issame = labels[0::2]
        tpr, fpr, accuracy, best_thresholds = evaluate_face(
            embeddings, issame, nrof_folds=10)
        acc, best_threshold = accuracy.mean(), best_thresholds.mean()

        num_test_samples = len(embeddings)
        test_logs[f'{dataname}_test_acc'] = acc
        test_logs[f'{dataname}_test_best_threshold'] = best_threshold
        test_logs[f'{dataname}_num_test_samples'] = num_test_samples

    test_acc = np.mean([
        test_logs[f'{dataname}_test_acc']
        for dataname in dataname_to_idx.keys()
        if f'{dataname}_test_acc' in test_logs
    ])

    test_logs['all_test_acc'] = test_acc
    return test_logs


def evaluate_face(embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, best_thresholds = calculate_roc(
        thresholds,
        embeddings1,
        embeddings2,
        np.asarray(actual_issame),
        nrof_folds=nrof_folds,
        pca=pca)
    return tpr, fpr, accuracy, best_thresholds


def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)
    dist = None

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # print('train_set', train_set)
        # print('test_set', test_set)
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            # print(_embed_train.shape)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            # print(embed1.shape, embed2.shape)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[
                fold_idx, threshold_idx], _ = calculate_accuracy(
                    threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(
            np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

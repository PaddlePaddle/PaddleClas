import time

import numpy as np
import paddle
from scipy.special import comb
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from ppcls.utils import logger
from ppcls.utils.misc import AverageMeter


def f1_score(query_labels, cluster_labels):
    # compute tp_plus_fp
    qlabels_set, qlabels_counts = np.unique(query_labels, return_counts=True)
    tp_plut_fp = sum([comb(item, 2) for item in qlabels_counts if item > 1])

    # compute tp
    tp = sum([
        sum([
            comb(item, 2)
            for item in np.unique(cluster_labels[query_labels == query_label],
                                  return_counts=True)[1] if item > 1
        ]) for query_label in qlabels_set
    ])

    # compute fp
    fp = tp_plut_fp - tp

    # compute fn
    fn = sum([
        comb(item, 2)
        for item in np.unique(cluster_labels, return_counts=True)[1]
        if item > 1
    ]) - tp

    # compute F1
    P, R = tp / (tp + fp), tp / (tp + fn)
    F1 = 2 * P * R / (P + R)
    return F1


def get_relevance_mask(shape, gt_labels, embeds_same_source, label_counts):
    relevance_mask = np.zeros(shape=shape, dtype=np.int)
    for k, v in label_counts.items():
        matching_rows = np.where(gt_labels == k)[0]
        max_column = v - 1 if embeds_same_source else v
        relevance_mask[matching_rows, :max_column] = 1
    return relevance_mask


def get_label_counts(ref_labels):
    unique_labels, label_counts = np.unique(ref_labels, return_counts=True)
    num_k = min(1023, int(np.max(label_counts)))
    return {k: v for k, v in zip(unique_labels, label_counts)}, num_k


def r_precision(knn_labels, gt_labels, embeds_same_source, label_counts):
    relevance_mask = get_relevance_mask(knn_labels.shape, gt_labels,
                                        embeds_same_source, label_counts)
    matches_per_row = np.sum(
        (knn_labels == gt_labels) * relevance_mask.astype(bool), axis=1)
    max_possible_matches_per_row = np.sum(relevance_mask, axis=1)
    accuracy_per_sample = matches_per_row / max_possible_matches_per_row
    return np.mean(accuracy_per_sample)


def mean_average_precision_at_r(knn_labels, gt_labels, embeds_same_source,
                                label_counts):
    relevance_mask = get_relevance_mask(knn_labels.shape, gt_labels,
                                        embeds_same_source, label_counts)
    num_samples, num_k = knn_labels.shape
    equality = (knn_labels == gt_labels) * relevance_mask.astype(bool)
    cumulative_correct = np.cumsum(equality, axis=1)
    k_idx = np.tile(np.arange(1, num_k + 1), (num_samples, 1))
    precision_at_ks = (cumulative_correct * equality) / k_idx
    summed_precision_pre_row = np.sum(precision_at_ks * relevance_mask, axis=1)
    max_possible_matches_per_row = np.sum(relevance_mask, axis=1)
    accuracy_per_sample = summed_precision_pre_row / max_possible_matches_per_row
    return np.mean(accuracy_per_sample)


def get_lone_query_labels(query_labels, ref_labels, ref_label_counts,
                          embeds_same_source):
    if embeds_same_source:
        return np.array([k for k, v in ref_label_counts.items() if v <= 1])
    else:
        return np.setdiff1d(query_labels, ref_labels)


def get_knn(ref_embeds, embeds, k, embeds_same_source=False):
    d = ref_embeds.shape[1]
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(ref_embeds)
    distances, indices = neigh.kneighbors(embeds, k + 1)
    if embeds_same_source:
        return indices[:, 1:], distances[:, 1:]
    else:
        return indices[:, :k], distances[:, :k]


def run_kmeans(x, num_clusters):
    _, d = x.shape

    # k-means
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x)
    return kmeans.labels_


def calculate_mean_average_precision_at_r(knn_labels, query_labels,
                                          not_lone_query_mask,
                                          embeds_same_source, label_counts):
    if not any(not_lone_query_mask):
        return 0
    knn_labels, query_labels = knn_labels[not_lone_query_mask], query_labels[
        not_lone_query_mask]
    return mean_average_precision_at_r(knn_labels, query_labels[:, None],
                                       embeds_same_source, label_counts)


def calculate_r_precision(knn_labels, query_labels, not_lone_query_mask,
                          embeds_same_source, label_counts):
    if not any(not_lone_query_mask):
        return 0
    knn_labels, query_labels = knn_labels[not_lone_query_mask], query_labels[
        not_lone_query_mask]
    return r_precision(knn_labels, query_labels[:, None], embeds_same_source,
                       label_counts)


def recall_at_k(knn_labels, gt_labels, k):
    accuracy_per_sample = np.array([
        float(gt_label in recalled_predictions[:k])
        for gt_label, recalled_predictions in zip(gt_labels, knn_labels)
    ])
    return np.mean(accuracy_per_sample)


def idml_eval(engine, epoch_id=0):
    time_info = {
        "batch_cost": AverageMeter("batch_cost", '.5f', postfix=" s,"),
        "reader_cost": AverageMeter("reader_cost", ".5f", postfix=" s,"),
    }
    print_batch_step = engine.config["Global"]["print_batch_step"]

    tic = time.time()
    embeddings = []
    labels = []
    for iter_id, batch in enumerate(engine.eval_dataloader):
        if iter_id == 5:
            for key in time_info:
                time_info[key].reset()
        time_info["reader_cost"].update(time.time() - tic)
        batch_size = batch[0].shape[0]
        embedding = engine.model(batch[0])['features']
        label = batch[1]
        embeddings.append(embedding.numpy())
        labels.append(label.numpy())
        time_info["batch_cost"].update(time.time() - tic)
        if iter_id % print_batch_step == 0:
            time_msg = "s, ".join([
                "{}: {:.5f}".format(key, time_info[key].avg)
                for key in time_info
            ])

            ips_msg = "ips: {:.5f} images/sec".format(
                batch_size / time_info["batch_cost"].avg)
            logger.info("[Eval][Epoch {}][Iter: {}/{}]{}, {}".format(
                epoch_id, iter_id, len(engine.eval_dataloader), time_msg,
                ips_msg))

        tic = time.time()
    logger.info("[Eval][Epoch {}]: Calculating metrics...".format(epoch_id))
    n_classes = engine.eval_dataloader.dataset.class_num
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    label_counts, num_k = get_label_counts(labels)
    knn_indices, knn_distances = get_knn(embeddings, embeddings, num_k, True)
    knn_labels = labels[knn_indices]
    lone_query_labels = get_lone_query_labels(labels, labels, label_counts,
                                              True)
    not_lone_query_mask = ~np.isin(labels, lone_query_labels)

    cluster_labels = run_kmeans(embeddings, n_classes)
    NMI = normalized_mutual_info_score(labels, cluster_labels)
    F1 = f1_score(labels, cluster_labels)
    MAP = calculate_mean_average_precision_at_r(knn_labels, labels,
                                                not_lone_query_mask, True,
                                                label_counts)
    RP = calculate_r_precision(knn_labels, labels, not_lone_query_mask, True,
                               label_counts)
    recall_all_k = []
    for k in [1, 2, 4, 8]:
        recall = recall_at_k(knn_labels, labels, k)
        recall_all_k.append(recall)

    metric_dict = {
        "F1": F1,
        "NMI": NMI,
        "recall@1": recall_all_k[0],
        "recall@2": recall_all_k[1],
        "recall@4": recall_all_k[2],
        "recall@8": recall_all_k[3],
        "MAP@R": MAP,
        "RP": RP
    }
    metric_msg = ", ".join([
        "{}: {:.5f}".format(key, value) for key, value in metric_dict.items()
    ])
    logger.info("[Eval][Epoch {}]: {}".format(epoch_id, metric_msg))
    return recall_all_k[0]

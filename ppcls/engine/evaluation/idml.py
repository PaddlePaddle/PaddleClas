import time

import numpy as np
import paddle
from sklearn.neighbors import NearestNeighbors

from ppcls.utils import logger
from ppcls.utils.misc import AverageMeter


def get_label_counts(ref_labels):
    unique_labels, label_counts = np.unique(ref_labels, return_counts=True)
    num_k = int(np.max(label_counts))
    return {k: v for k, v in zip(unique_labels, label_counts)}, num_k


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


def idml_eval(engine, epoch_id=0):
    time_info = {
        "batch_cost": AverageMeter(
            "batch_cost", '.5f', postfix=" s,"),
        "reader_cost": AverageMeter(
            "reader_cost", ".5f", postfix=" s,"),
    }
    print_batch_step = engine.config["Global"]["print_batch_step"]

    tic = time.time()
    embeddings = []
    labels = []
    for iter_id, batch in enumerate(engine.gallery_query_dataloader):
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
                epoch_id, iter_id,
                len(engine.gallery_query_dataloader), time_msg, ips_msg))

        tic = time.time()
    logger.info("[Eval][Epoch {}]: Calculating metrics...".format(epoch_id))
    n_classes = engine.gallery_query_dataloader.dataset.class_num
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    label_counts, num_k = get_label_counts(labels)
    knn_indices, knn_distances = get_knn(embeddings, embeddings, num_k, True)
    knn_labels = labels[knn_indices]
    lone_query_labels = get_lone_query_labels(labels, labels, label_counts,
                                              True)
    not_lone_query_mask = ~np.isin(labels, lone_query_labels)

    knn_labels, labels = knn_labels[not_lone_query_mask], labels[
        not_lone_query_mask]
    metric_dict = engine.eval_metric_func(knn_labels, labels)

    metric_key = None
    metric_info_list = []
    for key in metric_dict:
        if metric_key is None:
            metric_key = key
        metric_info_list.append("{}: {:.5f}".format(key, metric_dict[key]))
    metric_msg = ", ".join(metric_info_list)
    logger.info("metric [Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))
    return metric_dict[metric_key]

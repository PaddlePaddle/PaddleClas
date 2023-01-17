from __future__ import absolute_import, division, print_function
import time
import paddle
from ppcls.engine.train.train_fixmatch import get_loss
from ppcls.engine.train.utils import update_loss, update_metric, log_info
from ppcls.utils import profiler
from paddle.nn import functional as F
import numpy as np
import paddle


def train_epoch_fixmatch_ccssl(engine, epoch_id, print_batch_step):
    tic = time.time()
    if not hasattr(engine, 'train_dataloader_iter'):
        engine.train_dataloader_iter = iter(engine.train_dataloader)
        engine.unlabel_train_dataloader_iter = iter(engine.unlabel_train_dataloader)
    
    temperture = engine.config['SSL'].get("T", 1)
    threshold = engine.config['SSL'].get("threshold", 0.95)
    assert engine.iter_per_epoch is not None, "Global.iter_per_epoch need to be set"
    threshold = paddle.to_tensor(threshold)

    for iter_id in range(engine.iter_per_epoch):
        if iter_id >= engine.iter_per_epoch:
            break

        if iter_id == 5:
            for key in engine.time_info:
                engine.time_info[key].reset()

        try:
            label_data_batch = engine.train_dataloader_iter.next()
        except Exception:
            engine.train_dataloader_iter = iter(engine.train_dataloader)
            label_data_batch = engine.train_dataloader_iter.next()

        try:
            unlabel_data_batch = engine.unlabel_train_dataloader_iter.next()
        except Exception:
            engine.unlabel_train_dataloader_iter = iter(engine.unlabel_train_dataloader)
            unlabel_data_batch = engine.unlabel_train_dataloader_iter.next()

        assert len(unlabel_data_batch) in [3, 4]
        assert unlabel_data_batch[0].shape == unlabel_data_batch[1].shape == unlabel_data_batch[2].shape

        engine.time_info['reader_cost'].update(time.time() - tic)
        batch_size = label_data_batch[0].shape[0] \
                    + unlabel_data_batch[0].shape[0] \
                    + unlabel_data_batch[1].shape[0] \
                    + unlabel_data_batch[2].shape[0]
        engine.global_step += 1

        inputs_x, targets_x = label_data_batch
        inputs_w, inputs_s1, inputs_s2 = unlabel_data_batch[:3]
        batch_size_label = inputs_x.shape[0]
        inputs = paddle.concat([inputs_x, inputs_w, inputs_s1, inputs_s2], axis=0)

        loss_dict, logits_label = get_loss(engine, inputs, batch_size_label, 
                                           temperture, threshold, targets_x,
                                           )
        loss = loss_dict['loss']
        loss.backward()
        
        for i in range(len(engine.optimizer)):
            engine.optimizer[i].step()
        
        for i in range(len(engine.lr_sch)):
            if not getattr(engine.lr_sch[i], 'by_epoch', False):
                engine.lr_sch[i].step()

        for i in range(len(engine.optimizer)):
            engine.optimizer[i].clear_grad()

        if engine.ema:
            engine.model_ema.update(engine.model)
        update_metric(engine, logits_label, label_data_batch, batch_size)
        update_loss(engine, loss_dict, batch_size)
        engine.time_info['batch_cost'].update(time.time() - tic)
        if iter_id % print_batch_step == 0:
            log_info(engine, batch_size, epoch_id, iter_id)

        tic = time.time()

    for i in range(len(engine.lr_sch)):
        if getattr(engine.lr_sch[i], 'by_epoch', False):
            engine.lr_sch[i].step()

def get_loss(engine,
             inputs,
             batch_size_label,
             temperture,
             threshold,
             targets_x,
             **kwargs
             ):
    out = engine.model(inputs)

    logits, feats = out['logits'], out['features']
    feat_w, feat_s1, feat_s2 = feats[batch_size_label:].chunk(3)
    feat_x = feats[:batch_size_label]
    logits_x = logits[:batch_size_label]
    logits_w, logits_s1, logits_s2 = logits[batch_size_label:].chunk(3)
    loss_dict_label = engine.train_loss_func(logits_x, targets_x)
    probs_u_w = F.softmax(logits_w.detach() / temperture, axis=-1)
    max_probs, p_targets_u_w = probs_u_w.max(axis=-1), probs_u_w.argmax(axis=-1)
    mask = paddle.greater_equal(max_probs, threshold).astype('float')
    
    feats = paddle.concat([feat_s1.unsqueeze(1), feat_s2.unsqueeze(1)], axis=1)
    batch = {'logits_w': logits_w,
             'logits_s1': logits_s1,
             'p_targets_u_w': p_targets_u_w,
             'mask': mask,
             'max_probs': max_probs,
             }
    unlabel_loss = engine.unlabel_train_loss_func(feats, batch)
    loss_dict = {}
    for k, v in loss_dict_label.items():
        if k != 'loss':
            loss_dict[k] = v
    for k, v in unlabel_loss.items():
        if k != 'loss':
            loss_dict[k] = v
    loss_dict['loss'] = loss_dict_label['loss'] + unlabel_loss['loss']

    return loss_dict, logits_x

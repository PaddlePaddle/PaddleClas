import paddle
import math
import sys
from ppcls.engine.train.utils import update_loss, log_info
import time
import numpy as np
import paddle.distributed as dist

def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            # can not use `stop_gradient`
            p.clear_grad()

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def train_epoch_ibot(engine, epoch_id, print_batch_step):
    tic = time.time()
    lr_schedule = cosine_scheduler(
        engine.config["Global"]["lr"] * engine.config["Global"]["batch_size"] * dist.get_world_size() / 256,
        engine.config["Global"]["min_lr"],
        engine.config["Global"]["epochs"],len(engine.train_dataloader),
        warmup_epochs=engine.config["Global"]["warmup_epochs"]
    )
    wd_schedule = cosine_scheduler(
        engine.config["Global"]["weight_decay"],
        engine.config["Global"]["weight_decay_end"],
        engine.config["Global"]["epochs"],len(engine.train_dataloader)
    )
    momentum_schedule = cosine_scheduler(engine.config["Global"]["momentum_teacher"], 1,
                                               engine.config["Global"]["epochs"],len(engine.train_dataloader))

    for iter_id, (images, labels,masks) in enumerate(engine.train_dataloader):
        cur_iter_num = len(engine.train_dataloader) * epoch_id + iter_id

        if iter_id == 5:
            for key in engine.time_info:
                engine.time_info[key].reset()
        engine.time_info["reader_cost"].update(time.time() - tic)

        # for i, param_group in enumerate(engine.optimizer[0]._param_groups):
        #     engine.optimizer[0].set_lr(lr_schedule[cur_iter_num])  #报错
            # if i == 0:
            #     # only the first group is regularized
            #     param_group["weight_decay"] = wd_schedule[cur_iter_num]  #报错

        if engine.amp:
            amp_level = engine.config['AMP'].get("level", "O1").upper()
            with paddle.amp.auto_cast(level=amp_level):
                student_out, teacher_out, student_local_cls = engine.model.forward(images,masks)
                loss = engine.train_loss_func.loss_func[0](student_out, teacher_out,student_local_cls, masks, epoch_id)
        else:
            student_out, teacher_out, student_local_cls = engine.model.forward(images,masks)
            loss = engine.train_loss_func.loss_func[0](student_out, teacher_out,student_local_cls, masks, epoch_id)

        loss = loss["loss"]
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # log statistics
        # probs1 = teacher_output[0].chunk(args.global_crops_number)
        # probs2 = student_output[0].chunk(args.global_crops_number)
        #
        # if dist.is_initialized():
        #     pred1 = utils.concat_all_gather(paddle.argmax(probs1[0], axis=1))
        #     pred2 = utils.concat_all_gather(paddle.argmax(probs2[1], axis=1))
        # else:
        #     pred1 = paddle.argmax(probs1[0], axis=1)
        #     pred2 = paddle.argmax(probs2[1], axis=1)
        #
        # acc = ((pred1 == pred2).sum()) / pred1.shape[0]
        # pred_labels.append(pred1)
        # if dist.is_initialized():
        #     real_labels.append(utils.concat_all_gather(labels.cuda()))
        # else:
        #     real_labels.append(labels.cuda())

        # clear grad
        for i in range(len(engine.optimizer)):
            engine.optimizer[i].clear_grad()

            # student update
            if engine.amp:
                engine.scaler.scale(loss).backward()
                if engine.optimizer[0]._grad_clip is not None:
                    engine.scaler.unscale_(engine.optimizer[0])
                cancel_gradients_last_layer(epoch_id, engine.model.student, engine.config["Global"]["freeze_last_layer"])
                for i in range(len(engine.optimizer)):
                    engine.scaler.step(engine.optimizer[i])
                engine.scaler.update()
            else:
                loss.backward()
                cancel_gradients_last_layer(epoch_id, engine.model.student, engine.config["Global"]["freeze_last_layer"])
                for i in range(len(engine.optimizer)):
                    engine.optimizer[i].step()

            # step lr(by step)
            for i in range(len(engine.lr_sch)):
                if not getattr(engine.lr_sch[i], "by_epoch", False):
                    engine.lr_sch[i].step()

            batch_size = engine.train_dataloader.batch_size
            update_loss(engine, loss, batch_size)
            engine.time_info["batch_cost"].update(time.time() - tic)

            if iter_id % print_batch_step == 0:
                log_info(engine, batch_size, epoch_id, iter_id)

            # EMA update for the teacher
            with paddle.no_grad():
                m = momentum_schedule[iter_id]
                for param_stu, params_tea in zip(engine.model.student.parameters(),
                                                 engine.model.teacher_without_ddp.parameters()):
                    new_val = m * params_tea.numpy() + (1 - m) * param_stu.detach().numpy()
                    params_tea.set_value(new_val)

            tic = time.time()
            paddle.device.cuda.synchronize()
            engine.output_info['train_loss'] = loss.item()
            engine.output_info['train_lr'] = engine.optimizer.get_lr()
            engine.output_info['train_wd'] = engine.optimizer._param_groups[0]["weight_decay"]
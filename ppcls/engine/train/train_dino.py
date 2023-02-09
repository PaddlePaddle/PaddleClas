import paddle
import math
import sys
from ppcls.engine.train.utils import cancel_gradients_last_layer, update_loss, log_info


def train_epoch_dino(engine, epoch_id, print_batch_step, lr_schedule, wd_schedule, momentum_schedule, freeze_last_layer):
    for iter_id, images in enumerate(engine.train_dataloader):
        cur_iter_num = len(engine.train_dataloader) * epoch_id + iter_id
        engine.optimizer.set_lr(lr_schedule[cur_iter_num])

        for i, param_group in enumerate(engine.optimizer._param_groups):
            if i == 0:
                # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[cur_iter_num]

        if engine.amp:
            amp_level = engine.config['AMP'].get("level", "O1").upper()
            with paddle.amp.auto_cast(level=amp_level):
                student_out, teacher_out = engine.model.forward(images[0])
                loss = engine.train_loss_func(student_out, teacher_out, epoch_id)
        else:
            student_out, teacher_out = engine.model.forward(images[0])
            loss = engine.train_loss_func(student_out, teacher_out, epoch_id)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        engine.optimizer.clear_grad()

        # student update
        if engine.amp:
            engine.scaler.scale(loss).backward()
            if engine.optimizer._grad_clip is not None:
                engine.scaler.unscale_(engine.optimizer)
            cancel_gradients_last_layer(epoch_id, engine.model.student, freeze_last_layer)
            engine.scaler.step(engine.optimizer)
            engine.scaler.update()
        else:
            loss.backward()
            cancel_gradients_last_layer(epoch_id, engine.model.student, freeze_last_layer)
            engine.optimizer.step()

        batch_size = engine.train_dataloader.batch_size
        update_loss(engine, loss, batch_size)
        if iter_id % print_batch_step == 0:
            log_info(engine, batch_size, epoch_id, iter_id)

        # EMA update for the teacher
        with paddle.no_grad():
            m = momentum_schedule[cur_iter_num]
            for param_stu, params_tea in zip(engine.model.student.parameters(), engine.model.teacher_without_ddp.parameters()):
                new_val = m * params_tea.numpy() + (1 - m) * param_stu.detach().numpy()
                params_tea.set_value(new_val)

        engine.output_info['train_loss'] = loss.item()
        engine.output_info['train_lr'] = engine.optimizer.get_lr()
        engine.output_info['train_wd'] = engine.optimizer._param_groups[0]["weight_decay"]

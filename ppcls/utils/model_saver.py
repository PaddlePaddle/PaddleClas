import os
import paddle

from . import logger


def _mkdir_if_not_exist(path):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    'be happy if some process has already created {}'.format(
                        path))
            else:
                raise OSError('Failed to mkdir {}'.format(path))


def _extract_student_weights(all_params, student_prefix="Student."):
    s_params = {
        key[len(student_prefix):]: all_params[key]
        for key in all_params if student_prefix in key
    }
    return s_params


class ModelSaver(object):
    def __init__(self,
                 engine,
                 net_name="model",
                 loss_name="train_loss_func",
                 opt_name="optimizer",
                 model_ema_name="model_ema"):
        # net, loss, opt, model_ema, output_dir, 
        self.engine = engine
        self.net_name = net_name
        self.loss_name = loss_name
        self.opt_name = opt_name
        self.model_ema_name = model_ema_name

        arch_name = engine.config["Arch"]["name"]
        self.output_dir = os.path.join(engine.output_dir, arch_name)
        _mkdir_if_not_exist(self.output_dir)

    def save(self, metric_info, prefix='ppcls', save_student_model=False):

        if paddle.distributed.get_rank() != 0:
            return

        save_dir = os.path.join(self.output_dir, prefix)

        params_state_dict = getattr(self.engine, self.net_name).state_dict()
        loss = getattr(self.engine, self.loss_name)
        if loss is not None:
            loss_state_dict = loss.state_dict()
            keys_inter = set(params_state_dict.keys()) & set(
                loss_state_dict.keys())
            assert len(keys_inter) == 0, \
                f"keys in model and loss state_dict must be unique, but got intersection {keys_inter}"
            params_state_dict.update(loss_state_dict)

        if save_student_model:
            s_params = _extract_student_weights(params_state_dict)
            if len(s_params) > 0:
                paddle.save(s_params, save_dir + "_student.pdparams")

        paddle.save(params_state_dict, save_dir + ".pdparams")
        model_ema = getattr(self.engine, self.model_ema_name)
        if model_ema is not None:
            paddle.save(model_ema.module.state_dict(),
                        save_dir + ".ema.pdparams")
        optimizer = getattr(self.engine, self.opt_name)
        paddle.save([opt.state_dict() for opt in optimizer],
                    save_dir + ".pdopt")
        paddle.save(metric_info, save_dir + ".pdstates")
        logger.info("Already save model in {}".format(save_dir))

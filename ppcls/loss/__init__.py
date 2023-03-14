import copy

import paddle
import paddle.nn as nn
from ..utils import logger
from ..utils.amp import AMPForwardDecorator, AMP_forward_decorator

from .celoss import CELoss
from .googlenetloss import GoogLeNetLoss
from .centerloss import CenterLoss
from .contrasiveloss import ContrastiveLoss
from .contrasiveloss import ContrastiveLoss_XBM
from .emlloss import EmlLoss
from .msmloss import MSMLoss
from .npairsloss import NpairsLoss
from .trihardloss import TriHardLoss
from .triplet import TripletLoss, TripletLossV2
from .tripletangularmarginloss import TripletAngularMarginLoss, TripletAngularMarginLoss_XBM
from .supconloss import SupConLoss
from .softsuploss import SoftSupConLoss
from .ccssl_loss import CCSSLCELoss
from .pairwisecosface import PairwiseCosface
from .dmlloss import DMLLoss
from .distanceloss import DistanceLoss
from .softtargetceloss import SoftTargetCrossEntropy
from .distillationloss import DistillationCELoss
from .distillationloss import DistillationGTCELoss
from .distillationloss import DistillationDMLLoss
from .distillationloss import DistillationDistanceLoss
from .distillationloss import DistillationRKDLoss
from .distillationloss import DistillationKLDivLoss
from .distillationloss import DistillationDKDLoss
from .distillationloss import DistillationWSLLoss
from .distillationloss import DistillationSKDLoss
from .distillationloss import DistillationMultiLabelLoss
from .distillationloss import DistillationDISTLoss
from .distillationloss import DistillationPairLoss

from .multilabelloss import MultiLabelLoss
from .afdloss import AFDLoss

from .deephashloss import DSHSDLoss
from .deephashloss import LCDSHLoss
from .deephashloss import DCHLoss

from .metabinloss import CELossForMetaBIN
from .metabinloss import TripletLossForMetaBIN
from .metabinloss import InterDomainShuffleLoss
from .metabinloss import IntraDomainScatterLoss


class CombinedLoss(nn.Layer):
    def __init__(self, config_list, mode, amp_config=None):
        super().__init__()
        self.mode = mode
        loss_func = []
        self.loss_weight = []
        assert isinstance(config_list, list), (
            'operator config should be a list')
        for config in config_list:
            assert isinstance(config,
                              dict) and len(config) == 1, "yaml format error"
            name = list(config)[0]
            param = config[name]
            assert "weight" in param, "weight must be in param, but param just contains {}".format(
                param.keys())
            self.loss_weight.append(param.pop("weight"))
            loss_func.append(eval(name)(**param))
        self.loss_func = nn.LayerList(loss_func)
        logger.debug("build loss {} success.".format(loss_func))

        self.scaler = None
        if amp_config:
            if self.mode == "Train" or AMPForwardDecorator.amp_eval:
                self.scaler = paddle.amp.GradScaler(
                    init_loss_scaling=amp_config.get("scale_loss", 1.0),
                    use_dynamic_loss_scaling=amp_config.get(
                        "use_dynamic_loss_scaling", False))

    @AMP_forward_decorator
    def __call__(self, input, batch):
        loss_dict = {}
        # just for accelerate classification traing speed
        if len(self.loss_func) == 1:
            loss = self.loss_func[0](input, batch)
            loss_dict.update(loss)
            loss_dict["loss"] = list(loss.values())[0]
        else:
            for idx, loss_func in enumerate(self.loss_func):
                loss = loss_func(input, batch)
                weight = self.loss_weight[idx]
                loss = {key: loss[key] * weight for key in loss}
                loss_dict.update(loss)
            loss_dict["loss"] = paddle.add_n(list(loss_dict.values()))

        if self.scaler:
            self.scaler(loss_dict["loss"])
        return loss_dict


def build_loss(config, mode):
    if config["Loss"][mode] is None:
        return None
    module_class = CombinedLoss(
        copy.deepcopy(config["Loss"][mode]),
        mode,
        amp_config=config.get("AMP", None))

    if AMPForwardDecorator.amp_level is not None:
        if mode == "Train" or AMPForwardDecorator.amp_eval:
            module_class = paddle.amp.decorate(
                models=module_class,
                level=AMPForwardDecorator.amp_level,
                save_dtype='float32')

    logger.debug("build loss {} success.".format(module_class))
    return module_class

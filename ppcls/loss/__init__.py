import copy

import paddle
import paddle.nn as nn
from ppcls.utils import logger

from .celoss import CELoss, MixCELoss
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
    def __init__(self, config_list):
        super().__init__()
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
        return loss_dict


def build_loss(config, mode="train"):
    train_loss_func, unlabel_train_loss_func, eval_loss_func = None, None, None
    if mode == "train":
        label_loss_info = config["Loss"]["Train"]
        if label_loss_info:
            train_loss_func = CombinedLoss(copy.deepcopy(label_loss_info))
        unlabel_loss_info = config.get("UnLabelLoss", {}).get("Train", None)
        if unlabel_loss_info:
            unlabel_train_loss_func = CombinedLoss(
                copy.deepcopy(unlabel_loss_info))
    if mode == "eval" or (mode == "train" and
                          config["Global"]["eval_during_train"]):
        loss_config = config.get("Loss", None)
        if loss_config is not None:
            loss_config = loss_config.get("Eval")
            if loss_config is not None:
                eval_loss_func = CombinedLoss(copy.deepcopy(loss_config))

    return train_loss_func, unlabel_train_loss_func, eval_loss_func

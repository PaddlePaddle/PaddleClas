import importlib
import copy

from ppcls.loss.celoss import CELoss
from ppcls.loss.combined_loss import CombinedLoss
from ppcls.loss.triplet import TripletLoss, TripletLossV2
from ppcls.loss.msmloss import MSMLoss
from ppcls.loss.emlloss import EmlLoss
from ppcls.loss.npairsloss import NpairsLoss
from ppcls.loss.trihardloss import TriHardLoss
from ppcls.loss.centerloss import CenterLoss


def build_loss(config):
    config = copy.deepcopy(config)
    model_type = config.pop("name")
    module = importlib.import_module(__name__)
    arch = getattr(module, model_type)(**config)
    return arch

from typing import List
import torch
from methods.base.model import BaseModel
import tqdm
from torch import nn
import torch.nn.functional as F
from abc import abstractmethod

from utils.dl.common.model import LayerActivation
from .mmd import mmd_rbf
from ..pretrain_or_ft import PretrainOrFineTuningModel


class FeatureAlignmentModel(PretrainOrFineTuningModel):
    @abstractmethod
    def get_feature_hook(self):
        pass

    @abstractmethod
    def get_trained_params(self):
        pass
    
    @abstractmethod
    def get_params_names_of_each_block(self):
        pass
    
    def get_mmd_loss(self, f1: torch.Tensor, f2: torch.Tensor):
        return mmd_rbf(f1, f2)
    
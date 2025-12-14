from typing import List
import torch
from methods.base.model import BaseModel
import tqdm
from torch import nn
import torch.nn.functional as F
from abc import abstractmethod

from utils.dl.common.model import LayerActivation


class PretrainOrFineTuningModel(BaseModel):
    def get_required_model_components(self) -> List[str]:
        return ['main']
    
    @abstractmethod
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        raise NotImplementedError
    
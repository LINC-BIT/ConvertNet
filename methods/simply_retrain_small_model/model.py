from ..train_with_fbs.model import TrainWithFBSModel
from abc import abstractmethod
import torch
from torch import nn 


class SimplyRetrainSmallModelModel(TrainWithFBSModel):
    @abstractmethod
    def generate_small_model(self, given_target_samples: torch.Tensor) -> nn.Module:
        """
        Return generated small model and extracted features from given target samples
        """
        pass
    
    @abstractmethod
    def get_feature_hook(self):
        pass
from ..train_with_fbs.model import TrainWithFBSModel
from abc import abstractmethod
import torch
from torch import nn 


class GenScalingLawDataPointsModel(TrainWithFBSModel):
    
    @abstractmethod
    def get_feature_hook(self):
        pass
    
    @abstractmethod
    def get_output_entropy(self, samples):
        pass
from typing import List
from ..gen_scaling_law_data_points.model import GenScalingLawDataPointsModel
from abc import abstractmethod
import torch
from torch import nn 


class EdgeTAOnlineRunModel(GenScalingLawDataPointsModel):
    def get_required_model_components(self) -> List[str]:
        return ['fm', 'neuron_index', 'knowledge_base', 'scaling_law', 'profiles', 'bn_stats']
    
    @property
    def model(self):
        return self.knowledge_base
    
    # @model.setter
    # def model(self, _model):
    #     self.knowledge_base = _model
    
    @property
    def fm(self):
        return self.models_dict['fm']
    
    @property
    def neuron_index(self):
        return self.models_dict['neuron_index']
    
    @property
    def knowledge_base(self):
        return self.models_dict['knowledge_base']
    
    @knowledge_base.setter
    def knowledge_base(self, _kb):
        self.models_dict['knowledge_base'] = _kb
    
    @property
    def scaling_law(self):
        return self.models_dict['scaling_law']
    
    @property
    def profiles(self):
        return self.models_dict['profiles']
    
    @property
    def bn_stats(self):
        return self.models_dict['bn_stats']
    
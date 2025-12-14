from abc import abstractmethod
from ..pretrain_or_ft.model import PretrainOrFineTuningModel


class LoRAFineTuningModel(PretrainOrFineTuningModel):
    @abstractmethod
    def get_head_params(self):
        pass
    
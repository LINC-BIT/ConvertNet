from typing import List

from torch.nn.modules import Module
from methods.pretrain_or_ft import PretrainOrFineTuningModel
from methods.train_with_fbs import TrainWithFBSModel
from methods.lora_ft import LoRAFineTuningModel
from methods.gen_knowledge_base import GenKnowledgeBaseModel
from methods.gen_neuron_index import GenNeuronIndexModel
from methods.gen_scaling_law_data_points import GenScalingLawDataPointsModel
from methods.train_with_fbs.lib import clear_cache
from methods.simply_retrain_small_model import SimplyRetrainSmallModelModel
from methods.feature_alignment import FeatureAlignmentModel
from methods.feature_alignment_lora.model import FeatureAlignmentLoraModel
from methods.edgeta_online_run.model import EdgeTAOnlineRunModel
import torch
import tqdm
import copy
import random
from torch import nn
import torch.nn.functional as F

from utils.common.others import HiddenPrints
from utils.dl.common.model import get_module, set_module, get_model_size, LayerActivation2
from utils.common.log import logger

from torch.cuda.amp import autocast


class PretrainOrFineTuningModel_ViTImageClassification(PretrainOrFineTuningModel):
    def get_required_model_components(self) -> List[str]:
        return ['main']
    
    def get_accuracy(self, test_loader, *args, **kwargs):
        acc = 0
        sample_num = 0
        
        self.to_eval_mode()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y) in pbar:
                x, y = x.to(self.device), y.to(self.device)
                
                with autocast(self.fp16, dtype=torch.float16):
                    output = self.infer(x)
                pred = F.softmax(output, dim=1).argmax(dim=1)
                #correct = torch.eq(torch.argmax(output.logits,dim = 1), y).sum().item()
                correct = torch.eq(pred, y).sum().item()
                acc += correct
                sample_num += len(y)
                
                pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_correct: {correct}, '
                                     f'cur_batch_acc: {(correct / len(y)):.4f}')

        acc /= sample_num
        return acc
        
    def infer(self, x, *args, **kwargs):
        return self.model(pixel_values=x).logits
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        return F.cross_entropy(self.infer(x), y)
    
    
class TrainWithFBSModel_ViTImageClassification(TrainWithFBSModel, PretrainOrFineTuningModel_ViTImageClassification):
    pass


class LoRAFineTuningModel_ViTImageClassification(LoRAFineTuningModel, PretrainOrFineTuningModel_ViTImageClassification):
    def get_head_params(self):
        return self.model.classifier.parameters()

class GenKnowledgeBaseModel_ViTImageClassification(GenKnowledgeBaseModel, PretrainOrFineTuningModel_ViTImageClassification):
    pass

class GenNeuronIndexModel_ViTImageClassification(GenNeuronIndexModel, PretrainOrFineTuningModel_ViTImageClassification):
    pass


class GenScalingLawDataPointsModel_ViTImageClassification(GenScalingLawDataPointsModel, PretrainOrFineTuningModel_ViTImageClassification):
    def get_output_entropy(self, samples):
        x = self.infer(samples)
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)
    
    def get_feature_hook(self):
        def guess_linear_name():
            cand_linear_names = ['fc', 'linear', 'classifier']
            for name in cand_linear_names:
                if get_module(self.model, name) is not None:
                    return name
            raise NotImplementedError
        
        return LayerActivation2(get_module(self.model, guess_linear_name()))
    
    
class FeatureAlignmentModel_ViTImageClassification(FeatureAlignmentModel, PretrainOrFineTuningModel_ViTImageClassification):
    def guess_linear_name(self):
        cand_linear_names = ['fc', 'linear', 'classifier']
        for name in cand_linear_names:
            if get_module(self.model, name) is not None:
                return name
        raise NotImplementedError
    
    def get_feature_hook(self):
        return LayerActivation2(get_module(self.model, self.guess_linear_name()))
    
    def get_trained_params(self):
        res = []
        
        linear_name = self.guess_linear_name()
        for name, param in self.model.named_parameters():
            if name.startswith(linear_name):
                continue
            res += [param]
        return res
    
    def get_params_names_of_prefix(self, prefix):
        return [n for n, _ in self.model.named_parameters() if n.startswith(prefix)]
    
    def get_params_names_of_each_block(self):
        return [self.get_params_names_of_prefix('vit.embeddings')] + \
            [self.get_params_names_of_prefix(f'vit.encoder.layer.{i}') for i in range(12)] + \
            [self.get_params_names_of_prefix('classifier')]
    
    
class FeatureAlignmentLoraModel_ViTImageClassification(FeatureAlignmentLoraModel, FeatureAlignmentModel_ViTImageClassification):
    pass


class EdgeTAOnlineRunModel_ViTImageClassification(EdgeTAOnlineRunModel, GenScalingLawDataPointsModel_ViTImageClassification):
    pass
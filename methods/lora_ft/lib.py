import torch
from torch import nn 
from utils.dl.common.model import set_module
from utils.common.log import logger


class LoRA(nn.Linear):
    pass


class ToQorKorV_WrappedWithLoRA(nn.Module):
    def __init__(self, fc: nn.Linear, ab_r: int):
        super(ToQorKorV_WrappedWithLoRA, self).__init__()
        
        self.fc = fc
        self.ab = self.create_ab_as_linear(fc.weight.data, ab_r)
        
    def create_ab_as_linear(self, fc_weight: torch.Tensor, ab_r: int):
        n = max(1, fc_weight.size(0) // ab_r)
        res = nn.Sequential(
            LoRA(fc_weight.size(1), n, bias=False),
            LoRA(n, fc_weight.size(0), bias=False)
        ).to(fc_weight.device)
        nn.init.kaiming_uniform_(res[0].weight, a=5 ** 0.5)
        nn.init.zeros_(res[1].weight)
        return res
        
    def forward(self, x):
        x1 = self.fc(x)
        x2 = self.ab(x)
        return x1 + x2


def add_lora_ab_to_model(model: nn.Module, qkv_layers_name, qkv_as_a_whole, ab_r: int):
    model.eval()
    
    for name, module in model.named_modules():
        if name in qkv_layers_name:
            if not qkv_as_a_whole:
                logger.debug(f'add lora to {name}')
                set_module(model, name, ToQorKorV_WrappedWithLoRA(module, ab_r))
            else:
                raise NotImplementedError
    
    return model


def train_only_lora(model: nn.Module):
    res = []
    for n, m in model.named_modules():
        if isinstance(m, LoRA):
            for p in m.parameters():
                p.requires_grad = True
                res += [p]
        else:
            for p in m.parameters():
                p.requires_grad = False
    return res


def absorb_lora(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, ToQorKorV_WrappedWithLoRA):
            fc = module.fc
            ab = module.ab

            fc.weight.add_(ab[1].weight @ ab[0].weight)
            
            set_module(model, name, fc)
    return model
    
import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
from utils.dl.common.model import get_parameter, set_module
import tqdm
from utils.dl.common.model import get_model_device, get_module, set_module
from transformers.pytorch_utils import prune_linear_layer
import torch
from torch import nn 
import copy
from methods.edgeta_online_run.lib_transformer import prune_linear_layer_and_its_after_layer
from utils.common.log import logger
import os


def transform_linear_layer_and_its_after_layer(model, layer_name, after_layer_name, attention_value, device):
    
    # dummy_input = torch.rand((4, get_module(model, layer_name + '.0').in_features)).to(device)
    # module = nn.Sequential(get_module(model, layer_name + '.0'), 
    #                        get_module(model, layer_name + '.1'), 
    #                        get_module(model, after_layer_name))
    
    # o1 = module(dummy_input)
    
    unpruned_neurons_idx = attention_value.argsort(descending=True)[0: get_module(model, layer_name + '.0').out_features].sort()[0]
    attention_value = attention_value[unpruned_neurons_idx]
    unpruned_neurons_idx = attention_value.argsort(descending=True)
    
    set_module(model, layer_name + '.0', prune_linear_layer(get_module(model, layer_name + '.0'), unpruned_neurons_idx.to(device)))
    static_fbs = get_module(model, layer_name + '.1')
    static_fbs.w[0] = static_fbs.w[0][unpruned_neurons_idx]
    set_module(model, after_layer_name, prune_linear_layer(
        get_module(model, after_layer_name),
        unpruned_neurons_idx.to(device),
        dim=1
    ))
    
    return unpruned_neurons_idx
    
    # module = nn.Sequential(get_module(model, layer_name + '.0'), 
    #                        get_module(model, layer_name + '.1'), 
    #                        get_module(model, after_layer_name))
    # o2 = module(dummy_input)
    
    # diff = (o1 - o2).abs().sum()
    # assert diff < 1e-2, f'{diff}_{layer_name}_{after_layer_name}'
    
    
def make_linear_layer_and_its_after_layer_dynamic(model, layer_name, after_layer_name, neurons_groups, device, res_save_dir):
    
    # dummy_input = torch.rand((4, get_module(model, layer_name + '.0').in_features)).to(device)
    # module = nn.Sequential(get_module(model, layer_name + '.0'), 
    #                        get_module(model, layer_name + '.1'), 
    #                        get_module(model, after_layer_name))
    
    # o1 = module(dummy_input)
    
    # unpruned_neurons_idx = attention_value.argsort(descending=True)[0: get_module(model, layer_name + '.0').out_features].sort()[0]
    # attention_value = attention_value[unpruned_neurons_idx]
    # unpruned_neurons_idx = attention_value.argsort(descending=True)
    
    set_module(model, layer_name + '.0', NestedLinearLayerDim0(get_module(model, layer_name + '.0'), neurons_groups, 
                                                      os.path.join(res_save_dir, f'{layer_name}.0.pt')))
    # static_fbs = get_module(model, layer_name + '.1')
    # static_fbs.w[0] = static_fbs.w[0][unpruned_neurons_idx]
    set_module(model, layer_name + '.1', NestedStaticFBS(get_module(model, layer_name + '.1'), neurons_groups))
    
    
    set_module(model, after_layer_name, NestedLinearLayerDim1(get_module(model, after_layer_name), neurons_groups, 
                                                      os.path.join(res_save_dir, f'{after_layer_name}.pt')))
    
    
    # module = nn.Sequential(get_module(model, layer_name + '.0'), 
    #                        get_module(model, layer_name + '.1'), 
    #                        get_module(model, after_layer_name))
    # o2 = module(dummy_input)
    
    # diff = (o1 - o2).abs().sum()
    # assert diff < 1e-2, f'{diff}_{layer_name}_{after_layer_name}'


def equivalent_transform_neurons_according_to_importance(model: nn.Module, attention_values_of_layers,
                                                         qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name):
    
    large_model = copy.deepcopy(model)
    device = get_model_device(large_model)
    
    recovering_neurons_order_index_of_layers = {}
    
    for fbs_layer in attention_values_of_layers.keys():
        attention_value = attention_values_of_layers[fbs_layer]
        
        logger.debug(f'fbs_layer: {fbs_layer}')
        
        from utils.common.data import flatten_2d_arr
        qkv_layers_name = flatten_2d_arr(qkv_layers_name)
        for qkv_layer_name in qkv_layers_name:
            if not fbs_layer.startswith(qkv_layer_name):
                continue
            
            # prune [qkv].0 and [qkv].1
            unpruned_neurons_idx = transform_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1', 
                                                       attention_value, device)
            recovering_neurons_order_index_of_layers[fbs_layer] = unpruned_neurons_idx.argsort()
            break
        
        for proj_layer_name in proj_layers_name:
            if not fbs_layer.startswith(proj_layer_name):
                continue
            
            # prune [proj].0 and [proj].1
            unpruned_neurons_idx = transform_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1', 
                                                       attention_value, device)
            recovering_neurons_order_index_of_layers[fbs_layer] = unpruned_neurons_idx.argsort()
            break
        
        if isinstance(ff1_layers_name[0], list):
            for i, ff1_layer_name in enumerate(flatten_2d_arr(ff1_layers_name)):
                if not fbs_layer.startswith(ff1_layer_name):
                    continue
                # prune [ff1].0 and [ff2].0
                unpruned_neurons_idx = transform_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1',
                                                           attention_value, device)
                recovering_neurons_order_index_of_layers[fbs_layer] = unpruned_neurons_idx.argsort()
                break
        else:
            for i, ff1_layer_name in enumerate(ff1_layers_name):
                if not fbs_layer.startswith(ff1_layer_name):
                    continue
                # prune ff1 and ff2
                unpruned_neurons_idx = transform_linear_layer_and_its_after_layer(large_model, fbs_layer, ff2_layers_name[i], 
                                                           attention_value, device)
                recovering_neurons_order_index_of_layers[fbs_layer] = unpruned_neurons_idx.argsort()
                break
            
        if isinstance(ff1_layers_name[0], list):
            for i, ff2_layer_name in enumerate(ff2_layers_name):
                if not fbs_layer.startswith(ff2_layer_name):
                    continue
                # prune [ff1].0 and [ff2].0
                unpruned_neurons_idx = transform_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1',
                                                           attention_value, device)
                recovering_neurons_order_index_of_layers[fbs_layer] = unpruned_neurons_idx.argsort()
    
    return large_model, recovering_neurons_order_index_of_layers

class NestedLinearLayerDim0(nn.Module):
    def __init__(self, raw_layer: nn.Linear, neurons_groups: list, res_save_path) -> None:
        super(NestedLinearLayerDim0, self).__init__()
        
        self.nested_layers = []
        
        for ni, neurons_group in enumerate(neurons_groups):
            if len(neurons_group) == 0:
                continue
            layer = nn.Linear(raw_layer.in_features, len(neurons_group), bias=raw_layer.bias is not None).to(raw_layer.weight.device)
            layer.weight.data = raw_layer.weight[neurons_group]
            if raw_layer.bias is not None:
                layer.bias.data = raw_layer.bias[neurons_group]
            self.nested_layers.append(layer)
        # elif dim == 1:
        #     neurons_group = torch.cat(neurons_groups, dim=0)
        #     layer = nn.Linear(len(neurons_group), raw_layer.out_features, bias=raw_layer.bias is not None).to(raw_layer.weight.device)
        #     layer.weight.data = raw_layer.weight[:, neurons_group]
        #     if raw_layer.bias is not None:
        #         layer.bias.data = raw_layer.bias
        #     self.nested_layers.append(layer)
            
        self.nested_layers = nn.ModuleList(self.nested_layers)
        
        self.path = res_save_path
        self.save(res_save_path)
        
    def forward(self, x):
        return torch.cat([layer(x) for layer in list(self.nested_layers.children())], dim=-1)
    
    def save(self, path):
        for layer_i, layer in enumerate(self.nested_layers):
            torch.save(layer, f'{path}.{layer_i}')
            
    def load(self, path, n):
        for layer_i in range(len(self.nested_layers), n + 1):
            self.nested_layers.append(torch.load(f'{path}.{layer_i}'))
    
    def offload(self, n):
        for _ in range(len(self.nested_layers) - 1, n, -1):
            self.nested_layers = nn.ModuleList(list(self.nested_layers.children())[:-1])
            
    def scale(self, n):
        if n == len(self.nested_layers) - 1:
            return
        elif n > len(self.nested_layers) - 1:
            self.load(self.path, n)
        else:
            self.offload(n)
            
            
class NestedLinearLayerDim1(nn.Module):
    def __init__(self, raw_layer: nn.Linear, neurons_groups: list, res_save_path) -> None:
        super(NestedLinearLayerDim1, self).__init__()
        
        self.neurons_groups = neurons_groups
        
        neurons_group = torch.cat(neurons_groups, dim=0)
        layer = nn.Linear(len(neurons_group), raw_layer.out_features, bias=raw_layer.bias is not None).to(raw_layer.weight.device)
        layer.weight.data = raw_layer.weight[:, neurons_group]
        if raw_layer.bias is not None:
            layer.bias.data = raw_layer.bias
        self.nested_layer = layer
            
        # self.nested_layers = nn.ModuleList(self.nested_layers)
        
        self.path = res_save_path
        self.save(res_save_path)
        
    def forward(self, x):
        # return torch.cat([layer(x) for layer in list(self.nested_layers.children())], dim=-1)
        return self.nested_layer(x)
    
    def save(self, path):
        torch.save(self.nested_layer, f'{path}.full')
            
    def scale(self, n):
        raw_layer = torch.load(f'{self.path}.full')
        neurons_group = torch.cat(self.neurons_groups[0: n + 1], dim=0)
        layer = nn.Linear(len(neurons_group), raw_layer.out_features, bias=raw_layer.bias is not None).to(raw_layer.weight.device)
        layer.weight.data = raw_layer.weight[:, neurons_group]
        if raw_layer.bias is not None:
            layer.bias.data = raw_layer.bias
        self.nested_layer = layer
        
    # def offload(self, n):
    #     raw_layer = torch.load(f'{self.path}.full')
    #     neurons_group = torch.cat(self.neurons_groups[0: n + 1], dim=0)
    #     layer = nn.Linear(len(neurons_group), raw_layer.out_features, bias=raw_layer.bias is not None).to(raw_layer.weight.device)
    #     layer.weight.data = raw_layer.weight[:, neurons_group]
    #     if raw_layer.bias is not None:
    #         layer.bias.data = raw_layer.bias
    #     self.nested_layer = layer
            
    # def scale(self, n):
    #     if n == len(self.nested_layers) - 1:
    #         return
    #     elif n > len(self.nested_layers) - 1:
    #         self.load(self.path, n)
    #     else:
    #         self.offload(n)
            
            
class NestedStaticFBS(nn.Module):
    def __init__(self, raw_layer, neurons_groups: list):
        super(NestedStaticFBS, self).__init__()
        # assert w.dim() == 2 and w.size(0) == 1
        self.raw_w = raw_layer.w.data # (1, dim)
        self.w = nn.Parameter(raw_layer.w.data)
        # if window_merge is not None:
        #     self.register_buffer('window_merge', torch.tensor(window_merge, device=w.device))
        self.window_merge = raw_layer.window_merge
        
        self.neurons_groups = neurons_groups
        
    def forward(self, x):
        if self.window_merge is None:
            return x * self.w.unsqueeze(1)
        # print(11, self.window_merge, x.size())
        return x * self.w.repeat(x.size(0), 1).unsqueeze(1)
    
    def __repr__(self):
        return f'StaticFBS({self.w.size(1)})'

    def scale(self, n):
        # print(self.raw_w.size(), torch.cat(self.neurons_groups[0: n + 1]))
        self.w = nn.Parameter(self.raw_w[:, torch.cat(self.neurons_groups[0: n + 1])])
    

def make_model_to_dynamic_nestnn(model, neurons_groups_of_layers, 
                                                         qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name, res_save_dir):
    # import os
    
    # for name, module in model.named_modules():
    #     if isinstance(module, nn.Linear):
    #         set_module(model, name, NestedLinearLayer(module, neurons_groups_of_layers[name], 
    #                                                   os.path.join(res_save_dir, f'{name}.pt')))

    large_model = model
    device = get_model_device(large_model)
    
    # recovering_neurons_order_index_of_layers = {}
    
    for fbs_layer in neurons_groups_of_layers.keys():
        neurons_groups = neurons_groups_of_layers[fbs_layer]
        
        logger.debug(f'fbs_layer: {fbs_layer}')
        
        from utils.common.data import flatten_2d_arr
        qkv_layers_name = flatten_2d_arr(qkv_layers_name)
        for qkv_layer_name in qkv_layers_name:
            if not fbs_layer.startswith(qkv_layer_name):
                continue
            
            # prune [qkv].0 and [qkv].1
            make_linear_layer_and_its_after_layer_dynamic(large_model, fbs_layer, fbs_layer[0: -2] + '.1', 
                                                       neurons_groups, device, res_save_dir)
            break
        
        for proj_layer_name in proj_layers_name:
            if not fbs_layer.startswith(proj_layer_name):
                continue
            
            # prune [proj].0 and [proj].1
            make_linear_layer_and_its_after_layer_dynamic(large_model, fbs_layer, fbs_layer[0: -2] + '.1', 
                                                       neurons_groups, device, res_save_dir)
            break
        
        if isinstance(ff1_layers_name[0], list):
            for i, ff1_layer_name in enumerate(flatten_2d_arr(ff1_layers_name)):
                if not fbs_layer.startswith(ff1_layer_name):
                    continue
                # prune [ff1].0 and [ff2].0
                make_linear_layer_and_its_after_layer_dynamic(large_model, fbs_layer, fbs_layer[0: -2] + '.1',
                                                           neurons_groups, device, res_save_dir)
                break
        else:
            for i, ff1_layer_name in enumerate(ff1_layers_name):
                if not fbs_layer.startswith(ff1_layer_name):
                    continue
                # prune ff1 and ff2
                make_linear_layer_and_its_after_layer_dynamic(large_model, fbs_layer, ff2_layers_name[i], 
                                                           neurons_groups, device, res_save_dir)
                break
            
        if isinstance(ff1_layers_name[0], list):
            for i, ff2_layer_name in enumerate(ff2_layers_name):
                if not fbs_layer.startswith(ff2_layer_name):
                    continue
                # prune [ff1].0 and [ff2].0
                make_linear_layer_and_its_after_layer_dynamic(large_model, fbs_layer, fbs_layer[0: -2] + '.1',
                                                           neurons_groups, device, res_save_dir)
    
    return large_model
    

def preserve_first_n_nested_components(dynamic_nestnn: nn.Module, n):
    logger.info(f'preserve_first_n_nested_components: {n}')
    for name, module in dynamic_nestnn.named_modules():
        if isinstance(module, (NestedLinearLayerDim0, NestedLinearLayerDim1, NestedStaticFBS)):
            logger.debug(f'before module {name}: {module}')
            module.scale(n)
            logger.debug(f'after module {name}: {module}')


def retrain_first_n_nested_components(model: nn.Module, n: int):
    logger.info(f'retrain_first_n_nested_components: {n}')
    retrained_p = []
    retrained_p_name = []
    
    for name, p in model.named_parameters():
        if 'nested_layer.' in name:
            p.requires_grad = True
            retrained_p.append(p)
            retrained_p_name.append(name)
        
        if 'nested_layers.' in name:
            need_retraining = False
            for ni in range(n):
                if f'nested_layers.{ni}' in name:
                    need_retraining = True
                    break
            
            if need_retraining:
                p.requires_grad = True
                retrained_p.append(p)
                retrained_p_name.append(name)
            else:
                p.requires_grad = False
            
            continue
        
        # if 'classifier' not in name and 'fc' not in name and 'head' not in name and 'lm_head' not in name:
        #     p.requires_grad = False
        # else:
        p.requires_grad = True
        
    return retrained_p


def retrain_components(model: nn.Module, vn: int, h_index: list, params_names_of_each_block):
    # params_names_of_each_block should include the layers before and after Transformer blocks
    
    v_retrained_p_name = []
    
    for name, p in model.named_parameters():
        if 'nested_layer.' in name:
            p.requires_grad = True
            v_retrained_p_name.append(name)
        
        if 'nested_layers.' in name:
            need_retraining = False
            for ni in range(vn):
                if f'nested_layers.{ni}' in name:
                    need_retraining = True
                    break
            
            if need_retraining:
                v_retrained_p_name.append(name)
    
    h_retrained_p_name = [params_names_of_each_block[h] for h in h_index]
    from utils.common.data import flatten_2d_arr
    h_retrained_p_name = flatten_2d_arr(h_retrained_p_name)
    
    res = set(v_retrained_p_name) & set(h_retrained_p_name)
    res = list(res)
    
    for p in model.parameters():
        p.requires_grad = False
        
    trained_params = []
    for p_name in res:
        p = get_parameter(model, p_name)
        p.requires_grad = True
        trained_params += [p]
    
    return trained_params


def calculate_blocks_importance(model: nn.Module, params_names_of_each_block):
    blocks_importance = []
    
    for params_names_in_a_block in params_names_of_each_block:
        importances = []
        
        for name in params_names_in_a_block:
            p = get_parameter(model, name)
            p_grad = p.grad
            
            if p_grad is None:
                continue
            importances += [(p * p_grad).abs().mean()]
        
        avg_importance = sum(importances) / len(importances)
        blocks_importance += [float(avg_importance)]
    
    return blocks_importance
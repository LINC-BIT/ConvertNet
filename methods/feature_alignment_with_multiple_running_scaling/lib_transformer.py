from utils.dl.common.model import get_model_device, get_module, set_module
from transformers.pytorch_utils import prune_linear_layer
import torch
from torch import nn 
import copy
from ..edgeta_online_run.lib_transformer import prune_linear_layer_and_its_after_layer
from utils.common.log import logger
import os


def init_dynamic_scaling_model(large_model: nn.Module, sparsity_list,
                               qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name, res_save_dir,
                               only_add_fbs_in_qkv=False, dummy_input=None, forword_func=None):
    
    os.makedirs(res_save_dir, exist_ok=True)
    
    from ..edgeta_online_run.lib_transformer import generate_small_model
    
    # setted sparsity should be the small sparsity (sparsity_list[-1])
    biggest_small_model, attention_values_of_layers, unpruned_neurons_idx_of_layers = \
        generate_small_model_using_neuron_importance_values(large_model, sparsity_list[-1], qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name, 
                             only_add_fbs_in_qkv)
    logger.debug(f'biggest_small_model: {biggest_small_model}')
    logger.debug(f'attention_values_of_layers: {attention_values_of_layers}')
    biggest_small_model.eval()
    o1 = forword_func(biggest_small_model, dummy_input)
    
    neurons_groups = {}
    for name, module in biggest_small_model.named_modules():
        if name in attention_values_of_layers.keys():
            num_neurons = get_module(biggest_small_model, name + '.0').out_features
            
            num_neurons_in_different_sparsity = [0] + [int(num_neurons / (1 - sparsity_list[-1]) * (1 - sparsity)) for sparsity in sparsity_list]
            
            neurons_groups[name] = []
            for si in range(1, len(num_neurons_in_different_sparsity)):
                neurons_groups[name] += [torch.arange(
                    num_neurons_in_different_sparsity[si - 1],
                    num_neurons_in_different_sparsity[si]
                )]
            
    from .dyna_nest_transformer import equivalent_transform_neurons_according_to_importance, make_model_to_dynamic_nestnn
    transformed_small_model, recovering_neurons_order_index_of_layers = \
        equivalent_transform_neurons_according_to_importance(biggest_small_model, attention_values_of_layers,
                                                             qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name)
    logger.debug(f'transformed_small_model1: {transformed_small_model}')
    transformed_small_model.eval()
    # o2 = transformed_small_model(di)
    o2 = forword_func(transformed_small_model, dummy_input)
    diff = torch.abs(o1 - o2).mean()
    # assert diff < 1e-4, diff
    
    logger.info(f'equivalent transform diff: {diff}')
    
    make_model_to_dynamic_nestnn(transformed_small_model, neurons_groups, qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name,  res_save_dir)
    logger.debug(f'transformed_small_model2: {transformed_small_model}')
    
    transformed_small_model.eval()
    # o3 = transformed_small_model(di)
    o3 = forword_func(transformed_small_model, dummy_input)
    diff = torch.abs(o1 - o3).mean()
    # assert diff < 1e-4, diff
    
    logger.info(f'make dynamic nestnn diff: {diff}')
    
    # save nested components
    
    
    return transformed_small_model, recovering_neurons_order_index_of_layers, unpruned_neurons_idx_of_layers



def generate_small_model_using_neuron_importance_values(large_model: nn.Module, sparsity,
                                                        qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name, 
                                                        only_add_fbs_in_qkv=False):
    
    large_model = copy.deepcopy(large_model)
    device = get_model_device(large_model)
    
    from ..gen_neuron_index.lib_transformer import get_fbs_layers
    
    fbs_layers = get_fbs_layers(qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name, only_add_fbs_in_qkv)
    unpruned_neurons_idx_of_layers = {}
    attention_values_of_layers = {}
    
    for fbs_layer in fbs_layers:
        attention_value = get_module(large_model, fbs_layer).cached_raw_w
        window_merge = getattr(get_module(large_model, fbs_layer), 'window_merge', None)
        
        assert attention_value.size(0) == 1
        attention_value = attention_value[0]
        
        attention_values_of_layers[fbs_layer] = attention_value
        
        # attention_value = attention_values_of_layers[fbs_layer]
        # print(fbs_layer, sparsity, attention_value)
        
        # sparsity = get_module(large_model, fbs_layer).k_takes_all.k 
        
        # unpruned_neurons_idx = attention_value.nonzero(as_tuple=True)[0]
        # pruned_neurons_idx = get_module(large_model, fbs_layer).k_takes_all.cached_i[0].sort()[0]
        
        pruned_neurons_idx = attention_value.sort()[1][0: int(len(attention_value) * sparsity)]
        
        unpruned_neurons_idx = torch.LongTensor([ni for ni in range(len(attention_value)) if ni not in pruned_neurons_idx])
        attention_value = attention_value[unpruned_neurons_idx]
        
        unpruned_neurons_idx_of_layers[fbs_layer + '.raw_linear'] = unpruned_neurons_idx

        set_module(large_model, fbs_layer, get_module(large_model, fbs_layer).raw_linear)
        
        from utils.common.data import flatten_2d_arr
        qkv_layers_name = flatten_2d_arr(qkv_layers_name)
        for qkv_layer_name in qkv_layers_name:
            if not fbs_layer.startswith(qkv_layer_name):
                continue
            
            # prune [qkv].0 and [qkv].1
            prune_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1', 
                                                   unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
            break
        
        for proj_layer_name in proj_layers_name:
            if not fbs_layer.startswith(proj_layer_name):
                continue
            
            # prune [proj].0 and [proj].1
            prune_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1', 
                                                   unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
            break
        
        if isinstance(ff1_layers_name[0], list):
            for i, ff1_layer_name in enumerate(flatten_2d_arr(ff1_layers_name)):
                if not fbs_layer.startswith(ff1_layer_name):
                    continue
                # prune [ff1].0 and [ff2].0
                prune_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1',
                                                       unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
                break
        else:
            for i, ff1_layer_name in enumerate(ff1_layers_name):
                if not fbs_layer.startswith(ff1_layer_name):
                    continue
                # prune ff1 and ff2
                prune_linear_layer_and_its_after_layer(large_model, fbs_layer, ff2_layers_name[i], 
                                                       unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
                break
            
        if isinstance(ff1_layers_name[0], list):
            for i, ff2_layer_name in enumerate(ff2_layers_name):
                if not fbs_layer.startswith(ff2_layer_name):
                    continue
                # prune [ff1].0 and [ff2].0
                prune_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1',
                                                       unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
    
    logger.debug(f'Generated small model: {large_model}')
    
    return large_model, attention_values_of_layers, unpruned_neurons_idx_of_layers
    
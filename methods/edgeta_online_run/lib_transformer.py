from utils.dl.common.model import get_parameter
from methods.gen_scaling_law_data_points.lib_transformer import *
from methods.gen_neuron_index.lib_transformer import *
import torch
from torch import nn 
from copy import deepcopy


def get_matched_p_name_in_kb(proxy_model_p_name: str, proxy_model,
                             qkv_layers_name, proj_layers_name, 
                             ff1_layers_name, ff2_layers_name):
    
    s = proxy_model_p_name.split('.')
    proxy_model_layer_name = '.'.join(s[0: -1])
    proxy_model_p_component = s[-1]
    
    proxy_model_layer = get_module(proxy_model, proxy_model_layer_name)
    proxy_model_p = get_parameter(proxy_model, proxy_model_p_name)
    if isinstance(proxy_model_layer, nn.LayerNorm):
        return proxy_model_p_name
    if proxy_model_p_name.endswith('.w'):
        return None
    
    qkv_layers_name = flatten_2d_arr(qkv_layers_name)
    ff1_layers_name = flatten_2d_arr(ff1_layers_name)
    
    fbs_layers = get_fbs_layers(qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name)
    for fbs_layer in fbs_layers:
        if proxy_model_layer_name[0:-2] == fbs_layer:
            return proxy_model_layer_name[0:-2] + '.raw_linear.' + proxy_model_p_component
        
    for l in qkv_layers_name + proj_layers_name + ff1_layers_name + ff2_layers_name:
        if proxy_model_layer_name == l:
            return proxy_model_p_name
        elif proxy_model_layer_name == l + '.0':
            return proxy_model_layer_name + '.0.' + proxy_model_p_component
        elif proxy_model_layer_name == l + '.1':
            return proxy_model_layer_name + '.' + proxy_model_p_component
    
    return None


@torch.no_grad()
def _compute_diff(old, new):
    return (new - old).norm(1) / old.norm(1)


@torch.no_grad()
def proxy_model_feedback_to_knowledge_base(proxy_model: nn.Module, knowledge_base: nn.Module,
                                           qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name,
                                           unpruned_indexes_of_layers):
    
    logger.info(f'proxy model feedback to knowledge base')
    
    for p_name, p in proxy_model.named_parameters():
        logger.debug(f'if feedback: {p_name}')
        
        matched_md_param_name = get_matched_p_name_in_kb(p_name, proxy_model, qkv_layers_name, 
                                                         proj_layers_name, ff1_layers_name, ff2_layers_name)
        logger.debug(f'matched_md_param_name: {matched_md_param_name}')
        matched_md_param = get_parameter(knowledge_base, matched_md_param_name)
        
        if matched_md_param is None:
            # logger.info(f'not feedback: {p_name}')
            continue
        # logger.info(f'start feedback: {p_name}, {p.size()} -> {matched_md_param.size()}')
        
        matched_md_param = matched_md_param
        
        md_layer_name = '.'.join(matched_md_param_name.split('.')[0: -1])
        if md_layer_name in unpruned_indexes_of_layers.keys():
            cur_unpruned_indexes = unpruned_indexes_of_layers[md_layer_name]
            cur_unpruned_indexes_name = md_layer_name
        
        if p.size() != matched_md_param.size():
            # logger.info(f'cur unpruned indexes: {cur_unpruned_indexes_name}, {cur_unpruned_indexes.size()}')
            
            if p.dim() == 1: # norm
                new_p = deepcopy(matched_md_param)
                new_p[cur_unpruned_indexes] = p
            elif p.dim() == 2: # linear
                try:
                    if p.size(0) < matched_md_param.size(0): # output pruned
                        new_p = deepcopy(matched_md_param)
                        new_p[cur_unpruned_indexes] = p
                    else: # input pruned
                        new_p = deepcopy(matched_md_param)
                        new_p[:, cur_unpruned_indexes] = p
                except:
                    logger.debug(f'failed, skip')
            p = new_p
            
        assert p.size() == matched_md_param.size(), f'{p.size()}, {matched_md_param.size()}'

        res_p = (matched_md_param + p) / 2.
        diff = _compute_diff(matched_md_param, res_p)
        matched_md_param.copy_(res_p)
        logger.debug(f'end feedback: {p_name}, diff: {diff:.6f}')
        

@torch.no_grad()
def knowledge_base_feedback_to_fm(before_updating_knowledge_base, knowledge_base: nn.Module, fm: nn.Module, neuron_index, 
                                  qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name,
                                  feedback_alpha):
    
    logger.info('knowledge base feedback to fm')
        
    for (p_name, p), before_p in zip(knowledge_base.named_parameters(), before_updating_knowledge_base.parameters()):
        
        if p.dim() != 2:
            continue
        
        matched_fm_param_name = get_matched_p_name_in_fm(p_name, qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name)
        matched_fm_param = get_parameter(fm, matched_fm_param_name)
        
        # logger.info(f'if feedback: {p_name}')
        if matched_fm_param_name is None or matched_fm_param is None:
            continue
        if not (p.size(0) == matched_fm_param.size(0) or p.size(1) == matched_fm_param.size(1)):
            continue
        
        if p_name not in neuron_index.keys():
            continue
        # print(self.models_dict['indexes'].keys())
        index = neuron_index[p_name]
        # logger.info(f'start feedback: {p_name}, {p.size()} -> {matched_fm_param.size()}, index: {index.size()}')
        
        p_update = p - before_p
        
        t = False
        if p.dim() > 1 and index.size(0) == p.size(1) and index.size(1) == matched_fm_param.size(1):
            assert p.dim() == 2
            p_update = p_update.T
            matched_fm_param = matched_fm_param.T
            t = True
            # logger.info(f'transpose paramters')
        
        
        if p.dim() == 2:
            # p_update = upsample_2d_tensor(p_update, matched_fm_param.size(1))
            
            p_update = p_update.unsqueeze(1)
            index = index.unsqueeze(-1)
            
            # fast
            # agg_p_update = (p_update * index).sum(0)
            
            # balanced agg
            agg_p_update = 0
        
            cur_split_size = 64
            while index.size(0) % cur_split_size != 0:
                cur_split_size -= 1
            
            for i in range(0, index.size(0), cur_split_size):
                agg_p_update += p_update[i: i + cur_split_size] * index[i: i + cur_split_size]
            agg_p_update = agg_p_update.sum(0)
            
            
        else:
            agg_p_update = (p_update.unsqueeze(1) * index).sum(0)
        
        new_fm_param = matched_fm_param + agg_p_update * feedback_alpha
        
        diff = _compute_diff(matched_fm_param, new_fm_param)
        
        matched_fm_param.copy_(new_fm_param)
        
        logger.debug(f'end feedback: {p_name}, diff: {diff:.6f}')
        
        
@torch.no_grad()
def fm_feedback_to_knowledge_base(knowledge_base: nn.Module, fm: nn.Module, neuron_index, 
                                  qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name,
                                  feedback_alpha):
    
    logger.info('fm feedback to knowledge bae')
    
    for p_name, p in knowledge_base.named_parameters():
        if p.dim() != 2:
            continue
        
        matched_fm_param_name = get_matched_p_name_in_fm(p_name, qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name)
        matched_fm_param = get_parameter(fm, matched_fm_param_name)
        
        # logger.info(f'if feedback: {p_name}')
        if matched_fm_param_name is None or matched_fm_param is None:
            continue
        if not (p.size(0) == matched_fm_param.size(0) or p.size(1) == matched_fm_param.size(1)):
            continue
        if p_name not in neuron_index.keys():
            continue
        index = neuron_index[p_name]
        # logger.info(f'start feedback: {p_name}, {p.size()} -> {matched_fm_param.size()}, index: {index.size()}')
        
        if p.dim() > 1 and index.size(0) == p.size(1) and index.size(1) == matched_fm_param.size(1):
            assert p.dim() == 2
            p = p.T
            matched_fm_param = matched_fm_param.T
        
        if p.dim() == 2:
            # matched_fm_param = downsample_2d_tensor(matched_fm_param, p.size(1))
            
            matched_fm_param = matched_fm_param.unsqueeze(0)
            index = index.unsqueeze(-1)
            
            # fast
            # agg_p_update = (p_update * index).sum(0)
            
            # balanced agg
            agg_fm_param = 0
        
            cur_split_size = 64
            while index.size(1) % cur_split_size != 0:
                cur_split_size -= 1
            
            for i in range(0, index.size(1), cur_split_size):
                agg_fm_param += matched_fm_param[:, i: i + cur_split_size] * index[:, i: i + cur_split_size]
            agg_fm_param = agg_fm_param.sum(1)
            # agg_fm_param = downsample_2d_tensor(agg_fm_param, p.size(1))
            
        else:
            agg_fm_param = (matched_fm_param.unsqueeze(0) * index).sum(1)
            
        diff = _compute_diff(p, agg_fm_param)
        p.copy_(agg_fm_param * feedback_alpha + (1. - feedback_alpha) * p)
        
        logger.debug(f'end feedback: {p_name}, diff: {diff:.6f}')
        
        
if __name__ == '__main__':
    fm = torch.load('dianzixuebao/offline_preparing/vit_b_16/img_cls/results/lora_fine_tune.py/20240415/999997-195115-trial/models/main_best_acc=0.9627.pt')['main']
    knowledge_base = torch.load('dianzixuebao/offline_preparing/vit_b_16/img_cls/results/gen_neuron_index.py/20240416/999971-221828-trial/models/0.90/main_best_acc=0.7073.pt')['main']
    neuron_index = torch.load('dianzixuebao/offline_preparing/vit_b_16/img_cls/results/gen_neuron_index.py/20240416/999971-221828-trial/models/0.90/best_neuron_index.pt')
    # logger.info(knowledge_base.vit.encoder.layer[0])
    
    qkv_layers_name = [[f'vit.encoder.layer.{i}.attention.attention.{k}' for k in ['query', 'key', 'value']] for i in range(12)]
    proj_layers_name = [f'vit.encoder.layer.{i}.attention.output.dense' for i in range(12)]
    ff1_layers_name = [f'vit.encoder.layer.{i}.intermediate.dense' for i in range(12)]
    ff2_layers_name = [f'vit.encoder.layer.{i}.output.dense' for i in range(12)]
    
    from methods.gen_scaling_law_data_points.lib_transformer import generate_small_model
    with torch.no_grad():
        knowledge_base(**{'pixel_values': torch.rand((1, 3, 224, 224)).cuda()})
    proxy_model, unpruned_indexes_of_layers = generate_small_model(knowledge_base, qkv_layers_name, proj_layers_name, 
                                       ff1_layers_name, ff2_layers_name, return_detail=True)
    # logger.info(proxy_model.vit.encoder.layer[0])
    with torch.no_grad():
        for p in proxy_model.parameters():
            p.add_(torch.rand_like(p) * 0.01)
    
    
    proxy_model_feedback_to_knowledge_base(proxy_model, knowledge_base, qkv_layers_name, proj_layers_name, 
                                           ff1_layers_name, ff2_layers_name, unpruned_indexes_of_layers)
    
    logger.info(neuron_index.keys())
    updated_knowledge_base = deepcopy(knowledge_base)
    with torch.no_grad():
        for p in updated_knowledge_base.parameters():
            p.add_(torch.rand_like(p) * 0.01)
    knowledge_base_feedback_to_fm(knowledge_base, updated_knowledge_base, fm, neuron_index, 
                                  qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name, 0.01)
    
    fm_feedback_to_knowledge_base(updated_knowledge_base, fm, neuron_index, 
                                  qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name, 0.01)
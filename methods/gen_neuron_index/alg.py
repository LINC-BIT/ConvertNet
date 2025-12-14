from typing import Any, Dict
from schema import Schema, Or
from data import Scenario, MergedDataset
from methods.base.alg import BaseAlg
from data import build_dataloader
import torch.optim
import tqdm
from utils.dl.common.env import create_tbwriter
import os
from copy import deepcopy
import glob
from torch import nn 

from utils.dl.common.model import get_model_size, get_module, get_parameter, set_module
from .model import GenNeuronIndexModel
from torch.cuda.amp import autocast, GradScaler
from utils.common.log import logger
from utils.common.data import flatten_2d_arr
import random
import numpy as np


class GenNeuronIndexAlg(BaseAlg):
    def get_required_models_schema(self) -> Schema:
        return Schema({
            'fm': GenNeuronIndexModel,
            'kb': GenNeuronIndexModel
        })
        
    def get_required_hyp_schema(self) -> Schema:
        from schema import Optional
        return Schema({
            'launch_tbboard': bool,
            
            'example_sample': object,
            
            'qkv_layers_name': list,
            'proj_layers_name': list,
            'ff1_layers_name': list,
            'ff2_layers_name': list,
            'window_merge': object,
            
            'fbs_r': int,
            'min_sparsity': float,
            'max_sparsity': float,
            'sparsity_loss_alpha': float,
            'neuron_index_loss_alpha': float,
            
            'train_batch_size': int,
            'val_batch_size': int,
            'num_workers': int,            
            'optimizer': str,
            'optimizer_args': dict,
            'neuron_index_optimizer': str,
            'neuron_index_optimizer_args': dict,
            'scheduler': str,
            'scheduler_args': dict,
            'num_iters': int,
            'val_freq': int,
            Optional('only_gen_neuron_in_qkv', default=False): bool,
            Optional('collate_fn', default=None): object,
            Optional('neuron_compute_split_size', default=512): int,
            Optional('use_two_gpu', default=False): bool,
            
            'fp16': bool
        })
        
    def get_matched_p_name_in_fm(self, kb_p_name, hyps):
        from .lib_transformer import get_matched_p_name_in_fm
        return get_matched_p_name_in_fm(kb_p_name, hyps['qkv_layers_name'], hyps['proj_layers_name'], 
                                        hyps['ff1_layers_name'], hyps['ff2_layers_name'])
        # if '.fbs' in kb_p_name:
        #     return None
        
        # qkv_layers_name = flatten_2d_arr(hyps['qkv_layers_name'])
        # ff1_layers_name = flatten_2d_arr(hyps['ff1_layers_name'])
        
        # kb_module_name, p_name = '.'.join(kb_p_name.split('.')[:-1]), kb_p_name.split('.')[-1]
        
        # if kb_module_name.endswith('.raw_linear'):
        #     kb_module_name = kb_module_name[0: -11]

        # # qkv
        # if kb_module_name in qkv_layers_name:
        #     matched_p_name = kb_module_name + '.' + p_name
        # elif kb_module_name[0: -2] in qkv_layers_name:
        #     matched_p_name = kb_module_name[0: -2] + '.' + p_name
        
        # # proj
        # elif kb_module_name in hyps['proj_layers_name']:
        #     matched_p_name = kb_module_name + '.' + p_name
        # elif kb_module_name[0: -2] in hyps['proj_layers_name']:
        #     matched_p_name = kb_module_name[0: -2] + '.' + p_name
        
        # # ff1
        # elif kb_module_name in ff1_layers_name:
        #     matched_p_name = kb_module_name + '.' + p_name
        # elif kb_module_name[0: -2] in ff1_layers_name:
        #     matched_p_name = kb_module_name[0: -2] + '.' + p_name
            
        # # ff2
        # elif kb_module_name in hyps['ff2_layers_name']:
        #     matched_p_name = kb_module_name + '.' + p_name
        # elif kb_module_name[0: -2] in hyps['ff2_layers_name']:
        #     matched_p_name = kb_module_name[0: -2] + '.' + p_name
            
        # else:
        #     return None
        
        # # logger.debug(f'kb p name: {kb_p_name}, matched p name: {matched_p_name}')
        # # return get_parameter(self.models['fm'].model, matched_p_name)
        # return matched_p_name
    
    def init_neuron_index(self, device, hyps):
        neuron_index = {}
        
        for name, p in self.models['kb'].model.named_parameters():
            logger.debug(f'init neuron index | try: layer {name} in kb, {p.size()}')
            
            if p.dim() != 2:
                logger.debug(f'ignore layers that does not have 2-dim parameters')
                continue
            
            matched_p_name_in_fm = self.get_matched_p_name_in_fm(name, hyps)
            logger.debug(f'layer {name} in fm may have matched param: {matched_p_name_in_fm}')
            
            matched_p_in_fm = get_parameter(self.models['fm'].model, matched_p_name_in_fm)
            if matched_p_name_in_fm is None or matched_p_in_fm is None:
                logger.debug(f'layer {name} in fm has no no matched param')
                continue
            
            if not (p.size(0) == matched_p_in_fm.size(0) or p.size(1) == matched_p_in_fm.size(1)):
                logger.debug(f'layer {name} in fm and kb does not match due to static pruning and svd decomposition, skip')
                continue
            
            if not any([name.startswith(layer_name) for layer_name in flatten_2d_arr(hyps['qkv_layers_name'])]) and hyps['only_gen_neuron_in_qkv']:
                logger.debug(f'only_gen_neuron_in_qkv, but layer {name} is not qkv layer, skip')
                continue
            
            logger.debug(f'layer {name} in fm has matched param: {matched_p_in_fm.size()}')
            
            assert p.size(0) == matched_p_in_fm.size(0) or p.size(1) == matched_p_in_fm.size(1), f'{p.size()}, {matched_p_in_fm.size()}'
            
            if p.size(0) == matched_p_in_fm.size(0):
                neuron_index[name] = torch.zeros((p.size(1), matched_p_in_fm.size(1))).to(device)
                logger.debug(f'construct index {neuron_index[name].size()} between fm\'s {matched_p_name_in_fm} ({matched_p_in_fm.size()}) '
                             f'and kb\'s {name} ({p.size()}) in dim 1')

            elif p.size(1) == matched_p_in_fm.size(1):
                neuron_index[name] = torch.zeros((p.size(0), matched_p_in_fm.size(0))).to(device)
                logger.debug(f'construct index {neuron_index[name].size()} between fm\'s {matched_p_name_in_fm} ({matched_p_in_fm.size()}) '
                             f'and kb\'s {name} ({p.size()}) in dim 0')
            
            neuron_index[name].requires_grad = True
        
        return neuron_index
    
    def two_params_diff_fast_no_sample(self, trained_p: torch.Tensor, ref_p: torch.Tensor, 
                             index: torch.Tensor, 
                             split_size: int):

        assert trained_p.dim() == ref_p.dim()
        
        if trained_p.dim() > 1 and index.size(0) == trained_p.size(1) and index.size(1) == ref_p.size(1):
            assert trained_p.dim() == 2
            trained_p = trained_p.T
            ref_p = ref_p.T
        
        assert index.size(0) == trained_p.size(0) and index.size(1) == ref_p.size(0)

        ref_p = ref_p.detach()
        if trained_p.dim() > 1:
            trained_p = trained_p.flatten(1)
            ref_p = ref_p.flatten(1)
            
            index = index.unsqueeze(-1)
        
        if split_size is None:
            # old version: huge memory consumption, not recommended (although this is fastest)
            linear_combed_ref_p = (ref_p.unsqueeze(0) * index).sum(1)
        
        else:
            # new version
            linear_combed_ref_p = 0
            
            cur_split_size = split_size
            while index.size(1) % cur_split_size != 0:
                cur_split_size -= 1
            
            for i in range(0, index.size(1), cur_split_size):
                linear_combed_ref_p += ref_p.unsqueeze(0)[:, i: i + cur_split_size] * index[:, i: i + cur_split_size]
            linear_combed_ref_p = linear_combed_ref_p.sum(1)
            
        diff = (linear_combed_ref_p - trained_p).norm(2) ** 2
        return diff
    
    def get_neuron_index_loss(self, neuron_index, hyps):
        res = 0.

        for name, p in self.models['kb'].model.named_parameters():
            if name not in neuron_index.keys():
                continue
            
            raw_p = get_parameter(self.models['fm'].model, self.get_matched_p_name_in_fm(name, hyps))
            index = neuron_index[name]
            res += self.two_params_diff_fast_no_sample(p, raw_p, index, hyps['neuron_compute_split_size'])
            
        return res
    
    def get_fbs_layers(self, hyps):
        from .lib_transformer import get_fbs_layers
        return get_fbs_layers(hyps['qkv_layers_name'], hyps['proj_layers_name'], hyps['ff1_layers_name'], hyps['ff2_layers_name'])

    def run(self, scenario: Scenario, hyps: Dict) -> Dict[str, Any]:
        super().run(scenario, hyps)
        
        assert isinstance(self.models['fm'], GenNeuronIndexModel) # for auto completion
        assert isinstance(self.models['kb'], GenNeuronIndexModel)
        
        # generate knowledge base by pruning the original large model
        logger.debug(f'fm: {self.models["fm"].model}')
        logger.debug(f'kb: {self.models["kb"].model}')
        
        device = self.models['fm'].device
        
        # add FBS
        fbs_layers = self.get_fbs_layers(hyps)
        logger.debug(fbs_layers)
        
        fbs_ignore_layers = []
        for name, m in self.models['kb'].model.named_modules():
            if isinstance(m, nn.Linear) and name not in fbs_layers:
                fbs_ignore_layers += [name]
        
        from ..train_with_fbs.lib import add_FBS, get_importance_values, set_sparsity, get_l1_reg_in_model, clear_cache
        self.models['kb'].model = add_FBS(self.models['kb'].model, hyps['max_sparsity'], 
                                            hyps['fbs_r'], fbs_ignore_layers, True, 
                                            hyps['example_sample'], 
                                            hyps['window_merge'])
        
        # verify fbs works
        with torch.no_grad():
            set_sparsity(self.models['kb'].model, hyps['max_sparsity'])
            rep_target_sample = hyps['example_sample']
            o1 = self.models['kb'].infer(rep_target_sample)
            from ..gen_scaling_law_data_points.lib_transformer import generate_small_model
            small_model = generate_small_model(self.models['kb'].model, hyps['qkv_layers_name'], hyps['proj_layers_name'], 
                                hyps['ff1_layers_name'], hyps['ff2_layers_name'])
            large_model = self.models['kb'].model
            self.models['kb'].model = small_model
            self.models['kb'].to_eval_mode()
            o2 = self.models['kb'].infer(rep_target_sample)
            self.models['kb'].model = large_model
            diff = ((o1 - o2) ** 2).sum()
            assert diff < 1e-4, diff
            logger.info('FBS verify passed (kb size: {}MB, proxy model size: {}MB, diff: {})'.format(get_model_size(self.models['kb'].model, True), 
                                                                              get_model_size(small_model, True), diff))
        
        if hyps['use_two_gpu']:
            self.models['kb'].model = self.models['kb'].model.cpu()
            self.models['kb'].model = torch.nn.DataParallel(self.models['kb'].model, device_ids=[0, 1])
            self.models['kb'].model = self.models['kb'].model.to('cuda:0')
        
        neuron_index = self.init_neuron_index(device, hyps)
        tmp_indexes_file_path = os.path.join(self.res_save_dir, 'tmp-indexes.pt')
        torch.save([neuron_index], tmp_indexes_file_path)
        logger.info(f'generate neuron indexes ({(os.path.getsize(tmp_indexes_file_path) / 1024**2):.3f}MB)')
        os.remove(tmp_indexes_file_path)
        
        if hyps['fp16']:
            # self.models['main'].model = self.models['main'].model.half()
            scaler = GradScaler()
            self.models['fm'].fp16 = True
            self.models['kb'].fp16 = True
        
        offline_datasets = scenario.get_offline_datasets()
        train_dataset = MergedDataset([d['train'] for d in offline_datasets.values()])
        val_dataset = MergedDataset([d['val'] for d in offline_datasets.values()])
        train_loader = iter(build_dataloader(train_dataset, hyps['train_batch_size'], hyps['num_workers'],
                                        True, None, collate_fn=hyps['collate_fn']))
        val_loader = build_dataloader(val_dataset, hyps['val_batch_size'], hyps['num_workers'],
                                      False, False, collate_fn=hyps['collate_fn'])
        
        max_sparsity = hyps['max_sparsity']
        
        optimizer = torch.optim.__dict__[hyps['optimizer']]([
            {'params': self.models['kb'].model.parameters(), **hyps['optimizer_args']},
            {'params': list(neuron_index.values()), **hyps['neuron_index_optimizer_args']},
        ])
        scheduler = torch.optim.lr_scheduler.__dict__[hyps['scheduler']](optimizer, **hyps['scheduler_args'])
        tb_writer = create_tbwriter(os.path.join(self.res_save_dir, f'{max_sparsity:.2f}/tb_log'), launch_tbboard=hyps['launch_tbboard'])
        pbar = tqdm.tqdm(range(hyps['num_iters']), dynamic_ncols=True)
        
        for p in self.models['fm'].model.parameters():
            p.requires_grad = False
        
        best_avg_val_acc = 0.
        
        for iter_index in pbar:
            
            self.models['kb'].to_train_mode()
            
            if iter_index % 4 == 0:
                cur_sparsity = hyps['min_sparsity']
            elif 1 <= iter_index % 4 <= 2:
                cur_sparsity = random.random() * (hyps['max_sparsity'] - hyps['min_sparsity']) + hyps['min_sparsity']
            elif iter_index % 4 == 3:
                cur_sparsity = hyps['max_sparsity']
            
            set_sparsity(self.models['kb'].model, cur_sparsity)
            
            x, y = next(train_loader)
            if hyps['use_two_gpu']:
                _device = device + ':0'
            else:
                _device = device
            if isinstance(x, dict):
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(_device)
                y = y.to(_device)
            else:
                x, y = x.to(_device), y.to(_device)
                
            if hyps['fp16']:
                with autocast(enabled=True, dtype=torch.float16):
                    task_loss = self.models['kb'].forward_to_get_task_loss(x, y)
                    sparse_loss = hyps['sparsity_loss_alpha'] * get_l1_reg_in_model(self.models['kb'].model)
                    neuron_index_loss = hyps['neuron_index_loss_alpha'] * self.get_neuron_index_loss(neuron_index, hyps)
                    
                    total_loss = task_loss + sparse_loss + neuron_index_loss
                    
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                
            else:

                task_loss = self.models['kb'].forward_to_get_task_loss(x, y)
                sparse_loss = hyps['sparsity_loss_alpha'] * get_l1_reg_in_model(self.models['kb'].model)
                neuron_index_loss = hyps['neuron_index_loss_alpha'] * self.get_neuron_index_loss(neuron_index, hyps)
                
                total_loss = task_loss + sparse_loss + neuron_index_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                
                optimizer.step()
                scheduler.step()
            
            if (iter_index + 1) % 100 == 0:
                importance_values = get_importance_values(self.models['kb'].model)
                for k, v in importance_values.items():
                    tb_writer.add_histogram(f'importance/{k}', v, iter_index)
            
            clear_cache(self.models['kb'].model)
            
            if (iter_index + 1) % hyps['val_freq'] == 0:
                
                avg_val_acc = 0.
                val_accs = {}
                
                cur_model = self.models['kb'].model
                
                for sparsity in tqdm.tqdm(np.linspace(hyps['min_sparsity'], hyps['max_sparsity'], 4), 
                                        desc='val...', dynamic_ncols=True, leave=False):
                    
                    model_for_test = deepcopy(cur_model)
                    val_acc = 0.
                    
                    self.models['kb'].model = model_for_test
                    self.models['kb'].to_eval_mode()
                    set_sparsity(self.models['kb'].model, sparsity)
                    
                    # self.bn_cal(self.models['kb'], train_loader, hyps['bn_cal_num_iters'], device)
                    val_acc = self.models['kb'].get_accuracy(val_loader)
                    
                    avg_val_acc += val_acc
                    val_accs[f'{sparsity:.2f}'] = val_acc
                
                self.models['kb'].model = cur_model
                avg_val_acc /= 4
                tb_writer.add_scalars(f'accs/val_accs', val_accs, iter_index)
                
                if len(glob.glob(os.path.join(self.res_save_dir, f'models/{max_sparsity:.2f}/main_last_*.pt'))) > 0:
                    os.remove(glob.glob(os.path.join(self.res_save_dir, f'models/{max_sparsity:.2f}/main_last_*.pt'))[0])
                self.models['kb'].save_model(os.path.join(self.res_save_dir, f'models/{max_sparsity:.2f}/main_last_acc={avg_val_acc:.4f}.pt'))
                torch.save(neuron_index, os.path.join(self.res_save_dir, f'models/{max_sparsity:.2f}/last_neuron_index.pt'))
                
                if avg_val_acc > best_avg_val_acc:
                    best_avg_val_acc = avg_val_acc
                    if len(glob.glob(os.path.join(self.res_save_dir, f'models/{max_sparsity:.2f}/main_best_*.pt'))) > 0:
                        os.remove(glob.glob(os.path.join(self.res_save_dir, f'models/{max_sparsity:.2f}/main_best_*.pt'))[0])
                    self.models['kb'].save_model(os.path.join(self.res_save_dir, f'models/{max_sparsity:.2f}/main_best_acc={best_avg_val_acc:.4f}.pt'))
                    torch.save(neuron_index, os.path.join(self.res_save_dir, f'models/{max_sparsity:.2f}/best_neuron_index.pt'))
                
            tb_writer.add_scalars(f'losses', dict(task=task_loss, sparse=sparse_loss, neuron_index=neuron_index_loss), iter_index)
            pbar.set_description(f'loss: {total_loss:.6f} (task: {task_loss:.3f}, sparse: {sparse_loss:.3f}, neuron_index: {neuron_index_loss:.3f})')
            if (iter_index + 1) >= hyps['val_freq']:
                tb_writer.add_scalar(f'accs/val_acc', avg_val_acc, iter_index)
                pbar.set_description(f'loss: {total_loss:.6f} (task: {task_loss:.3f}, sparse: {sparse_loss:.3f}, neuron_index: {neuron_index_loss:.3f}), val_acc: {avg_val_acc:.4f}')
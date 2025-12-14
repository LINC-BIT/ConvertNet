from typing import Any, Dict
from schema import Schema, Or
from data import Scenario, MergedDataset
from methods.base.alg import BaseAlg
from data import build_dataloader
import torch.optim
import tqdm
import numpy as np
import random
from utils.dl.common.env import create_tbwriter
import os
from copy import deepcopy
import glob
from torch import nn
from .model import TrainWithFBSModel
from utils.common.log import logger
from .lib import add_FBS, get_importance_values, set_sparsity, get_l1_reg_in_model, clear_cache, get_fbs_params, get_fbs_module_names, bn_cal
from torch.cuda.amp import autocast, GradScaler


class TrainWithFBSAlg(BaseAlg):
    def get_required_models_schema(self) -> Schema:
        return Schema({
            'main': TrainWithFBSModel
        })
        
    def get_required_hyp_schema(self) -> Schema:
        from schema import Optional
        return Schema({
            'launch_tbboard': bool,
            
            'example_sample': object,
            
            'r': int,
            'ignore_layers': list,
            'max_sparsity_list': object,
            'min_sparsity': float,
            'sparsity_loss_alpha': float,
            'bn_cal_num_iters': object,
            
            'train_batch_size': int,
            'val_batch_size': int,
            'num_workers': int,            
            'optimizer': str,
            'optimizer_args': dict,
            'scheduler': str,
            'scheduler_args': dict,
            'num_iters': int,
            'val_freq': int,
            
            'fp16': bool
        })

    def run(self, scenario: Scenario, hyps: Dict) -> Dict[str, Any]:
                
        super().run(scenario, hyps)
        
        assert isinstance(self.models['main'], TrainWithFBSModel) # for auto completion
        
        if hyps['fp16']:
            # self.models['main'].model = self.models['main'].model.half()
            scaler = GradScaler()
            self.models['main'].fp16 = True
        
        self.models['main'].model = add_FBS(self.models['main'].model, hyps['max_sparsity_list'][-1], 
                                            hyps['r'], hyps['ignore_layers'], True, 
                                            hyps['example_sample'])

        device = self.models['main'].device
        
        offline_datasets = scenario.get_offline_datasets()
        train_dataset = MergedDataset([d['train'] for d in offline_datasets.values()])
        val_dataset = MergedDataset([d['val'] for d in offline_datasets.values()])
        train_loader = iter(build_dataloader(train_dataset, hyps['train_batch_size'], hyps['num_workers'],
                                        True, None))
        val_loader = build_dataloader(val_dataset, hyps['val_batch_size'], hyps['num_workers'],
                                      False, False)
        
        # logger.info(f'init model acc: {self.models["main"].get_accuracy(val_loader):.4f}')
        
        for max_sparsity in hyps['max_sparsity_list']:
            hyps['max_sparsity'] = max_sparsity
            
            optimizer = torch.optim.__dict__[hyps['optimizer']]([
                {'params': self.models['main'].model.parameters(), **hyps['optimizer_args']}
            ])
            scheduler = torch.optim.lr_scheduler.__dict__[hyps['scheduler']](optimizer, **hyps['scheduler_args'])
            tb_writer = create_tbwriter(os.path.join(self.res_save_dir, f'{max_sparsity:.2f}/tb_log'), launch_tbboard=hyps['launch_tbboard'])
            pbar = tqdm.tqdm(range(hyps['num_iters']), dynamic_ncols=True)
            
            best_avg_val_acc = 0.
            
            for iter_index in pbar:
                
                self.models['main'].to_train_mode()
                
                if iter_index % 4 == 0:
                    cur_sparsity = hyps['min_sparsity']
                elif 1 <= iter_index % 4 <= 2:
                    cur_sparsity = random.random() * (hyps['max_sparsity'] - hyps['min_sparsity']) + hyps['min_sparsity']
                elif iter_index % 4 == 3:
                    cur_sparsity = hyps['max_sparsity']
                
                set_sparsity(self.models['main'].model, cur_sparsity)
                
                x, y = next(train_loader)
                if isinstance(x, dict):
                    for k, v in x.items():
                        if isinstance(v, torch.Tensor):
                            x[k] = v.to(device)
                    y = y.to(device)
                else:
                    x, y = x.to(device), y.to(device)
                    
                if hyps['fp16']:
                    with autocast(enabled=True, dtype=torch.float16):
                        task_loss = self.models['main'].forward_to_get_task_loss(x, y)
                        sparse_loss = hyps['sparsity_loss_alpha'] * get_l1_reg_in_model(self.models['main'].model)
                        
                        total_loss = task_loss + sparse_loss
                        
                    optimizer.zero_grad()
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scheduler.step()
                    scaler.update()
                    
                else:

                    task_loss = self.models['main'].forward_to_get_task_loss(x, y)
                    sparse_loss = hyps['sparsity_loss_alpha'] * get_l1_reg_in_model(self.models['main'].model)
                    
                    total_loss = task_loss + sparse_loss
                    
                    optimizer.zero_grad()
                    total_loss.backward()
                    
                    optimizer.step()
                    scheduler.step()
                
                if (iter_index + 1) % 10 == 0:
                    importance_values = get_importance_values(self.models['main'].model)
                    for k, v in importance_values.items():
                        tb_writer.add_histogram(f'importance/{k}', v, iter_index)
                
                clear_cache(self.models['main'].model)
                
                if (iter_index + 1) % hyps['val_freq'] == 0:
                    
                    avg_val_acc = 0.
                    val_accs = {}
                    
                    cur_model = self.models['main'].model
                    
                    for sparsity in tqdm.tqdm(np.linspace(hyps['min_sparsity'], hyps['max_sparsity'], 4), 
                                            desc='val...', dynamic_ncols=True, leave=False):
                        
                        model_for_test = deepcopy(cur_model)
                        val_acc = 0.
                        
                        self.models['main'].model = model_for_test
                        self.models['main'].to_eval_mode()
                        set_sparsity(self.models['main'].model, sparsity)
                        
                        self.bn_cal(self.models['main'], train_loader, hyps['bn_cal_num_iters'], device)
                        val_acc = self.models['main'].get_accuracy(val_loader)
                        
                        avg_val_acc += val_acc
                        val_accs[f'{sparsity:.2f}'] = val_acc
                    
                    self.models['main'].model = cur_model
                    avg_val_acc /= 4
                    tb_writer.add_scalars(f'accs/val_accs', val_accs, iter_index)
                    
                    if len(glob.glob(os.path.join(self.res_save_dir, f'models/{max_sparsity:.2f}/main_last_*.pt'))) > 0:
                        os.remove(glob.glob(os.path.join(self.res_save_dir, f'models/{max_sparsity:.2f}/main_last_*.pt'))[0])
                    self.models['main'].save_model(os.path.join(self.res_save_dir, f'models/{max_sparsity:.2f}/main_last_acc={avg_val_acc:.4f}.pt'))
                    
                    if avg_val_acc > best_avg_val_acc:
                        best_avg_val_acc = avg_val_acc
                        if len(glob.glob(os.path.join(self.res_save_dir, f'models/{max_sparsity:.2f}/main_best_*.pt'))) > 0:
                            os.remove(glob.glob(os.path.join(self.res_save_dir, f'models/{max_sparsity:.2f}/main_best_*.pt'))[0])
                        self.models['main'].save_model(os.path.join(self.res_save_dir, f'models/{max_sparsity:.2f}/main_best_acc={best_avg_val_acc:.4f}.pt'))
                    
                tb_writer.add_scalars(f'losses', dict(task=task_loss, sparse=sparse_loss), iter_index)
                pbar.set_description(f'loss: {total_loss:.6f} (task: {task_loss:.6f}, sparse: {sparse_loss:.6f})')
                if (iter_index + 1) >= hyps['val_freq']:
                    tb_writer.add_scalar(f'accs/val_acc', avg_val_acc, iter_index)
                    pbar.set_description(f'loss: {total_loss:.6f} (task: {task_loss:.6f}, sparse: {sparse_loss:.6f}), val_acc: {avg_val_acc:.4f}')
    
    @torch.no_grad()
    def bn_cal(self, model: TrainWithFBSModel, train_loader, num_iters, device):
        return bn_cal(model.model, train_loader, num_iters, device)
    
    
class BNCalAlg(BaseAlg):
    def get_required_models_schema(self) -> Schema:
        return Schema({
            'main': TrainWithFBSModel
        })
        
    def get_required_hyp_schema(self) -> Schema:
        from schema import Optional
        return Schema({
            'launch_tbboard': bool,
            
            'example_sample': object,
            
            'optional_sparsity_list': object,
            'bn_cal_num_iters': object,
            
            'train_batch_size': int,
            'val_batch_size': int,
            'num_workers': int
        })

    def run(self, scenario: Scenario, hyps: Dict) -> Dict[str, Any]:
                
        super().run(scenario, hyps)
        
        assert isinstance(self.models['main'], TrainWithFBSModel) # for auto completion
        
        device = self.models['main'].device
        
        offline_datasets = scenario.get_offline_datasets()
        train_dataset = MergedDataset([d['train'] for d in offline_datasets.values()])
        val_dataset = MergedDataset([d['val'] for d in offline_datasets.values()])
        train_loader = iter(build_dataloader(train_dataset, hyps['train_batch_size'], hyps['num_workers'],
                                        True, None))
        val_loader = build_dataloader(val_dataset, hyps['val_batch_size'], hyps['num_workers'],
                                      False, False)
        
        avg_val_acc = 0.
        val_accs = {}
        res_bn_stats = {}
        
        cur_model = self.models['main'].model
        
        logger.info(f'optional sparsity list: {hyps["optional_sparsity_list"]}')

        for sparsity in tqdm.tqdm(hyps['optional_sparsity_list'], 
                                  desc='val...', dynamic_ncols=True, leave=False):
            
            model_for_test = deepcopy(cur_model)
            val_acc = 0.
            
            self.models['main'].model = model_for_test
            self.models['main'].to_eval_mode()
            set_sparsity(self.models['main'].model, sparsity)
            
            res_bn_stats[f'{sparsity}:.2f'] = self.bn_cal(self.models['main'], train_loader, hyps['bn_cal_num_iters'], device)
            val_acc = self.models['main'].get_accuracy(val_loader)
            
            avg_val_acc += val_acc
            val_accs[f'{sparsity:.2f}'] = val_acc
            
            logger.info(f'sparsity: {sparsity:.2f}, val_acc: {val_acc:.4f}')
            
        self.models['main'].model = cur_model
        self.models['main'].models_dict['bn_stats'] = res_bn_stats
        
        base_filename = os.path.basename(self.models['main'].models_dict_path)
        self.models['main'].save_model(os.path.join(self.res_save_dir, f'models/{base_filename}.with_bn_stats'))
        
        avg_val_acc /= len(hyps['optional_sparsity_list'])
        logger.info(f'avg_val_acc: {avg_val_acc:.4f}')
        
    @torch.no_grad()
    def bn_cal(self, model: TrainWithFBSModel, train_loader, num_iters, device):
        return bn_cal(model.model, train_loader, num_iters, device)
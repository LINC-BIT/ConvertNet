from typing import Any, Dict, List
from schema import Schema, Optional
from data import Scenario, MergedDataset
from methods.base.alg import BaseAlg
from methods.base.model import BaseModel
from data import build_dataloader
import torch.optim
import tqdm
import os
import time
from abc import abstractmethod
import matplotlib.pyplot as plt
import random
from ..feature_alignment.model import FeatureAlignmentModel
from data import ABDataset, get_dataset
from torch.cuda.amp import autocast, GradScaler
from utils.common.log import logger
from utils.dl.common.model import get_num_params
import numpy as np


class FeatureAlignmentWithMultipleRunningScalingAlg(BaseAlg):
    def get_required_models_schema(self) -> Schema:
        return Schema({
            'main': FeatureAlignmentModel
        })
        
    def get_required_hyp_schema(self) -> Schema:
        return Schema({
            'train_batch_size': int,
            'val_batch_size': int,
            'num_workers': int,
            'optimizer': str,
            'optimizer_args': dict,
            'scheduler': str,
            'scheduler_args': dict,
            'num_iters': int,
            'val_freq': int,
            'feat_align_loss_weight': float,
            'freeze_bn': bool,
            
            'knowledge_base': object,
            'attention_values_of_layers': dict,
            'unpruned_indexes_of_layers': dict,
            'scaling_points': list,
            
            'qkv_layers_name': list,
            'proj_layers_name': list,
            'ff1_layers_name': list,
            'ff2_layers_name': list,
            'only_add_fbs_in_qkv': bool,
            
            'random_sim_policy': object,
            'auged_source_dataset_name': object,
            
            'fp16': bool,
            Optional('collate_fn', default=None): object,
            
            'use_train_data_for_test': bool,
            
            'for_profiling': bool
        })
        
    def run(self, scenario: Scenario, hyps: Dict) -> Dict[str, Any]:
        super().run(scenario, hyps)
        
        assert isinstance(self.models['main'], FeatureAlignmentModel) # for auto completion
        
        if hyps['random_sim_policy'] is not None:
            offline_datasets = scenario.get_offline_datasets(use_before_res=True)
            source_datasets = offline_datasets[hyps['auged_source_dataset_name']]
            sim_target_datasets = {}
            
            # augment dataset
            from utils.dl.auto_augment import generate_sim_datasets_with_same_aug
            aug_datasets = generate_sim_datasets_with_same_aug([source_datasets['train'], source_datasets['val']], 
                                                                    hyps['random_sim_policy'])
            sim_target_datasets['train'] = aug_datasets[0]
            sim_target_datasets['val'] = aug_datasets[1]
            
            source_train_loader = iter(build_dataloader(source_datasets['train'], hyps['train_batch_size'], hyps['num_workers'],
                                            True, None, collate_fn=hyps['collate_fn']))
            train_loader = iter(build_dataloader(sim_target_datasets['train'], hyps['train_batch_size'], hyps['num_workers'],
                                            True, None, collate_fn=hyps['collate_fn']))
            test_dataset = sim_target_datasets['val']
            
        else:
            cur_domain_name = scenario.target_domains_order[scenario.cur_domain_index]
            datasets_for_training = scenario.get_online_cur_domain_datasets_for_training()
            train_dataset = datasets_for_training[cur_domain_name]['train']
            val_dataset = datasets_for_training[cur_domain_name]['val']
            datasets_for_inference = scenario.get_online_cur_domain_datasets_for_inference()
            test_dataset = datasets_for_inference
            
            if hyps['use_train_data_for_test']:
                test_dataset = train_dataset
                logger.info('use train data for test')
            
            train_loader = iter(build_dataloader(train_dataset, hyps['train_batch_size'], hyps['num_workers'],
                                True, None, collate_fn=hyps['collate_fn']))
            
            source_datasets = [d['train'] for n, d in datasets_for_training.items() if n != cur_domain_name]
            source_dataset = MergedDataset(source_datasets)
            source_train_loader = iter(build_dataloader(source_dataset, hyps['train_batch_size'], hyps['num_workers'],
                                True, None, collate_fn=hyps['collate_fn']))
        
        logger.debug(f'init model: {self.models["main"].model}')
        
        device = self.models['main'].device
        trained_params = self.models['main'].get_trained_params()
        optimizer = torch.optim.__dict__[hyps['optimizer']](trained_params, **hyps['optimizer_args'])
        if hyps['scheduler'] != '':
            scheduler = torch.optim.lr_scheduler.__dict__[hyps['scheduler']](optimizer, **hyps['scheduler_args'])
        else:
            scheduler = None
        
        pbar = tqdm.tqdm(range(hyps['num_iters']), dynamic_ncols=True, desc='feature alignment...')
        task_losses, mmd_losses, total_losses = [], [], []
        accs = []
        times = []
        total_train_time = 0.
        
        if hyps['fp16']:
            # self.models['main'].model = self.models['main'].model.half()
            scaler = GradScaler()
            self.models['main'].fp16 = True
        
        # feature_hook = self.models['main'].get_feature_hook()
        
        scaling_iters, scaling_actions = [a[0] for a in hyps['scaling_points']], [a[1] for a in hyps['scaling_points']]
        # cur_n_for_retraining = 0
        
        if hyps['for_profiling']:
            hyps['val_freq'] = hyps['num_iters'] - 1
        
        for iter_index in pbar:
            
            if iter_index % hyps['val_freq'] == 0:
                from data import split_dataset
                cur_test_batch_dataset = split_dataset(test_dataset, hyps['val_batch_size'], iter_index)[0]
                cur_test_batch_dataloader = build_dataloader(cur_test_batch_dataset, hyps['train_batch_size'], hyps['num_workers'], False, False, collate_fn=hyps['collate_fn'])
                # print(next(iter(cur_test_batch_dataloader)))
                cur_acc = self.models['main'].get_accuracy(cur_test_batch_dataloader)
                accs += [{
                    'iter': iter_index,
                    'acc': float(cur_acc)
                }]
                
            if iter_index in scaling_iters:
                scaled_blocks_sparsity_index = scaling_iters.index(iter_index)
                scaled_action = scaling_actions[scaled_blocks_sparsity_index]
                
                # from .dyna_nest_transformer import retrain_first_n_nested_components, preserve_first_n_nested_components
                from .dyna_nest_transformer import retrain_first_n_nested_components, retrain_components, \
                    preserve_first_n_nested_components, calculate_blocks_importance
                
                if 's' == scaled_action[0]:
                    num_params_before = get_num_params(self.models['main'].model) // 1024**2
                    preserve_first_n_nested_components(self.models['main'].model, scaled_action[1])
                    trained_params = retrain_first_n_nested_components(self.models['main'].model, scaled_action[1])
                    num_params_after = get_num_params(self.models['main'].model) // 1024**2
                    logger.info(f'preserve first {scaled_action[1]} nested components ({num_params_before}M -> {num_params_after}M)')
                    logger.info(f'retrain components: {scaled_action[1]}')
                    logger.info(f'number of retrained params: {sum([p.numel() for p in trained_params]) // 1024**2}M')
                    # re-init optimizers...
                    optimizer = torch.optim.__dict__[hyps['optimizer']](trained_params, **hyps['optimizer_args'])
                    if hyps['scheduler'] != '':
                        scheduler = torch.optim.lr_scheduler.__dict__[hyps['scheduler']](optimizer, 
                                                                                        last_epoch=iter_index, **hyps['scheduler_args'])
                    else:
                        scheduler = None
                    
                elif 'r' in scaled_action[0]:
                    if isinstance(scaled_action[1][1], list):
                        trained_params = retrain_components(self.models['main'].model, scaled_action[1][0], scaled_action[1][1], 
                                                            self.models['main'].get_params_names_of_each_block())
                    else: # smart
                        logger.info(f'smartly retrain blocks {trained_blocks_index_by_importance}')
                        trained_params = retrain_components(self.models['main'].model, scaled_action[1][0], trained_blocks_index_by_importance, 
                                                            self.models['main'].get_params_names_of_each_block())
                        
                    logger.info(f'number of retrained params: {sum([p.numel() for p in trained_params]) // 1024**2}M')
                    logger.info(f'retrain components: {scaled_action[1]}')
                    
                    # re-init optimizers...
                    optimizer = torch.optim.__dict__[hyps['optimizer']](trained_params, **hyps['optimizer_args'])
                    if hyps['scheduler'] != '':
                        scheduler = torch.optim.lr_scheduler.__dict__[hyps['scheduler']](optimizer, 
                                                                                        last_epoch=iter_index, **hyps['scheduler_args'])
                    else:
                        scheduler = None
                
                # test accuracy of new model
                from data import split_dataset
                cur_test_batch_dataset = split_dataset(test_dataset, hyps['val_batch_size'], iter_index)[0]
                cur_test_batch_dataloader = build_dataloader(cur_test_batch_dataset, hyps['train_batch_size'], hyps['num_workers'], False, False, collate_fn=hyps['collate_fn'])
                cur_acc = self.models['main'].get_accuracy(cur_test_batch_dataloader)
                accs += [{
                    'iter': iter_index,
                    'acc': float(cur_acc)
                }]
                
                if hyps['for_profiling']:
                    torch.cuda.empty_cache()
                
            feature_hook = self.models['main'].get_feature_hook()
            
            cur_start_time = time.time()
            
            if hyps['freeze_bn']:
                self.models['main'].to_eval_mode()
            else:
                self.models['main'].to_train_mode()
            
            x, _ = next(train_loader)
            
            if isinstance(x, dict):
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(device)
            else:
                x = x.to(device)
            
            source_x, source_y = next(source_train_loader)
            
            if isinstance(source_x, dict):
                for k, v in source_x.items():
                    if isinstance(v, torch.Tensor):
                        source_x[k] = v.to(device)
                source_y = source_y.to(device)
            else:
                source_x, source_y = source_x.to(device), source_y.to(device)
            
            if hyps['fp16']:
                with autocast(enabled=True, dtype=torch.float16):
                    task_loss = self.models['main'].forward_to_get_task_loss(source_x, source_y)
                    source_features = feature_hook.input
                
                    self.models['main'].infer(x)
                    target_features = feature_hook.input

                    try:
                        mmd_loss = hyps['feat_align_loss_weight'] * self.models['main'].get_mmd_loss(source_features, target_features)
                    except:
                        mmd_loss = torch.FloatTensor([0.]).cuda()[0]
                        logger.info(f'iter {iter_index}, mmd_loss compute failed! (I don\'t know why!)')
                        
                    loss = task_loss + mmd_loss
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                if (iter_index + 1) in scaling_iters and scaling_actions[scaling_iters.index(iter_index + 1)][0] == 'r' and \
                    isinstance(scaling_actions[scaling_iters.index(iter_index + 1)][1][1], int):
                        
                    blocks_importance = calculate_blocks_importance(self.models['main'].model, 
                                                                    self.models['main'].get_params_names_of_each_block())
                    trained_blocks_index_by_importance = np.array(blocks_importance).argsort()[::-1][0: scaling_actions[scaling_iters.index(iter_index + 1)][1][1]]
                
                scaler.step(optimizer)
                if scheduler is not None:
                    scheduler.step()
                scaler.update()
                    
            else:
            
                task_loss = self.models['main'].forward_to_get_task_loss(source_x, source_y)
                source_features = feature_hook.input
                
                self.models['main'].infer(x)
                target_features = feature_hook.input
                
                mmd_loss = hyps['feat_align_loss_weight'] * self.models['main'].get_mmd_loss(source_features, target_features)
                
                loss = task_loss + mmd_loss
                
                optimizer.zero_grad()
                loss.backward() 
                
                if (iter_index + 1) in scaling_iters and scaling_actions[scaling_iters.index(iter_index + 1)][0] == 'r' and \
                    isinstance(scaling_actions[scaling_iters.index(iter_index + 1)][1][1], int):
                        
                    blocks_importance = calculate_blocks_importance(self.models['main'].model, 
                                                                    self.models['main'].get_params_names_of_each_block())
                    trained_blocks_index_by_importance = np.array(blocks_importance).argsort()[::-1][0: scaling_actions[scaling_iters.index(iter_index + 1)][1][1]]

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            
            pbar.set_description(f'feature alignment... | cur_acc: {cur_acc:.4f}, task_loss: {task_loss:.6f}, mmd_loss: {mmd_loss:.6f}')
            task_losses += [float(task_loss.cpu().item())]
            mmd_losses += [float(mmd_loss.cpu().item())]
            total_losses += [float(task_loss + mmd_loss)]
            
            times += [time.time() - cur_start_time]
            total_train_time += times[-1]
            
            feature_hook.remove()
        
        # feature_hook.remove()
        
        time_usage = total_train_time
        
        # cur_test_batch_dataset = split_dataset(test_dataset, hyps['train_batch_size'], iter_index + 1)[0]
        # cur_test_batch_dataloader = build_dataloader(cur_test_batch_dataset, len(cur_test_batch_dataset), hyps['num_workers'], False, False)
        cur_test_batch_dataset = split_dataset(test_dataset, hyps['val_batch_size'], iter_index + 1)[0]
        cur_test_batch_dataloader = build_dataloader(cur_test_batch_dataset, hyps['train_batch_size'], hyps['num_workers'], False, False, collate_fn=hyps['collate_fn'])
        cur_acc = self.models['main'].get_accuracy(cur_test_batch_dataloader)
        accs += [{
            'iter': iter_index + 1,
            'acc': float(cur_acc)
        }]
        
        plt.plot(task_losses, label='task')
        plt.plot(mmd_losses, label='mmd')
        plt.plot(total_losses, label='total')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(os.path.join(self.res_save_dir, 'loss.png'))
        plt.clf()
        
        plt.plot([int(i['iter']) for i in accs], [float(i['acc']) for i in accs])
        plt.xlabel('iteration')
        plt.ylabel('acc')
        plt.savefig(os.path.join(self.res_save_dir, 'acc.png'))
        plt.clf()
        
        retraining_info = {
            'accs': accs,
            'time': time_usage,
            'times': times,
            'total_losses': total_losses
        }
        torch.save(retraining_info, os.path.join(self.res_save_dir, 'retraining_info.pth'))
        return retraining_info, self.models
        
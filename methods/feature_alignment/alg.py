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
from .model import FeatureAlignmentModel
from data import ABDataset, get_dataset
from torch.cuda.amp import autocast, GradScaler
from utils.common.log import logger


class FeatureAlignmentAlg(BaseAlg):
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
            
            'random_sim_policy': object,
            'auged_source_dataset_name': object,
            
            'fp16': bool,
            Optional('collate_fn', default=None): object
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
            
            train_loader = iter(build_dataloader(train_dataset, hyps['train_batch_size'], hyps['num_workers'],
                                True, None, collate_fn=hyps['collate_fn']))
            
            source_datasets = [d['train'] for n, d in datasets_for_training.items() if n != cur_domain_name]
            source_dataset = MergedDataset(source_datasets)
            source_train_loader = iter(build_dataloader(source_dataset, hyps['train_batch_size'], hyps['num_workers'],
                                True, None, collate_fn=hyps['collate_fn']))
        
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
        
        total_train_time = 0.
        
        if hyps['fp16']:
            # self.models['main'].model = self.models['main'].model.half()
            scaler = GradScaler()
            self.models['main'].fp16 = True
        
        # feature_hook = self.models['main'].get_feature_hook()
        
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
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            
            pbar.set_description(f'feature alignment... | cur_acc: {cur_acc:.4f}, task_loss: {task_loss:.6f}, mmd_loss: {mmd_loss:.6f}')
            task_losses += [float(task_loss.cpu().item())]
            mmd_losses += [float(mmd_loss.cpu().item())]
            total_losses += [float(task_loss + mmd_loss)]
            
            total_train_time += time.time() - cur_start_time
            
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
            'total_losses': total_losses,
            'task_losses': task_losses,
            'mmd_losses': mmd_losses
        }
        torch.save(retraining_info, os.path.join(self.res_save_dir, 'retraining_info.pth'))
        return retraining_info, self.models
        
    
    
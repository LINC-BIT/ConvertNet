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
from .model import LoRAFineTuningModel
from torch.cuda.amp import autocast, GradScaler
from utils.common.log import logger


class LoRATuningAlg(BaseAlg):
    def get_required_models_schema(self) -> Schema:
        return Schema({
            'main': LoRAFineTuningModel
        })
        
    def get_required_hyp_schema(self) -> Schema:
        from schema import Optional
        return Schema({
            'launch_tbboard': bool,
            
            'example_sample': object,
            
            'qkv_layers_name': object,
            'lora_r': 8,
            
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
        
        assert isinstance(self.models['main'], LoRAFineTuningModel) # for auto completion

        device = self.models['main'].device
        
        from .lib import add_lora_ab_to_model, absorb_lora, train_only_lora
        qkv_layers = []
        for l in hyps['qkv_layers_name']:
            qkv_layers += l
        add_lora_ab_to_model(self.models['main'].model, qkv_layers, isinstance(hyps['qkv_layers_name'][0], str), 
                             hyps['lora_r'])
        logger.debug(f'LoRA added to model: {self.models["main"].model}')
        
        if hyps['fp16']:
            # self.models['main'].model = self.models['main'].model.half()
            scaler = GradScaler()
            self.models['main'].fp16 = True
        
        offline_datasets = scenario.get_offline_datasets()
        train_dataset = MergedDataset([d['train'] for d in offline_datasets.values()])
        val_dataset = MergedDataset([d['val'] for d in offline_datasets.values()])
        train_loader = iter(build_dataloader(train_dataset, hyps['train_batch_size'], hyps['num_workers'],
                                        True, None))
        val_loader = build_dataloader(val_dataset, hyps['val_batch_size'], hyps['num_workers'],
                                      False, False)
        
        lora_params = train_only_lora(self.models['main'].model)
        head_params = self.models['main'].get_head_params()
        for p in head_params:
            p.requires_grad = True
        
        optimizer = torch.optim.__dict__[hyps['optimizer']]([
            {'params': lora_params + list(head_params), **hyps['optimizer_args']}
        ])
        scheduler = torch.optim.lr_scheduler.__dict__[hyps['scheduler']](optimizer, **hyps['scheduler_args'])
        tb_writer = create_tbwriter(os.path.join(self.res_save_dir, 'tb_log'), launch_tbboard=hyps['launch_tbboard'])
        pbar = tqdm.tqdm(range(hyps['num_iters']), dynamic_ncols=True)
        
        best_avg_val_acc = 0.
        
        for iter_index in pbar:
            
            self.models['main'].to_train_mode()
            
            x, y = next(train_loader)
            if isinstance(x, dict):
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(device)
                        # if hyps['fp16']:
                        #     x[k] = x[k].half()
                y = y.to(device)
            else:
                x, y = x.to(device), y.to(device)
                # if hyps['fp16']:
                #     x = x.half()
            
            if hyps['fp16']:
                with autocast(enabled=True, dtype=torch.float16):
                    task_loss = self.models['main'].forward_to_get_task_loss(x, y)
                
                total_loss = task_loss
                
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
            else:
                task_loss = self.models['main'].forward_to_get_task_loss(x, y)
                
                total_loss = task_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()
            
            if (iter_index + 1) % hyps['val_freq'] == 0:
                
                cur_model = self.models['main'].model
                model_for_test = deepcopy(cur_model)
                val_acc = 0.
                
                self.models['main'].model = model_for_test
                self.models['main'].to_eval_mode()
                val_acc = self.models['main'].get_accuracy(val_loader)
                
                # self.models['main'].model = cur_model
                
                model_for_save = absorb_lora(self.models['main'].model)
                self.models['main'].model = model_for_save
                
                if len(glob.glob(os.path.join(self.res_save_dir, 'models/main_last_*.pt'))) > 0:
                    os.remove(glob.glob(os.path.join(self.res_save_dir, 'models/main_last_*.pt'))[0])
                self.models['main'].save_model(os.path.join(self.res_save_dir, f'models/main_last_acc={val_acc:.4f}.pt'))
                
                if val_acc > best_avg_val_acc:
                    best_avg_val_acc = val_acc
                    if len(glob.glob(os.path.join(self.res_save_dir, 'models/main_best_*.pt'))) > 0:
                        os.remove(glob.glob(os.path.join(self.res_save_dir, 'models/main_best_*.pt'))[0])
                    self.models['main'].save_model(os.path.join(self.res_save_dir, f'models/main_best_acc={best_avg_val_acc:.4f}.pt'))
                
                self.models['main'].model = cur_model
                
                torch.cuda.empty_cache()
                
            tb_writer.add_scalars(f'losses', dict(task=task_loss), iter_index)
            pbar.set_description(f'loss: {total_loss:.6f}')
            if (iter_index + 1) >= hyps['val_freq']:
                tb_writer.add_scalar(f'accs/val_acc', val_acc, iter_index)
                pbar.set_description(f'loss: {total_loss:.6f}, val_acc: {val_acc:.4f}')
            
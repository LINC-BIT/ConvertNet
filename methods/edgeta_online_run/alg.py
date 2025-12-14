from typing import Any, Dict
from schema import Schema, Or
from data import Scenario, MergedDataset
from methods.base.alg import BaseAlg
from data import build_dataloader, get_dataset, ABDataset
import torch.optim
import tqdm
import numpy as np
import random
from utils.dl.common.env import create_tbwriter
import os
from copy import deepcopy
import glob
import time
from torch import nn
import copy
import shutil
from utils.dl.common.model import get_model_size, get_module
from .model import EdgeTAOnlineRunModel
from utils.common.log import logger
from utils.common.others import get_tmp_filepath
from ..train_with_fbs.lib import set_sparsity, switch_bn_stats
from utils.dl.auto_augment import gen_random_sim_policy, generate_sim_datasets_with_same_aug
import matplotlib.pyplot as plt
    
    
class EdgeTAOnlineRunAlg(BaseAlg):
    def get_required_models_schema(self) -> Schema:
        return Schema({
            'main': EdgeTAOnlineRunModel
        })
        
    def get_required_hyp_schema(self) -> Schema:
        return Schema({
            'sparsity': float,
            'obtain_features_num_iters': int,
            'obtain_larget_model_target_loss_num_iters': int,
            
            'qkv_layers_name': list,
            'proj_layers_name': list,
            'ff1_layers_name': list,
            'ff2_layers_name': list,

            'retraining_alg_cls': object,
            'retraining_model_cls': object,
            'retraining_hyps': object,
            
            'retraining_window': int,
            
            'knowledge_base_to_fm_alpha': float,
            'fm_to_knowledge_base_alpha': float,
            
            'opt_strategy': object,
            'up_bound_pred_acc': float
        })

    def run(self, scenario: Scenario, hyps):
        """
        Return the inputs and output of our scaling law
        """
        
        super().run(scenario, hyps)
        
        assert isinstance(self.models['main'], EdgeTAOnlineRunModel) # for auto completion
        
        cur_domain_name = scenario.target_domains_order[scenario.cur_domain_index]
        datasets_for_training = scenario.get_online_cur_domain_datasets_for_training()
        train_dataset = datasets_for_training[cur_domain_name]['train']
        # val_dataset = datasets_for_training[cur_domain_name]['val']
        datasets_for_inference = scenario.get_online_cur_domain_datasets_for_inference()
        test_dataset = datasets_for_inference
        
        # print(hyps['retraining_hyps'])
        # collate_fn = getattr(hyps['retraining_hyps'], 'collate_fn', None)
        # print(collate_fn)
        # exit()
        
        if 'collate_fn' in hyps['retraining_hyps'].keys():
            collate_fn = hyps['retraining_hyps']['collate_fn']
            print('use customized collate_fn')
        else:
            collate_fn = None
        
        target_train_dataloader = iter(build_dataloader(train_dataset, hyps['retraining_hyps']['train_batch_size'], hyps['retraining_hyps']['num_workers'],
                            True, None, collate_fn=collate_fn))
        
        source_datasets = [d['train'] for n, d in datasets_for_training.items() if n != cur_domain_name]
        source_dataset = MergedDataset(source_datasets)
        source_train_dataloader = iter(build_dataloader(source_dataset, hyps['retraining_hyps']['train_batch_size'], hyps['retraining_hyps']['num_workers'],
                            True, None, collate_fn=collate_fn))
        
        # set sparsity
        set_sparsity(self.models['main'].knowledge_base, 0.)
        switch_bn_stats(self.models['main'].knowledge_base, self.models['main'].bn_stats)
        self.models['main'].to_eval_mode()
        
        # obtain source/target features and calculate distance bewteen them
        # given_target_samples = next(target_train_dataloader)[0]
        # small_model, _, _ = self.models['main'].generate_small_model(given_target_samples)
        
        device = self.models['main'].device
        self.models['main'].to_eval_mode()
        
        def _to(sample):
            if isinstance(sample, dict):
                for k, v in sample.items():
                    if isinstance(v, torch.Tensor):
                        sample[k] = v.to(device)
                return sample
            else:
                return sample.to(device)
            
        def _ith_sample(samples, i):
            i = int(i)
            if isinstance(samples, dict):
                return {k: v[i: i+1] if isinstance(v, (torch.Tensor, list)) else v for k, v in samples.items()}
            else:
                return samples[i: i+1]
        
        if hyps['opt_strategy'] == 'heuristic' or hyps['opt_strategy'] == 'heuristic_inverse':
            target_features = []
            hook = self.models['main'].get_feature_hook()
            for _ in range(hyps['obtain_features_num_iters']):
                target_samples = _to(next(target_train_dataloader)[0])
                self.models['main'].infer(target_samples)
                target_features += [hook.input.detach()]
            hook.remove()
            target_features = torch.cat(target_features)
            
            source_features = []
            hook = self.models['main'].get_feature_hook()
            for _ in range(hyps['obtain_features_num_iters']):
                source_samples = _to(next(source_train_dataloader)[0])
                self.models['main'].infer(source_samples)
                source_features += [hook.input.detach()]
            hook.remove()
            source_features = torch.cat(source_features)
                
            # small_model_size = get_model_size(small_model, True)
            
            from ..gen_scaling_law_data_points.fid_distance import calculate_frechet_distance
            source_target_dist_distance = calculate_frechet_distance(source_features.cpu(), target_features.cpu())
            # logger.info(f'source_target_dist_distance: {source_target_dist_distance:.4f}')
            
            f1, f2 = source_features.detach().cpu().numpy(), target_features.detach().cpu().numpy()
            mu1, sigma1 = np.mean(f1, axis=0), np.cov(f1, rowvar=False)
            mu2, sigma2 = np.mean(f2, axis=0), np.cov(f2, rowvar=False)
            features_stats = (mu1, mu2, sigma1, sigma2)
            
            # get large_model_loss_in_target_dist
            tmp_large_model_path = get_tmp_filepath()
            # self.models['main'].save_model(tmp_large_model_path)
            torch.save({'main': self.models['main'].knowledge_base}, tmp_large_model_path)
            large_model_model = hyps['retraining_model_cls'](
                name='tmp_model',
                models_dict_path=tmp_large_model_path,
                device=self.models['main'].device
            )
            large_model_model.num_classes = scenario.num_classes
            retraining_alg = hyps['retraining_alg_cls'](
                models={
                    'main': large_model_model
                },
                res_save_dir=os.path.join(self.res_save_dir, f'kb_compute_loss')
            )
            large_model_loss_in_target_dist = retraining_alg.run(scenario, {
                **hyps['retraining_hyps'], 
                'num_iters': hyps['obtain_larget_model_target_loss_num_iters'],
                'optimizer_args': {'lr': 1e-9},
                'freeze_bn': True,
                'random_sim_policy': None,
                'auged_source_dataset_name': ''
            })[0]['total_losses']
            logger.info(f'large_model_loss_in_target_dist: {large_model_loss_in_target_dist}')
            large_model_loss_in_target_dist = sum(large_model_loss_in_target_dist) / len(large_model_loss_in_target_dist)
            shutil.rmtree(os.path.join(self.res_save_dir, f'kb_compute_loss'))
            os.remove(tmp_large_model_path)
        
        
        
        if isinstance(hyps['opt_strategy'], tuple):
            solved_whether_compress_block, solved_num_retraining_iters = hyps['opt_strategy']
            solved_num_retraining_iters = int(solved_num_retraining_iters)
            logger.info(f'solved_whether_compress_block: {solved_whether_compress_block}, solved_num_retraining_iters: {solved_num_retraining_iters}')
            blocks_sparsity = [hyps['sparsity'] if c else 0. for c in solved_whether_compress_block]
            attns_sparsity = ffns_sparsity = blocks_sparsity
            
        elif hyps['opt_strategy'] == 'heuristic' or hyps['opt_strategy'] == 'heuristic_inverse':
            s = time.time()
            # optimization
            edge_scaling_law = self.models['main'].scaling_law
            
            p1, p2 = edge_scaling_law.get_p1_p2(torch.tensor([source_target_dist_distance]).float().cuda(),
                                        torch.tensor([large_model_loss_in_target_dist]).float().cuda(),
                                        *[torch.tensor(f).unsqueeze(0).float().cuda() for f in features_stats])
            
            @torch.no_grad()
            def acc_func(blocks_pruning_strategy, num_iters, use_batch=False):
                if use_batch:
                    num_params_each_block = torch.tensor([[self.models['main'].profiles['num_blocks_params'][bi][int(i)] for bi, i in enumerate(blocks_pruning_strategy_i)] 
                                                        for blocks_pruning_strategy_i in blocks_pruning_strategy], device=device).float()
                    res = edge_scaling_law.scaling_law(
                                            torch.tensor(num_iters, device=device).float(),
                                            torch.tensor([ni * hyps['retraining_hyps']['train_batch_size'] for ni in num_iters], device=device).float(),
                                            num_params_each_block, p1, p2)
                    res[res > hyps['up_bound_pred_acc']] = hyps['up_bound_pred_acc']
                    return res.detach().cpu().numpy().tolist()
                else:
                    num_params_each_block = torch.tensor([[self.models['main'].profiles['num_blocks_params'][bi][int(i)] for bi, i in enumerate(blocks_pruning_strategy)]], device=device).float()
                    res = edge_scaling_law.scaling_law(
                                            torch.tensor([num_iters], device=device).float(),
                                            torch.tensor([num_iters * hyps['retraining_hyps']['train_batch_size']], device=device).float(),
                                            num_params_each_block, p1, p2)[0].item()
                    res = min(res, hyps['up_bound_pred_acc'])
                    return float(res)
            
            from .heuristic_opt import heuristic_solve
            solved_whether_compress_block, solved_num_retraining_iters = heuristic_solve(
                acc_func, self.models['main'].profiles['blocks_retraining_time_per_iteration'],
                10, hyps['retraining_window'], inverse_debug=(hyps['opt_strategy'] == 'heuristic_inverse')
            )
            solved_num_retraining_iters = int(solved_num_retraining_iters)
            logger.info(f'solved_whether_compress_block: {solved_whether_compress_block}, solved_num_retraining_iters: {solved_num_retraining_iters}')
            blocks_sparsity = [hyps['sparsity'] if c else 0. for c in solved_whether_compress_block]
            attns_sparsity = ffns_sparsity = blocks_sparsity
            logger.info(f'solving using {(time.time() - s):.4f}s')
            
        elif hyps['opt_strategy'] == 'compress_all_blocks':
            solved_num_retraining_iters = 450
            blocks_sparsity = attns_sparsity = ffns_sparsity = [hyps['sparsity']] * 12
        elif hyps['opt_strategy'] == 'compress_no_blocks':
            solved_num_retraining_iters = 300
            blocks_sparsity = attns_sparsity = ffns_sparsity = [0.] * 12
        elif hyps['opt_strategy'].startswith('random_'):
            for _ in range(int(hyps['opt_strategy'].split('_')[-1])):
                solved_num_retraining_iters = random.randint(200, 400)
                blocks_sparsity = [0. for _ in range(len(hyps['qkv_layers_name']))]
                pruned_blocks_index = random.choices(list(range(len(hyps['qkv_layers_name']))), 
                                                    k=random.randint(1, len(hyps['qkv_layers_name'])))
                for pi in pruned_blocks_index:
                    blocks_sparsity[pi] = hyps['sparsity']
                attns_sparsity = ffns_sparsity = blocks_sparsity
        elif hyps['opt_strategy'] == 'compress_all_blocks_for_generate_baseline_model':
            solved_num_retraining_iters = 0
            blocks_sparsity = attns_sparsity = ffns_sparsity = [hyps['sparsity']] * 12
        
        from utils.common.others import longest_common_prefix
        for i, layer_name in enumerate(hyps['qkv_layers_name']):
            if isinstance(layer_name, list):
                layer_name = longest_common_prefix(layer_name)
                if layer_name.endswith('.'):
                    layer_name = layer_name[0: -1]
            # print(layer_name)
            set_sparsity(get_module(self.models['main'].knowledge_base, layer_name), attns_sparsity[i])
            
            if i < len(hyps['proj_layers_name']) - 1: # NOTE: to fix a manually created bug
                set_sparsity(get_module(self.models['main'].knowledge_base, hyps['proj_layers_name'][i]), attns_sparsity[i])
        for i, layer_name in enumerate(hyps['ff1_layers_name']):
            if isinstance(layer_name, list):
                layer_name = longest_common_prefix(layer_name)[0: -1]
                # print(layer_name)
                set_sparsity(get_module(self.models['main'].knowledge_base, layer_name), ffns_sparsity[i])
            else:
                set_sparsity(get_module(self.models['main'].knowledge_base, layer_name), ffns_sparsity[i])
            set_sparsity(get_module(self.models['main'].knowledge_base, hyps['ff2_layers_name'][i]), ffns_sparsity[i])
        
        
        # print(next(target_train_dataloader))
        given_target_samples = _to(next(target_train_dataloader)[0])
            
        # small_model, _, _ = self.models['main'].generate_small_model(given_target_samples)
        with torch.no_grad():
            output_entropy = self.models['main'].get_output_entropy(given_target_samples)
            # rep_target_sample = given_target_samples[output_entropy.argmax(): output_entropy.argmax() + 1]
            rep_target_sample = _ith_sample(given_target_samples, output_entropy.argmax())
            # print(rep_target_sample)
            
            s = time.time()
            o1 = self.models['main'].infer(rep_target_sample)
            logger.info(f'infer time: {(time.time() - s):.4f}s')
            
            from .lib_transformer import generate_small_model
            
            s = time.time()
            small_model, unpruned_neurons_idx_of_layers, _ = generate_small_model(self.models['main'].knowledge_base, 
                                                                               hyps['qkv_layers_name'], hyps['proj_layers_name'], 
                                hyps['ff1_layers_name'], hyps['ff2_layers_name'], return_detail=True)
            logger.info(f'model generation time: {(time.time() - s):.4f}s')
            
            large_model = self.models['main'].knowledge_base
            self.models['main'].knowledge_base = small_model
            self.models['main'].to_eval_mode()
            small_model.eval()
            o2 = self.models['main'].infer(rep_target_sample)
            self.models['main'].knowledge_base = large_model
            
            diff = ((o1 - o2) ** 2).sum()
            assert diff < 1e-3, diff
            
        # real run: retraining small model
        tmp_small_model_path = get_tmp_filepath()
        torch.save({'main': small_model}, tmp_small_model_path)
        
        if hyps['opt_strategy'] == 'compress_all_blocks_for_generate_baseline_model':
            shutil.copyfile(tmp_small_model_path, os.path.join(self.res_save_dir, 'small_model.pth'))
            return
        
        small_model_model = hyps['retraining_model_cls'](
            name='tmp_model',
            models_dict_path=tmp_small_model_path,
            device=self.models['main'].device
        )
        small_model_model.num_classes = scenario.num_classes
        os.remove(tmp_small_model_path)
        retraining_alg = hyps['retraining_alg_cls'](
            models={
                'main': small_model_model
            },
            res_save_dir=os.path.join(self.res_save_dir, f'retraining_proxy_model')
        )
        small_model_retraining_info = retraining_alg.run(scenario, {
            **hyps['retraining_hyps'], 
            # 'random_sim_policy': None,
            # 'auged_source_dataset_name': '',
            
            'optimizer_args': {'lr': hyps['retraining_hyps']['optimizer_args']['lr'], **hyps['retraining_hyps']['optimizer_args']},
            'num_iters': solved_num_retraining_iters
        })[0]
        
        retrained_proxy_model = small_model_model.model
        
        from .lib_transformer import proxy_model_feedback_to_knowledge_base, knowledge_base_feedback_to_fm, fm_feedback_to_knowledge_base
        before_updating_knowledge_base = copy.deepcopy(self.models['main'].knowledge_base)
        proxy_model_feedback_to_knowledge_base(
            retrained_proxy_model, self.models['main'].knowledge_base,
            hyps['qkv_layers_name'], hyps['proj_layers_name'], hyps['ff1_layers_name'], hyps['ff2_layers_name'],
            unpruned_neurons_idx_of_layers
        )
        knowledge_base_feedback_to_fm(before_updating_knowledge_base, self.models['main'].knowledge_base, self.models['main'].fm,
                                      self.models['main'].neuron_index, hyps['qkv_layers_name'], hyps['proj_layers_name'],
                                      hyps['ff1_layers_name'], hyps['ff2_layers_name'], hyps['knowledge_base_to_fm_alpha'])
        fm_feedback_to_knowledge_base(self.models['main'].knowledge_base, self.models['main'].fm,
                                      self.models['main'].neuron_index, hyps['qkv_layers_name'], hyps['proj_layers_name'],
                                      hyps['ff1_layers_name'], hyps['ff2_layers_name'], hyps['fm_to_knowledge_base_alpha'])
        
        return small_model_retraining_info, self.models['main']
        
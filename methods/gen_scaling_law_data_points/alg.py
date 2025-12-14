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
from torch import nn
import copy
import shutil
from utils.dl.common.model import get_model_size, get_module
from .model import GenScalingLawDataPointsModel
from utils.common.log import logger
from utils.common.others import get_tmp_filepath
from ..train_with_fbs.lib import set_sparsity, switch_bn_stats, debug_sparsity
from utils.dl.auto_augment import gen_random_sim_policy, generate_sim_datasets_with_same_aug
import matplotlib.pyplot as plt
    
    
class GenScalingLawDataPointsAlg(BaseAlg):
    def get_required_models_schema(self) -> Schema:
        return Schema({
            'main': GenScalingLawDataPointsModel
        })
        
    def get_required_hyp_schema(self) -> Schema:
        from schema import Optional
        return Schema({
            
            'sparsity': float,
            'optional_batch_sizes': object,
            'max_num_trials': int,
            'obtain_features_num_iters': int,
            'obtain_larget_model_target_loss_num_iters': int,
            
            'qkv_layers_name': list,
            'proj_layers_name': list,
            'ff1_layers_name': list,
            'ff2_layers_name': list,

            'retraining_alg_cls': object,
            'retraining_model_cls': object,
            'retraining_hyps': object
        })

    def run(self, scenario: Scenario, hyps: Dict) -> Dict[str, Any]:
                
        super().run(scenario, hyps)
        
        assert isinstance(self.models['main'], GenScalingLawDataPointsModel) # for auto completion
        assert len(hyps['optional_batch_sizes']) == 1
        
        scaling_law_data_points = []
        cand_factors = []
        
        # formal code
        for _ in range(hyps['max_num_trials']):
            batch_size = random.choice(hyps['optional_batch_sizes'])
            random_sim_policy = gen_random_sim_policy(random.randint(5, 50), False)
            blocks_sparsity = [0. for _ in range(len(hyps['qkv_layers_name']))]
            pruned_blocks_index = random.choices(list(range(len(hyps['qkv_layers_name']))), 
                                                 k=random.randint(1, len(hyps['qkv_layers_name'])))
            for pi in pruned_blocks_index:
                blocks_sparsity[pi] = hyps['sparsity']
            
            cand_factors += [(blocks_sparsity, batch_size, random_sim_policy)]
        
        # for motivation
        # random_sim_policy2 = gen_random_sim_policy(random.randint(5, 50), False)
        # random_sim_policy1 = gen_random_sim_policy(random.randint(5, 50), False)
        # random_sim_policy3 = gen_random_sim_policy(random.randint(5, 50), False)
        # random_sim_policy4 = gen_random_sim_policy(random.randint(5, 50), False)
        
        # cand_factors += [(
        #     [0. for _ in range(12)],
        #     [0. for _ in range(12)],
        #     64,
        #     random_sim_policy1
        # )]
        # for pruned_attn_index in range(12):
        #     cand_factors += [(
        #         [0.9 if i == pruned_attn_index else 0. for i in range(12)],
        #         [0. for _ in range(12)],
        #         64,
        #         random_sim_policy1
        #     )]
        # for pruned_ffn_index in range(12):
        #     cand_factors += [(
        #         [0. for _ in range(12)],
        #         [0.9 if i == pruned_ffn_index else 0. for i in range(12)],
        #         64,
        #         random_sim_policy1
        #     )]
        
        # cand_factors += [(
        #     [0. for _ in range(12)],
        #     [0. for _ in range(12)],
        #     64,
        #     random_sim_policy4
        # )]
        # for pruned_attn_index in range(12):
        #     cand_factors += [(
        #         [0.9 if i == pruned_attn_index else 0. for i in range(12)],
        #         [0. for _ in range(12)],
        #         64,
        #         random_sim_policy4
        #     )]
        # for pruned_ffn_index in range(12):
        #     cand_factors += [(
        #         [0. for _ in range(12)],
        #         [0.9 if i == pruned_ffn_index else 0. for i in range(12)],
        #         64,
        #         random_sim_policy4
        #     )]
        # for motivation end
        
        self.source_features = None
        
        def arr_to_str(arr):
            return [f'{ai:.2f}' for ai in arr]
        
        trial_index = 0
        for blocks_sparsity, batch_size, random_sim_policy in cand_factors:
            logger.info(f'---->\n\tretraining trial {trial_index}/{len(cand_factors)} | '
                        f'ffns_sparsity: {arr_to_str(blocks_sparsity)}, '
                        f'batch_size: {batch_size}, sim_aug_magnitude: {random_sim_policy}')
            scaling_law_data_points += self.retraining_with_random_conditions(scenario, blocks_sparsity, 
                                                                              batch_size, random_sim_policy, trial_index, hyps)
            torch.save(scaling_law_data_points, os.path.join(self.res_save_dir, 'scaling_law_data_points.pth'))
            trial_index += 1
    
    def retraining_with_random_conditions(self, 
                                          scenario: Scenario,
                                          blocks_sparsity, batch_size, random_sim_policy,
                                          trial_index,
                                          hyps):
        """
        Return the inputs and output of our scaling law
        """
        
        assert isinstance(self.models['main'], GenScalingLawDataPointsModel) # for auto completion
        
        # torch.cuda.empty_cache()
        
        offline_datasets = scenario.get_offline_datasets(use_before_res=True)
        random_source_dataset_name = random.choice(list(offline_datasets.keys()))
        # random_source_dataset_name = 'SYNSIGNS'
        source_datasets = offline_datasets[random_source_dataset_name]
        logger.info(f'randomly choose {random_source_dataset_name} as source dataset')
        sim_target_datasets = {}
        
        # generate simulated target distribution
        aug_datasets = generate_sim_datasets_with_same_aug([source_datasets['train'], source_datasets['val']],
                                                            random_sim_policy)
        sim_target_datasets['train'] = aug_datasets[0]
        sim_target_datasets['val'] = aug_datasets[1]
        
        retraining_hyps = hyps['retraining_hyps']
        source_train_dataloader = iter(build_dataloader(source_datasets['train'], retraining_hyps['train_batch_size'], retraining_hyps['num_workers'],
                                        True, None))
        target_train_dataloader = iter(build_dataloader(sim_target_datasets['train'], retraining_hyps['train_batch_size'], retraining_hyps['num_workers'],
                                        True, None))
        
        # set sparsity
        attns_sparsity = ffns_sparsity = blocks_sparsity
        from utils.common.others import longest_common_prefix
        for i, layer_name in enumerate(hyps['qkv_layers_name']):
            if isinstance(layer_name, list):
                layer_name = longest_common_prefix(layer_name)
                if layer_name.endswith('.'):
                    layer_name = layer_name[0: -1]
            # print(layer_name)
            set_sparsity(get_module(self.models['main'].model, layer_name), attns_sparsity[i])
            set_sparsity(get_module(self.models['main'].model, hyps['proj_layers_name'][i]), attns_sparsity[i])
        for i, layer_name in enumerate(hyps['ff1_layers_name']):
            if isinstance(layer_name, list):
                layer_name = longest_common_prefix(layer_name)
                set_sparsity(get_module(self.models['main'].model, layer_name), ffns_sparsity[i])
            else:
                set_sparsity(get_module(self.models['main'].model, layer_name), ffns_sparsity[i])
            set_sparsity(get_module(self.models['main'].model, hyps['ff2_layers_name'][i]), ffns_sparsity[i])
        
        debug_sparsity(self.models['main'].model)
        
        if 'bn_stats' in self.models['main'].models_dict.keys():
            switch_bn_stats(self.models['main'].model, self.models['main'].models_dict['bn_stats'])
        self.models['main'].to_eval_mode()
        
        device = self.models['main'].device
        
        # obtain source/target features and calculate distance bewteen them
        given_target_samples = next(target_train_dataloader)[0].to(device)
        # small_model, _, _ = self.models['main'].generate_small_model(given_target_samples)
        with torch.no_grad():
            output_entropy = self.models['main'].get_output_entropy(given_target_samples)
            rep_target_sample = given_target_samples[output_entropy.argmax(): output_entropy.argmax() + 1]
            o1 = self.models['main'].infer(rep_target_sample)
            from .lib_transformer import generate_small_model
            small_model = generate_small_model(self.models['main'].model, hyps['qkv_layers_name'], hyps['proj_layers_name'], 
                                hyps['ff1_layers_name'], hyps['ff2_layers_name'])
            large_model = self.models['main'].model
            self.models['main'].model = small_model
            self.models['main'].to_eval_mode()
            o2 = self.models['main'].infer(rep_target_sample)
            self.models['main'].model = large_model
            
            diff = ((o1 - o2) ** 2).sum()
            assert diff < 1e-4, diff
            
        # torch.cuda.empty_cache()
            
        large_model_size = get_model_size(self.models['main'].model, True)
        small_model_size = get_model_size(small_model, True)
        logger.info(f'large_model_size: {large_model_size:.3f}MB, small_model_size: {small_model_size:.3f}MB (output diff: {diff:.6f})')
        
        self.models['main'].to_eval_mode()
        
        with torch.no_grad():
            target_features = []
            hook = self.models['main'].get_feature_hook()
            for _ in range(hyps['obtain_features_num_iters']):
                target_samples = next(target_train_dataloader)[0].to(device)
                self.models['main'].infer(target_samples)
                target_features += [hook.input.detach()]
            hook.remove()
            target_features = torch.cat(target_features)
            
            if self.source_features is None:
                source_features = []
                hook = self.models['main'].get_feature_hook()
                for _ in range(hyps['obtain_features_num_iters']):
                    source_samples = next(source_train_dataloader)[0].to(device)
                    self.models['main'].infer(source_samples)
                    source_features += [hook.input.detach()]
                hook.remove()
                source_features = torch.cat(source_features)
            else:
                source_features = self.source_features
            
            from .fid_distance import calculate_frechet_distance
            source_target_dist_distance = calculate_frechet_distance(source_features.cpu(), target_features.cpu())
            logger.info(f'source_target_dist_distance: {source_target_dist_distance:.4f} (calculated using {source_features.size(0)} samples)')
            
            f1, f2 = source_features.detach().cpu().numpy(), target_features.detach().cpu().numpy()
            mu1, sigma1 = np.mean(f1, axis=0), np.cov(f1, rowvar=False)
            mu2, sigma2 = np.mean(f2, axis=0), np.cov(f2, rowvar=False)
            features_stats = (mu1, sigma1, mu2, sigma2)
            
        # torch.cuda.empty_cache()
        
        # get large_model_loss_in_target_dist
        tmp_large_model_path = get_tmp_filepath()
        self.models['main'].save_model(tmp_large_model_path)
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
            res_save_dir=os.path.join(self.res_save_dir, f'retraining_trials/{trial_index:04d}')
        )
        large_model_loss_in_target_dist = retraining_alg.run(scenario, {
            **hyps['retraining_hyps'], 
            'num_iters': hyps['obtain_larget_model_target_loss_num_iters'],
            'optimizer_args': {'lr': 1e-9},
            'freeze_bn': True,
            'random_sim_policy': random_sim_policy,
            'auged_source_dataset_name': random_source_dataset_name
        })[0]['total_losses']
        logger.info(f'large_model_loss_in_target_dist: {large_model_loss_in_target_dist}')
        large_model_loss_in_target_dist = sum(large_model_loss_in_target_dist) / len(large_model_loss_in_target_dist)
        shutil.rmtree(os.path.join(self.res_save_dir, f'retraining_trials/{trial_index:04d}'))
        os.remove(tmp_large_model_path)
        
        # torch.cuda.empty_cache()
        
        # real run: retraining small model
        tmp_small_model_path = get_tmp_filepath()
        torch.save({'main': small_model}, tmp_small_model_path)
        small_model_model = hyps['retraining_model_cls'](
            name='tmp_model',
            models_dict_path=tmp_small_model_path,
            device=self.models['main'].device
        )
        small_model_model.num_classes = scenario.num_classes
        retraining_alg = hyps['retraining_alg_cls'](
            models={
                'main': small_model_model
            },
            res_save_dir=os.path.join(self.res_save_dir, f'retraining_trials/{trial_index:04d}')
        )
        small_model_retraining_info = retraining_alg.run(scenario, {
            **hyps['retraining_hyps'], 
            'random_sim_policy': random_sim_policy,
            'auged_source_dataset_name': random_source_dataset_name,
            
            'train_batch_size': batch_size,
            'optimizer_args': {**hyps['retraining_hyps']['optimizer_args'],
                               'lr': hyps['retraining_hyps']['optimizer_args']['lr'] * batch_size / hyps['retraining_hyps']['train_batch_size']},
        })[0]
        os.remove(tmp_small_model_path)
        shutil.rmtree(os.path.join(self.res_save_dir, f'retraining_trials/{trial_index:04d}/backup_codes'))

        retraining_accs_info = small_model_retraining_info['accs']
        scaling_law_data_points = []
        for retraining_acc_info in retraining_accs_info:
            scaling_law_inputs = {
                'small_model_size': small_model_size,
                'attns_sparsity': attns_sparsity,
                'ffns_sparsity': ffns_sparsity,
                'batch_size': batch_size,
                'source_dataset_index': list(offline_datasets.keys()).index(random_source_dataset_name),
                
                'source_target_dist_distance': source_target_dist_distance,
                'large_model_loss_in_target_dist': large_model_loss_in_target_dist,
                
                'num_retraining_iters': retraining_acc_info['iter'],
                
                'features_stats': features_stats
            }
            scaling_law_output = retraining_acc_info['acc']
            aux_info = {
                'source_dataset_name': random_source_dataset_name,
                'sim_policy': random_sim_policy.to_json(),
                # 'source_features': source_features.cpu(),
                # 'target_features': target_features.cpu(),
                'hyps': hyps,
                'retraining_acc_info': retraining_acc_info
            }
            
            scaling_law_data_points += [(scaling_law_inputs, scaling_law_output, aux_info)]
            
        return scaling_law_data_points
    
    @torch.no_grad()
    def get_source_features_and_target_features_mean_covariance(self, scenario, hyps, data_points_file_path):
        assert isinstance(self.models['main'], GenScalingLawDataPointsModel) # for auto completion
        
        from utils.dl.auto_augment import generate_sim_datasets_with_same_aug, SimTargetDomainPolicy
        
        retraining_hyps = hyps['retraining_hyps']
        res_save_dir = os.path.dirname(data_points_file_path)
        self.source_features = None
        
        offline_datasets = scenario.get_offline_datasets(use_before_res=True)
        source_train_dataloaders = {
            k: iter(build_dataloader(v['train'], retraining_hyps['train_batch_size'], retraining_hyps['num_workers'], True, None))
            for k, v in offline_datasets.items()
        }
        
        data_settings = {}
        data_points = torch.load(data_points_file_path)
        for scaling_law_inputs, scaling_law_output, aux_info in data_points:
            if scaling_law_inputs['num_retraining_iters'] != 0:
                continue
            
            data_settings[str((
                aux_info['source_dataset_name'],
                scaling_law_inputs['sparsity'],
                aux_info['sim_policy']
            ))] = (
                aux_info['source_dataset_name'],
                scaling_law_inputs['sparsity'],
                aux_info['sim_policy']
            )
        
        device = self.models['main'].device
        
        res = {}
        final_res = []
        
        for source_dataset_name, model_sparsity, sim_policy_args in tqdm.tqdm(data_settings.values(), dynamic_ncols=True):
            set_sparsity(self.models['main'].model, model_sparsity)
            switch_bn_stats(self.models['main'].model, self.models['main'].models_dict['bn_stats'])
            self.models['main'].to_eval_mode()
            
            sim_policy = SimTargetDomainPolicy(sim_policy_args['funcs'], sim_policy_args['magnitudes'])

            source_dataset = offline_datasets[source_dataset_name]['train']
            auged_source_dataset = generate_sim_datasets_with_same_aug([source_dataset], sim_policy)[0]
            source_train_dataloader = source_train_dataloaders[source_dataset_name]
            target_train_dataloader = iter(build_dataloader(auged_source_dataset, retraining_hyps['train_batch_size'], retraining_hyps['num_workers'],
                                            True, None))
            
            target_features = []
            hook = self.models['main'].get_feature_hook()
            for _ in range(hyps['obtain_features_num_iters']):
                target_samples = next(target_train_dataloader)[0].to(device)
                self.models['main'].infer(target_samples)
                target_features += [hook.input.detach()]
            hook.remove()
            target_features = torch.cat(target_features)
            
            if self.source_features is None:
                source_features = []
                hook = self.models['main'].get_feature_hook()
                for _ in range(hyps['obtain_features_num_iters']):
                    source_samples = next(source_train_dataloader)[0].to(device)
                    self.models['main'].infer(source_samples)
                    source_features += [hook.input.detach()]
                hook.remove()
                source_features = torch.cat(source_features)
            else:
                source_features = self.source_features
                
            f1, f2 = source_features.detach().cpu().numpy(), target_features.detach().cpu().numpy()
            mu1, sigma1 = np.mean(f1, axis=0), np.cov(f1, rowvar=False)
            mu2, sigma2 = np.mean(f2, axis=0), np.cov(f2, rowvar=False)
            
            res[str((source_dataset_name, model_sparsity, sim_policy_args))] = (mu1, sigma1, mu2, sigma2)
            torch.save(res, os.path.join(res_save_dir, 'source_target_features_mean_covariance.pth.tmp'))
            
        for scaling_law_inputs, scaling_law_output, aux_info in data_points:
            final_res += [res[str((
                aux_info['source_dataset_name'],
                scaling_law_inputs['sparsity'],
                aux_info['sim_policy']
            ))]]
        torch.save(final_res, os.path.join(res_save_dir, 'source_target_features_mean_covariance.pth'))
        
    def get_extracted_small_model_importance_score(self, scenario, hyps, data_points_file_path):
        assert isinstance(self.models['main'], GenScalingLawDataPointsModel) # for auto completion
        
        from utils.dl.auto_augment import generate_sim_datasets_with_same_aug, SimTargetDomainPolicy
        
        retraining_hyps = hyps['retraining_hyps']
        res_save_dir = os.path.dirname(data_points_file_path)
        self.source_features = None
        
        offline_datasets = scenario.get_offline_datasets(use_before_res=True)
        source_train_dataloaders = {
            k: iter(build_dataloader(v['train'], retraining_hyps['train_batch_size'], retraining_hyps['num_workers'], True, None))
            for k, v in offline_datasets.items()
        }
        
        data_settings = {}
        before_accs = {}
        after_accs = {}
        data_points = torch.load(data_points_file_path)
        for scaling_law_inputs, scaling_law_output, aux_info in data_points:
            if scaling_law_inputs['num_retraining_iters'] == 0:
            
                data_settings[str((
                    aux_info['source_dataset_name'],
                    scaling_law_inputs['sparsity'],
                    aux_info['sim_policy']
                ))] = (
                    aux_info['source_dataset_name'],
                    scaling_law_inputs['sparsity'],
                    aux_info['sim_policy']
                )
                before_accs[str((
                    aux_info['source_dataset_name'],
                    scaling_law_inputs['sparsity'],
                    aux_info['sim_policy']
                ))] = scaling_law_output
            
            elif scaling_law_inputs['num_retraining_iters'] == 500:
                after_accs[str((
                    aux_info['source_dataset_name'],
                    scaling_law_inputs['sparsity'],
                    aux_info['sim_policy']
                ))] = scaling_law_output
        
        device = self.models['main'].device
        
        res = {}
        final_res = []
        
        # source
        importance_scores_sort_index_of_source_datasets = {}
        for source_dataset_name, source_dataset in offline_datasets.items():
            set_sparsity(self.models['main'].model, 0.8)
            switch_bn_stats(self.models['main'].model, self.models['main'].models_dict['bn_stats'])
            self.models['main'].to_eval_mode()

            source_train_dataloader = source_train_dataloaders[source_dataset_name]
            
            importance_scores_sort_indexes = []
            for trial_index in range(2):
                source_samples = next(source_train_dataloader)[0].to(device)
                small_model, _, importance_scores = self.models['main'].generate_small_model(source_samples)
                
                importance_scores_sort_index = {k: v.argsort().cpu().numpy() for k, v in importance_scores.items()}
                importance_scores_sort_indexes += [importance_scores_sort_index]
            importance_scores_sort_index_of_source_datasets[source_dataset_name] = importance_scores_sort_indexes[-1]
            
            avg_kendalltau_score = 0.
            avg_kendalltau_score_n = 0
            for i1, r1 in enumerate(importance_scores_sort_indexes):
                for i2, r2 in enumerate(importance_scores_sort_indexes):
                    if i1 == i2:
                        continue
                    avg_kendalltau_score += self.compare_two_argsort_results_dict(r1, r2)
                    avg_kendalltau_score_n += 1
            avg_kendalltau_score /= avg_kendalltau_score_n
            
            logger.info('source dataset: {}, inside avg. kendalltau score: {}'.format(source_dataset_name, avg_kendalltau_score))
        # exit()
        
        avg_kendalltau_score_between_source_and_target = 0.
        
        kendalltau_scores_between_source_and_target = []
        res_before_accs = []
        res_after_accs = []
        
        for source_dataset_name, model_sparsity, sim_policy_args in tqdm.tqdm(data_settings.values(), dynamic_ncols=True):
            set_sparsity(self.models['main'].model, model_sparsity)
            switch_bn_stats(self.models['main'].model, self.models['main'].models_dict['bn_stats'])
            self.models['main'].to_eval_mode()
            
            sim_policy = SimTargetDomainPolicy(sim_policy_args['funcs'], sim_policy_args['magnitudes'])

            source_dataset = offline_datasets[source_dataset_name]['train']
            auged_source_dataset = generate_sim_datasets_with_same_aug([source_dataset], sim_policy)[0]
            source_train_dataloader = source_train_dataloaders[source_dataset_name]
            target_train_dataloader = iter(build_dataloader(auged_source_dataset, retraining_hyps['train_batch_size'], retraining_hyps['num_workers'],
                                            True, None))
            
            given_target_samples = next(target_train_dataloader)[0]
            small_model, _, importance_scores = self.models['main'].generate_small_model(given_target_samples)
            importance_scores_sort_index = {k: v.argsort().cpu().numpy() for k, v in importance_scores.items()}
            
            # get importance score
            # importance_score_per_layer = [float(sum(v) / len(v)) for v in importance_scores.values()]
            # logger.info(f'avg. importance score per layer: {importance_score_per_layer}')
            # print(f'avg. importance score per layer: {importance_scores}')
            
            # res[str((source_dataset_name, model_sparsity, sim_policy_args))] = importance_score_per_layer
            # torch.save(res, os.path.join(res_save_dir, 'importance_score_per_layer.pth.tmp'))
            
            kendalltau_score = self.compare_two_argsort_results_dict(importance_scores_sort_index_of_source_datasets[source_dataset_name], 
                                                                     importance_scores_sort_index)
            iou = self.compare_two_selection_iou(importance_scores_sort_index_of_source_datasets[source_dataset_name], 
                                                importance_scores_sort_index)
            res[str((source_dataset_name, model_sparsity, sim_policy_args))] = (importance_scores_sort_index_of_source_datasets,
                                                                                importance_scores_sort_index)
            torch.save(res, os.path.join(res_save_dir, 'importance_score_per_layer.pth.tmp'))
            
            cur_before_acc = before_accs[str((source_dataset_name, model_sparsity, sim_policy_args))]
            cur_after_acc = after_accs[str((source_dataset_name, model_sparsity, sim_policy_args))]
            
            logger.info(f'kendalltau score: {kendalltau_score:.6f}, before acc: {cur_before_acc:.6f}, after acc: {cur_after_acc:.6f}')
            
            kendalltau_scores_between_source_and_target += [kendalltau_score]
            res_before_accs += [cur_before_acc]
            res_after_accs += [cur_after_acc]

            plt.scatter(kendalltau_scores_between_source_and_target, res_before_accs)
            plt.xlabel('kendall tau score')
            plt.ylabel('acc before retraining')
            plt.savefig(os.path.join(res_save_dir, 'kendalltau_score_vs_before_acc.png'))
            plt.clf()
            
            plt.scatter(kendalltau_scores_between_source_and_target, res_after_accs)
            plt.xlabel('kendall tau score')
            plt.ylabel('acc after retraining')
            plt.savefig(os.path.join(res_save_dir, 'kendalltau_score_vs_after_acc.png'))
            plt.clf()
            
            avg_kendalltau_score_between_source_and_target += kendalltau_score
        
        avg_kendalltau_score_between_source_and_target /= len(data_settings)
        logger.info(f'avg. kendalltau score between source and target: {avg_kendalltau_score_between_source_and_target}')
            
        for scaling_law_inputs, scaling_law_output, aux_info in data_points:
            final_res += [res[str((
                aux_info['source_dataset_name'],
                scaling_law_inputs['sparsity'],
                aux_info['sim_policy']
            ))]]
        torch.save(final_res, os.path.join(res_save_dir, 'importance_score_per_layer.pth'))
        
    def compare_two_argsort_results_dict(self, r1, r2):
        from scipy.stats import kendalltau
        
        res = 0.
        for v1, v2 in zip(r1.values(), r2.values()):
            res += kendalltau(v1, v2).correlation
        res /= len(r1)
        return res
    
    def compare_two_selection_iou(self, r1, r2):
        from scipy.stats import kendalltau
        
        res = 0.
        for v1, v2 in zip(r1.values(), r2.values()):
            v1, v2 = set(v1), set(v2)
            res += len(v1.intersection(v2)) / len(v1.union(v2))
        res /= len(r1)
        return res
    
    @torch.no_grad()
    def get_source_target_bn_stats_difference(self, scenario, hyps, data_points_file_path):
        assert isinstance(self.models['main'], GenScalingLawDataPointsModel) # for auto completion
        
        from utils.dl.auto_augment import generate_sim_datasets_with_same_aug, SimTargetDomainPolicy
        
        retraining_hyps = hyps['retraining_hyps']
        res_save_dir = os.path.dirname(data_points_file_path)
        # res_save_dir = self.res_save_dir
        self.source_features = None
        
        offline_datasets = scenario.get_offline_datasets(use_before_res=True)
        source_train_dataloaders = {
            k: iter(build_dataloader(v['train'], retraining_hyps['train_batch_size'], retraining_hyps['num_workers'], True, None))
            for k, v in offline_datasets.items()
        }
        
        data_settings = {}
        before_accs = {}
        after_accs = {}
        data_points = torch.load(data_points_file_path)
        for scaling_law_inputs, scaling_law_output, aux_info in data_points:
            if scaling_law_inputs['num_retraining_iters'] == 0:
            
                data_settings[str((
                    aux_info['source_dataset_name'],
                    scaling_law_inputs['sparsity'],
                    aux_info['sim_policy']
                ))] = (
                    aux_info['source_dataset_name'],
                    scaling_law_inputs['sparsity'],
                    aux_info['sim_policy']
                )
                before_accs[str((
                    aux_info['source_dataset_name'],
                    scaling_law_inputs['sparsity'],
                    aux_info['sim_policy']
                ))] = scaling_law_output
            
            elif scaling_law_inputs['num_retraining_iters'] == 500:
                after_accs[str((
                    aux_info['source_dataset_name'],
                    scaling_law_inputs['sparsity'],
                    aux_info['sim_policy']
                ))] = scaling_law_output
        
        device = self.models['main'].device
        
        res = {}
        final_res = []
        
        avg_diffs = {}
        res_before_accs, res_after_accs = {}, {}
        
        for source_dataset_name, model_sparsity, sim_policy_args in tqdm.tqdm(data_settings.values(), dynamic_ncols=True):
            set_sparsity(self.models['main'].model, model_sparsity)
            switch_bn_stats(self.models['main'].model, self.models['main'].models_dict['bn_stats'])
            self.models['main'].to_eval_mode()
            
            sim_policy = SimTargetDomainPolicy(sim_policy_args['funcs'], sim_policy_args['magnitudes'])

            source_dataset = offline_datasets[source_dataset_name]['train']
            auged_source_dataset = generate_sim_datasets_with_same_aug([source_dataset], sim_policy)[0]
            source_train_dataloader = source_train_dataloaders[source_dataset_name]
            target_train_dataloader = iter(build_dataloader(auged_source_dataset, retraining_hyps['train_batch_size'], retraining_hyps['num_workers'],
                                            True, None))
            
            # cal self.models['main'] bn stats on target samples
            from methods.train_with_fbs.lib import bn_cal
            
            model_for_bn_cal = copy.deepcopy(self.models['main'].model)
            bn_cal(model_for_bn_cal, source_train_dataloader, 1, device)
            source_bn_stats = {}
            for n, m in model_for_bn_cal.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    source_bn_stats[n] = copy.deepcopy(m)
                    
            model_for_bn_cal = copy.deepcopy(self.models['main'].model)
            bn_cal(model_for_bn_cal, target_train_dataloader, 1, device)
            target_bn_stats = {}
            for n, m in model_for_bn_cal.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    target_bn_stats[n] = copy.deepcopy(m)
                    
            res[str((source_dataset_name, model_sparsity, sim_policy_args))] = (source_bn_stats, target_bn_stats)
            torch.save(res, os.path.join(res_save_dir, 'bn_stats_per_layer.pth.tmp'))
            
            # compare source_bn_stats and target_bn_stats
            avg_diff = 0.
            for m1, m2 in zip(source_bn_stats.values(), target_bn_stats.values()):
                diff = (m1.running_mean - m2.running_mean).norm() + (m1.running_var - m2.running_var).norm()
                avg_diff += diff
            avg_diff /= len(source_bn_stats)
            avg_diff = float(avg_diff)
            
            cur_before_acc = before_accs[str((source_dataset_name, model_sparsity, sim_policy_args))]
            cur_after_acc = after_accs[str((source_dataset_name, model_sparsity, sim_policy_args))]
            
            logger.info(f'avg. diff between source/target bn stats: {avg_diff:.6f}, before acc: {cur_before_acc:.6f}, after acc: {cur_after_acc:.6f}')
            
            if str(model_sparsity) not in avg_diffs.keys():
                avg_diffs[str(model_sparsity)] = []
                res_before_accs[str(model_sparsity)] = []
                res_after_accs[str(model_sparsity)] = []
            
            avg_diffs[str(model_sparsity)] += [avg_diff]
            res_before_accs[str(model_sparsity)] += [cur_before_acc]
            res_after_accs[str(model_sparsity)] += [cur_after_acc]

            plt.scatter(avg_diffs[str(model_sparsity)], res_before_accs[str(model_sparsity)])
            plt.xlabel('avg diff')
            plt.ylabel('acc before retraining')
            plt.savefig(os.path.join(res_save_dir, f'bn_stats_diff_vs_before_acc_{[str(model_sparsity)]}.png'))
            plt.clf()
            
            plt.scatter(avg_diffs[str(model_sparsity)], res_after_accs[str(model_sparsity)])
            plt.xlabel('avg diff')
            plt.ylabel('acc after retraining')
            plt.savefig(os.path.join(res_save_dir, f'bn_stats_diff_score_vs_after_acc_{[str(model_sparsity)]}.png'))
            plt.clf()
            
        for scaling_law_inputs, scaling_law_output, aux_info in data_points:
            final_res += [res[str((
                aux_info['source_dataset_name'],
                scaling_law_inputs['sparsity'],
                aux_info['sim_policy']
            ))]]
        torch.save(final_res, os.path.join(res_save_dir, 'bn_stats_per_layer.pth'))
        
    def run_real_target(self, scenario: Scenario, hyps: Dict) -> Dict[str, Any]:
                
        super().run(scenario, hyps)
        
        assert isinstance(self.models['main'], GenScalingLawDataPointsModel) # for auto completion
        assert len(hyps['optional_batch_sizes']) == 1
        
        scaling_law_data_points = []
        cand_factors = []
        
        # formal code
        for _ in range(hyps['max_num_trials']):
            batch_size = random.choice(hyps['optional_batch_sizes'])
            random_sim_policy = gen_random_sim_policy(random.randint(5, 50), False)
            blocks_sparsity = [0. for _ in range(len(hyps['qkv_layers_name']))]
            pruned_blocks_index = random.choices(list(range(len(hyps['qkv_layers_name']))), 
                                                 k=random.randint(1, len(hyps['qkv_layers_name'])))
            for pi in pruned_blocks_index:
                blocks_sparsity[pi] = hyps['sparsity']
            
            cand_factors += [(blocks_sparsity, batch_size, random_sim_policy)]

        self.source_features = None
        
        def arr_to_str(arr):
            return [f'{ai:.2f}' for ai in arr]
        
        trial_index = 0
        for blocks_sparsity, batch_size, random_sim_policy in cand_factors:
            logger.info(f'---->\n\tretraining trial {trial_index}/{len(cand_factors)} | '
                        f'ffns_sparsity: {arr_to_str(blocks_sparsity)}, '
                        f'batch_size: {batch_size}, sim_aug_magnitude: {random_sim_policy}')
            scaling_law_data_points += self.retraining_with_random_conditions_in_real_target_dataset(scenario, blocks_sparsity, 
                                                                              batch_size, random_sim_policy, trial_index, hyps)
            torch.save(scaling_law_data_points, os.path.join(self.res_save_dir, 'scaling_law_data_points_real_target.pth'))
            trial_index += 1
        
    def retraining_with_random_conditions_in_real_target_dataset(self, 
                                          scenario: Scenario,
                                          blocks_sparsity, batch_size, random_sim_policy,
                                          trial_index,
                                          hyps):
        """
        Return the inputs and output of our scaling law
        """
        
        assert isinstance(self.models['main'], GenScalingLawDataPointsModel) # for auto completion
        
        # torch.cuda.empty_cache()
        
        target_dataset_name = random.choice(scenario.target_domains_order)
        source_dataset_name = list(scenario.target_source_map[target_dataset_name].keys())[0]
        
        offline_datasets = scenario.get_offline_datasets(use_before_res=True)
        source_datasets = offline_datasets[source_dataset_name]
        
        scenario.cur_domain_index = scenario.target_domains_order.index(target_dataset_name)

        logger.info(f'choose {source_dataset_name} as source dataset, {target_dataset_name} as target dataset')
        sim_target_datasets = {}
        
        # generate simulated target distribution
        # aug_datasets = generate_sim_datasets_with_same_aug([source_datasets['train'], source_datasets['val']],
        #                                                     random_sim_policy)

        online_datasets = scenario.get_online_cur_domain_datasets_for_training()
        train_dataset = online_datasets[target_dataset_name]['train']
        val_dataset = online_datasets[target_dataset_name]['val']
        datasets_for_inference = scenario.get_online_cur_domain_datasets_for_inference()
        test_dataset = datasets_for_inference
        
        sim_target_datasets['train'] = train_dataset
        sim_target_datasets['val'] = test_dataset
        
        retraining_hyps = hyps['retraining_hyps']
        source_train_dataloader = iter(build_dataloader(source_datasets['train'], retraining_hyps['train_batch_size'], retraining_hyps['num_workers'],
                                        True, None))
        target_train_dataloader = iter(build_dataloader(sim_target_datasets['train'], retraining_hyps['train_batch_size'], retraining_hyps['num_workers'],
                                        True, None))
        
        # set sparsity
        attns_sparsity = ffns_sparsity = blocks_sparsity
        from utils.common.others import longest_common_prefix
        for i, layer_name in enumerate(hyps['qkv_layers_name']):
            if isinstance(layer_name, list):
                layer_name = longest_common_prefix(layer_name)
                if layer_name.endswith('.'):
                    layer_name = layer_name[0: -1]
            # print(layer_name)
            set_sparsity(get_module(self.models['main'].model, layer_name), attns_sparsity[i])
            set_sparsity(get_module(self.models['main'].model, hyps['proj_layers_name'][i]), attns_sparsity[i])
        for i, layer_name in enumerate(hyps['ff1_layers_name']):
            if isinstance(layer_name, list):
                layer_name = longest_common_prefix(layer_name)
                set_sparsity(get_module(self.models['main'].model, layer_name), ffns_sparsity[i])
            else:
                set_sparsity(get_module(self.models['main'].model, layer_name), ffns_sparsity[i])
            set_sparsity(get_module(self.models['main'].model, hyps['ff2_layers_name'][i]), ffns_sparsity[i])
        
        debug_sparsity(self.models['main'].model)
        
        if 'bn_stats' in self.models['main'].models_dict.keys():
            switch_bn_stats(self.models['main'].model, self.models['main'].models_dict['bn_stats'])
        self.models['main'].to_eval_mode()
        
        device = self.models['main'].device
        
        # obtain source/target features and calculate distance bewteen them
        given_target_samples = next(target_train_dataloader)[0].to(device)
        # small_model, _, _ = self.models['main'].generate_small_model(given_target_samples)
        with torch.no_grad():
            output_entropy = self.models['main'].get_output_entropy(given_target_samples)
            rep_target_sample = given_target_samples[output_entropy.argmax(): output_entropy.argmax() + 1]
            o1 = self.models['main'].infer(rep_target_sample)
            from .lib_transformer import generate_small_model
            small_model = generate_small_model(self.models['main'].model, hyps['qkv_layers_name'], hyps['proj_layers_name'], 
                                hyps['ff1_layers_name'], hyps['ff2_layers_name'])
            large_model = self.models['main'].model
            self.models['main'].model = small_model
            self.models['main'].to_eval_mode()
            o2 = self.models['main'].infer(rep_target_sample)
            self.models['main'].model = large_model
            
            diff = ((o1 - o2) ** 2).sum()
            assert diff < 1e-4, diff
            
        # torch.cuda.empty_cache()
            
        large_model_size = get_model_size(self.models['main'].model, True)
        small_model_size = get_model_size(small_model, True)
        logger.info(f'large_model_size: {large_model_size:.3f}MB, small_model_size: {small_model_size:.3f}MB (output diff: {diff:.6f})')
        
        self.models['main'].to_eval_mode()
        
        with torch.no_grad():
            target_features = []
            hook = self.models['main'].get_feature_hook()
            for _ in range(hyps['obtain_features_num_iters']):
                target_samples = next(target_train_dataloader)[0].to(device)
                self.models['main'].infer(target_samples)
                target_features += [hook.input.detach()]
            hook.remove()
            target_features = torch.cat(target_features)
            
            if self.source_features is None:
                source_features = []
                hook = self.models['main'].get_feature_hook()
                for _ in range(hyps['obtain_features_num_iters']):
                    source_samples = next(source_train_dataloader)[0].to(device)
                    self.models['main'].infer(source_samples)
                    source_features += [hook.input.detach()]
                hook.remove()
                source_features = torch.cat(source_features)
            else:
                source_features = self.source_features
            
            from .fid_distance import calculate_frechet_distance
            source_target_dist_distance = calculate_frechet_distance(source_features.cpu(), target_features.cpu())
            logger.info(f'source_target_dist_distance: {source_target_dist_distance:.4f} (#samples: {source_features.size(0)})')
            
            f1, f2 = source_features.detach().cpu().numpy(), target_features.detach().cpu().numpy()
            mu1, sigma1 = np.mean(f1, axis=0), np.cov(f1, rowvar=False)
            mu2, sigma2 = np.mean(f2, axis=0), np.cov(f2, rowvar=False)
            features_stats = (mu1, sigma1, mu2, sigma2)
            
        # torch.cuda.empty_cache()
        
        # get large_model_loss_in_target_dist
        tmp_large_model_path = get_tmp_filepath()
        self.models['main'].save_model(tmp_large_model_path)
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
            res_save_dir=os.path.join(self.res_save_dir, f'retraining_trials/{trial_index:04d}')
        )
        large_model_loss_in_target_dist = retraining_alg.run(scenario, {
            **hyps['retraining_hyps'], 
            'num_iters': hyps['obtain_larget_model_target_loss_num_iters'],
            'optimizer_args': {'lr': 1e-9},
            'freeze_bn': True,
            # 'random_sim_policy': random_sim_policy,
            # 'auged_source_dataset_name': random_source_dataset_name
        })[0]['total_losses']
        logger.info(f'large_model_loss_in_target_dist: {large_model_loss_in_target_dist}')
        large_model_loss_in_target_dist = sum(large_model_loss_in_target_dist) / len(large_model_loss_in_target_dist)
        shutil.rmtree(os.path.join(self.res_save_dir, f'retraining_trials/{trial_index:04d}'))
        os.remove(tmp_large_model_path)
        
        # torch.cuda.empty_cache()
        
        # real run: retraining small model
        tmp_small_model_path = get_tmp_filepath()
        torch.save({'main': small_model}, tmp_small_model_path)
        small_model_model = hyps['retraining_model_cls'](
            name='tmp_model',
            models_dict_path=tmp_small_model_path,
            device=self.models['main'].device
        )
        small_model_model.num_classes = scenario.num_classes
        retraining_alg = hyps['retraining_alg_cls'](
            models={
                'main': small_model_model
            },
            res_save_dir=os.path.join(self.res_save_dir, f'retraining_trials/{trial_index:04d}')
        )
        small_model_retraining_info = retraining_alg.run(scenario, {
            **hyps['retraining_hyps'], 
            # 'random_sim_policy': random_sim_policy,
            # 'auged_source_dataset_name': random_source_dataset_name,
            
            'train_batch_size': batch_size,
            'optimizer_args': {**hyps['retraining_hyps']['optimizer_args'],
                               'lr': hyps['retraining_hyps']['optimizer_args']['lr'] * batch_size / hyps['retraining_hyps']['train_batch_size']},
        })[0]
        os.remove(tmp_small_model_path)
        shutil.rmtree(os.path.join(self.res_save_dir, f'retraining_trials/{trial_index:04d}/backup_codes'))

        retraining_accs_info = small_model_retraining_info['accs']
        scaling_law_data_points = []
        for retraining_acc_info in retraining_accs_info:
            scaling_law_inputs = {
                'small_model_size': small_model_size,
                'attns_sparsity': attns_sparsity,
                'ffns_sparsity': ffns_sparsity,
                'batch_size': batch_size,
                'source_dataset_index': list(offline_datasets.keys()).index(source_dataset_name),
                
                'source_target_dist_distance': source_target_dist_distance,
                'large_model_loss_in_target_dist': large_model_loss_in_target_dist,
                
                'num_retraining_iters': retraining_acc_info['iter'],
                
                'features_stats': features_stats
            }
            scaling_law_output = retraining_acc_info['acc']
            aux_info = {
                'source_dataset_name': source_dataset_name,
                'target_dataset_name': target_dataset_name,
                'sim_policy': random_sim_policy.to_json(),
                # 'source_features': source_features.cpu(),
                # 'target_features': target_features.cpu(),
                'hyps': hyps,
                'retraining_acc_info': retraining_acc_info
            }
            
            scaling_law_data_points += [(scaling_law_inputs, scaling_law_output, aux_info)]
            
        return scaling_law_data_points
    
    
    def run_for_motivation(self, scenario: Scenario, hyps: Dict) -> Dict[str, Any]:
                
        super().run(scenario, hyps)
        
        assert isinstance(self.models['main'], GenScalingLawDataPointsModel) # for auto completion
        assert len(hyps['optional_batch_sizes']) == 1
        
        scaling_law_data_points = []
        cand_factors = []
        
        # formal code
        for _ in range(hyps['max_num_trials']):
            batch_size = random.choice(hyps['optional_batch_sizes'])
            random_sim_policy = gen_random_sim_policy(random.randint(5, 50), False)
            blocks_sparsity = [0. for _ in range(len(hyps['qkv_layers_name']))]
            pruned_blocks_index = random.choices(list(range(len(hyps['qkv_layers_name']))), 
                                                 k=random.randint(1, len(hyps['qkv_layers_name'])))
            for pi in pruned_blocks_index:
                blocks_sparsity[pi] = hyps['sparsity']
            
            cand_factors += [(blocks_sparsity, batch_size, random_sim_policy)]
        
        # for motivation
        random_sim_policy2 = gen_random_sim_policy(random.randint(5, 50), False)
        random_sim_policy1 = gen_random_sim_policy(random.randint(5, 50), False)
        random_sim_policy3 = gen_random_sim_policy(random.randint(5, 50), False)
        random_sim_policy4 = gen_random_sim_policy(random.randint(5, 50), False)
        
        # cand_factors += [(
        #     [0. for _ in range(12)],
        #     [0. for _ in range(12)],
        #     64,
        #     random_sim_policy1
        # )]
        for policy in [random_sim_policy1, random_sim_policy4]:
            cand_factors += [(
                [0.9 for i in range(12)],
                [0.9 for _ in range(12)],
                64,
                random_sim_policy1
            )]
            
            random_s = [random.random() * 0.9 for _ in range(12)]
            cand_factors += [(
                random_s,
                random_s,
                64,
                random_sim_policy1
            )]
            
            best_s = []
        # for motivation end
        
        self.source_features = None
        
        def arr_to_str(arr):
            return [f'{ai:.2f}' for ai in arr]
        
        trial_index = 0
        for blocks_sparsity, batch_size, random_sim_policy in cand_factors:
            logger.info(f'---->\n\tretraining trial {trial_index}/{len(cand_factors)} | '
                        f'ffns_sparsity: {arr_to_str(blocks_sparsity)}, '
                        f'batch_size: {batch_size}, sim_aug_magnitude: {random_sim_policy}')
            scaling_law_data_points += self.retraining_with_random_conditions(scenario, blocks_sparsity, 
                                                                              batch_size, random_sim_policy, trial_index, hyps)
            torch.save(scaling_law_data_points, os.path.join(self.res_save_dir, 'scaling_law_data_points.pth'))
            trial_index += 1
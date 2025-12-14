from utils.dl.common.env import set_random_seed
set_random_seed(1)

import sys
import torch
from torch import nn 
from utils.common.exp import get_res_save_dir, save_models_dict_for_init


if __name__ == '__main__':
    # 1. Data
    from exp.offline_preparing.vit_b_16.img_cls.scenario import scenario
    
    # 2. Model(s)
    fm = torch.load('/data/zql/concept-drift-in-edge-projects/network-scaling-in-retraining/dianzixuebao/offline_preparing/vit_b_16/img_cls/results/lora_fine_tune.py/20240415/999997-195115-trial/models/main_best_acc=0.9627.pt')
    knowledge_base = torch.load('/data/zql/concept-drift-in-edge-projects/network-scaling-in-retraining/dianzixuebao/offline_preparing/vit_b_16/img_cls/results/gen_neuron_index.py/20240416/999971-221828-trial/models/0.90/main_best_acc=0.7073.pt')
    neuron_index = torch.load('/data/zql/concept-drift-in-edge-projects/network-scaling-in-retraining/dianzixuebao/offline_preparing/vit_b_16/img_cls/results/gen_neuron_index.py/20240416/999971-221828-trial/models/0.90/best_neuron_index.pt')
    from analysis.scaling_law_trial.only_consider_pruned_block_index2 import *
    scaling_law = torch.load('/data/zql/concept-drift-in-edge-projects/network-scaling-in-retraining/analysis/scaling_law_trial/results/only_consider_pruned_block_index2.py/20240511/999999-105211-use_real_data_points/best_edge_scaling_law_fcn.pt')
    
    qkv_layers_name = [[f'vit.encoder.layer.{i}.attention.attention.{k}' for k in ['query', 'key', 'value']] for i in range(12)]
    proj_layers_name = [f'vit.encoder.layer.{i}.attention.output.dense' for i in range(12)]
    ff1_layers_name = [f'vit.encoder.layer.{i}.intermediate.dense' for i in range(12)]
    ff2_layers_name = [f'vit.encoder.layer.{i}.output.dense' for i in range(12)]
    
    models_dict = {
        'fm': fm['main'],
        'knowledge_base': knowledge_base['main'],
        'bn_stats': {},
        'neuron_index': neuron_index,
        'scaling_law': scaling_law,
        'profiles': {
            'blocks_retraining_time_per_iteration': [[0.16/12, 0.15/12]] * 12,
            'num_blocks_params': [[1193344 / 1024**2, 1193344 / 1024**2 * 0.125]] * 12,
        }
    }
    
    models_dict_path = models_dict
    
    from exp.online_retraining.vit_b_16.img_cls.model_impl import EdgeTAOnlineRunModel_ViTImageClassification, FeatureAlignmentModel_ViTImageClassification
    da_model = EdgeTAOnlineRunModel_ViTImageClassification(
        name='vit_b_16_img_cls_edgeta_da_model',
        models_dict_path=models_dict_path,
        device='cuda'
    )
    
    # 3. Algorithm
    from methods.edgeta_online_run_with_multiple_running_scaling import EdgeTAOnlineRunWithMultipleRunningScalingAlg
    from methods.feature_alignment_with_multiple_running_scaling.alg import FeatureAlignmentWithMultipleRunningScalingAlg
    
    # 4. Run!
    qkv_layers = [[f'vit.encoder.layer.{i}.attention.attention.{k}' for k in ['query', 'key', 'value']] for i in range(12)]
    proj_layers = [f'vit.encoder.layer.{i}.attention.output.dense' for i in range(12)]
    ff1_layers = [f'vit.encoder.layer.{i}.intermediate.dense' for i in range(12)]
    ff2_layers = [f'vit.encoder.layer.{i}.output.dense' for i in range(12)]
    
    from exp.online_retraining.vit_b_16.img_cls.da import da_exp
    
    da_exp(
        app_name='vit-b-16-img-cls-da-edgeta',
        scenario=scenario,
        da_alg=EdgeTAOnlineRunWithMultipleRunningScalingAlg,
        da_alg_hyp={
            'sparsity': 7/8,
            'sparsity_list': [7/8, 6/8, 5/8, 4/8],
            
            'obtain_features_num_iters': 200,
            'obtain_larget_model_target_loss_num_iters': 1,
            
            'qkv_layers_name': qkv_layers,
            'proj_layers_name': proj_layers,
            'ff1_layers_name': ff1_layers,
            'ff2_layers_name': ff2_layers,
            'only_add_fbs_in_qkv': False,
            
            'retraining_alg_cls': FeatureAlignmentWithMultipleRunningScalingAlg,
            'retraining_model_cls': FeatureAlignmentModel_ViTImageClassification,
            'retraining_hyps': {
                'train_batch_size': 64, # std batch_size
                'val_batch_size': 1024, # num samples for evaluating retraining accuracy
                'num_workers': 16,
                'optimizer': 'AdamW',
                'optimizer_args': {'lr': 1e-4}, # std lr for batch_size=64
                'scheduler': '',
                'scheduler_args': {},
                'num_iters': 500,
                'val_freq': 2,
                'feat_align_loss_weight': 3.,
                'freeze_bn': False,
                'random_sim_policy': None,
                'auged_source_dataset_name': None,
                'fp16': False,
                'collate_fn': None,
                
                'knowledge_base': None,
                'attention_values_of_layers': None,
                'unpruned_indexes_of_layers': None,
                
                # 'scaling_points': [],
                # 'scaling_points': [(100, 0.5)],
                # 'scaling_points': [(100, 0.5), (200, 0.875)],
                'scaling_points': eval(sys.argv[1]),
                
                'qkv_layers_name': qkv_layers,
                'proj_layers_name': proj_layers,
                'ff1_layers_name': ff1_layers,
                'ff2_layers_name': ff2_layers,
                'only_add_fbs_in_qkv': False,
                
                'use_train_data_for_test': True,
                
                'for_profiling': False
            },
            'retraining_window': 60,
            'knowledge_base_to_fm_alpha': 0.1,
            'fm_to_knowledge_base_alpha': 0.1,
            # 'opt_strategy': f'heuristic',
            'opt_strategy': ([True] * 12, 50),
            'up_bound_pred_acc': 0.7,
            'forward_func': lambda model, x: model(x).logits
        },
        da_model=da_model,
        device='cuda',
        __entry_file__=__file__,
        tag=sys.argv[1],
        use_entry_model_in_new_dist=False
    )
    
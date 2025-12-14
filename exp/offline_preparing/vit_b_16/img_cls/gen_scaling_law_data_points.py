from utils.dl.common.env import set_random_seed
import sys
# set_random_seed(5) 1st run
# set_random_seed(10) # 2nd run
# set_random_seed(15) # 2nd run
set_random_seed(int(sys.argv[1])) # 2nd run



import torch
import numpy as np
from utils.common.exp import get_res_save_dir, save_models_dict_for_init


if __name__ == '__main__':
    # 1. Data
    from exp.offline_preparing.vit_b_16.img_cls.scenario import scenario
    
    # 2. Model(s)
    models_dict_path = save_models_dict_for_init(
        torch.load('dianzixuebao/offline_preparing/vit_b_16/img_cls/results/gen_neuron_index.py/20240416/999971-221828-trial/models/0.90/main_best_acc=0.7073.pt'),
        __file__
    )
    from exp.online_retraining.vit_b_16.img_cls.model_impl import GenScalingLawDataPointsModel_ViTImageClassification
    model = GenScalingLawDataPointsModel_ViTImageClassification(
        name='vit_b_16',
        models_dict_path=models_dict_path,
        device='cuda'
    )
    
    # 3. Algorithm
    from methods.gen_scaling_law_data_points import GenScalingLawDataPointsAlg
    alg = GenScalingLawDataPointsAlg(
        models={
            'main': model
        },
        res_save_dir=get_res_save_dir(__file__, sys.argv[1])
    )
    
    qkv_layers = [[f'vit.encoder.layer.{i}.attention.attention.{k}' for k in ['query', 'key', 'value']] for i in range(12)]
    proj_layers = [f'vit.encoder.layer.{i}.attention.output.dense' for i in range(12)]
    ff1_layers = [f'vit.encoder.layer.{i}.intermediate.dense' for i in range(12)]
    ff2_layers = [f'vit.encoder.layer.{i}.output.dense' for i in range(12)]
    
    # 4. Run!
    from methods.feature_alignment import FeatureAlignmentAlg
    from exp.online_retraining.vit_b_16.img_cls.model_impl import FeatureAlignmentModel_ViTImageClassification
    alg.run(
        scenario=scenario, 
        hyps={
            'sparsity': 0.875,
            'optional_batch_sizes': [64],
            'max_num_trials': 5000,
            
            'obtain_features_num_iters': 200,
            'obtain_larget_model_target_loss_num_iters': 1,
            
            'qkv_layers_name': qkv_layers,
            'proj_layers_name': proj_layers,
            'ff1_layers_name': ff1_layers,
            'ff2_layers_name': ff2_layers,
            
            'retraining_alg_cls': FeatureAlignmentAlg,
            'retraining_model_cls': FeatureAlignmentModel_ViTImageClassification,
            'retraining_hyps': {
                'train_batch_size': 64, # std batch_size
                'val_batch_size': 1024, # num samples for evaluating retraining accuracy
                'num_workers': 16,
                'optimizer': 'AdamW',
                'optimizer_args': {'lr': 3e-4}, # std lr for batch_size=64
                'scheduler': '',
                'scheduler_args': {},
                'num_iters': 500,
                'val_freq': 10,
                'feat_align_loss_weight': 3.,
                'freeze_bn': False,
                'random_sim_policy': None,
                'auged_source_dataset_name': None,
                'fp16': True
            }
        }
    )
    
    
    # alg.run_real_target(
    #     scenario=scenario, 
    #     hyps={
    #         'sparsity': 0.875,
    #         'optional_batch_sizes': [64],
    #         'max_num_trials': 5000,
            
    #         'obtain_features_num_iters': 200,
    #         'obtain_larget_model_target_loss_num_iters': 1,
            
    #         'qkv_layers_name': qkv_layers,
    #         'proj_layers_name': proj_layers,
    #         'ff1_layers_name': ff1_layers,
    #         'ff2_layers_name': ff2_layers,
            
    #         'retraining_alg_cls': FeatureAlignmentAlg,
    #         'retraining_model_cls': FeatureAlignmentModel_ViTImageClassification,
    #         'retraining_hyps': {
    #             'train_batch_size': 64, # std batch_size
    #             'val_batch_size': 1024, # num samples for evaluating retraining accuracy
    #             'num_workers': 16,
    #             'optimizer': 'AdamW',
    #             'optimizer_args': {'lr': 3e-4}, # std lr for batch_size=64
    #             'scheduler': '',
    #             'scheduler_args': {},
    #             'num_iters': 500,
    #             'val_freq': 10,
    #             'feat_align_loss_weight': 3.,
    #             'freeze_bn': False,
    #             'random_sim_policy': None,
    #             'auged_source_dataset_name': None,
    #             'fp16': True
    #         }
    #     }
    # )
    
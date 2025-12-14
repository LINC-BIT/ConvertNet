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
    fm_models_dict_path = save_models_dict_for_init(
        torch.load('dianzixuebao/offline_preparing/vit_b_16/img_cls/results/lora_fine_tune.py/20240415/999997-195115-trial/models/main_best_acc=0.9627.pt'),
        __file__, uid='fm'
    )
    import glob
    kb_models_dict_path = save_models_dict_for_init(
        torch.load(glob.glob('dianzixuebao/offline_preparing/vit_b_16/img_cls/results/gen_knowledge_base.py/20240416/999997-115154-trial/models/main_best_acc=0.7347.pt')[0]),
        __file__, uid='kb'
    )
    from exp.online_retraining.vit_b_16.img_cls.model_impl import GenNeuronIndexModel_ViTImageClassification
    fm_model = GenNeuronIndexModel_ViTImageClassification(
        name='vit_b_16_fm',
        models_dict_path=fm_models_dict_path,
        device='cuda'
    )
    kb_model = GenNeuronIndexModel_ViTImageClassification(
        name='vit_b_16_kb',
        models_dict_path=kb_models_dict_path,
        device='cuda'
    )
    
    # 3. Algorithm
    from methods.gen_neuron_index import GenNeuronIndexAlg
    alg = GenNeuronIndexAlg(
        models={
            'fm': fm_model,
            'kb': kb_model
        },
        res_save_dir=get_res_save_dir(__file__, sys.argv[1])
    )
    
    # 4. Run!
    qkv_layers = [[f'vit.encoder.layer.{i}.attention.attention.{k}' for k in ['query', 'key', 'value']] for i in range(12)]
    proj_layers = [f'vit.encoder.layer.{i}.attention.output.dense' for i in range(12)]
    ff1_layers = [f'vit.encoder.layer.{i}.intermediate.dense' for i in range(12)]
    ff2_layers = [f'vit.encoder.layer.{i}.output.dense' for i in range(12)]
    
    alg.run(
        scenario=scenario, 
        hyps={
            'launch_tbboard': True,
                
            'example_sample': torch.rand((1, 3, 224, 224)),
            
            'qkv_layers_name': qkv_layers,
            'proj_layers_name': proj_layers,
            'ff1_layers_name': ff1_layers,
            'ff2_layers_name': ff2_layers,
            
            'fbs_r': 8,
            'min_sparsity': 0.,
            'max_sparsity': 0.9,
            'sparsity_loss_alpha': 1e-8,
            'neuron_index_loss_alpha': 1e-4,

            'train_batch_size': 128,
            'val_batch_size': 512,
            'num_workers': 16,
            'optimizer': 'AdamW',
            'optimizer_args': {'lr': 3e-4, 'momentum': 0.9, 'weight_decay': 1e-5},
            'neuron_index_optimizer': 'AdamW',
            'neuron_index_optimizer_args': {'lr': 3e-3, 'betas': [0.9, 0.999], 'weight_decay': 0.1},
            'scheduler': 'StepLR',
            'scheduler_args': {'step_size': 30000, 'gamma': 0.1},
            'num_iters': 80000,
            'val_freq': 2000,
            
            'fp16': True
        }
    )
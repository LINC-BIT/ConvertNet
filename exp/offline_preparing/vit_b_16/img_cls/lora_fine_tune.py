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
    from transformers import AutoModelForImageClassification, ViTForImageClassification
    model = AutoModelForImageClassification.from_pretrained('google/vit-base-patch16-224')
    model.classifier = nn.Linear(768, scenario.num_classes)
    models_dict_path = save_models_dict_for_init({
        'main': model
    }, __file__)
    from exp.online_retraining.vit_b_16.img_cls.model_impl import LoRAFineTuningModel_ViTImageClassification
    model = LoRAFineTuningModel_ViTImageClassification(
        name='vit_b_16',
        models_dict_path=models_dict_path,
        device='cuda'
    )
    
    # 3. Algorithm
    from methods.lora_ft import LoRATuningAlg
    alg = LoRATuningAlg(
        models={
            'main': model
        },
        res_save_dir=get_res_save_dir(__file__, sys.argv[1])
    )
    
    # 4. Run!
    qkv_layers = []
    for block_i in range(12):
        l = []
        for layer_name in ['attention.attention.query', 'attention.attention.key', 
                           'attention.attention.value']:
            l += [f'vit.encoder.layer.{block_i}.{layer_name}']
        qkv_layers += [l]
            
    alg.run(
        scenario=scenario, 
        hyps={
            'launch_tbboard': True,
                
            'example_sample': torch.rand((1, 3, 224, 224)),
            
            'qkv_layers_name': qkv_layers,
            'lora_r': 8,

            'train_batch_size': 128,
            'val_batch_size': 512,
            'num_workers': 16,
            'optimizer': 'AdamW',
            'optimizer_args': {'lr': 3e-4, 'momentum': 0.9, 'weight_decay': 1e-5},
            'scheduler': 'StepLR',
            'scheduler_args': {'step_size': 4000, 'gamma': 0.1},
            'num_iters': 10000,
            'val_freq': 1000,
            
            'fp16': True # 18GB vs 23GB
        }
    )
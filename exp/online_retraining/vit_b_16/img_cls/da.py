from copy import deepcopy
import sys
from utils.dl.common.env import set_random_seed
set_random_seed(1)

from typing import List
from data.dataloader import build_dataloader
from data import Scenario

import torch
import sys
from torch import nn
from utils.common.file import ensure_dir
from utils.dl.common.model import LayerActivation, get_module, get_parameter
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
from utils.dl.common.loss import CrossEntropyLossSoft
import torch.nn.functional as F
from utils.dl.common.env import create_tbwriter
import os
import shutil
from utils.common.log import logger
from utils.common.data_record import write_json
from methods.base.alg import BaseAlg
from methods.base.model import BaseModel


def da_exp(app_name: str,
           scenario: Scenario, 
           da_alg: BaseAlg, 
           da_alg_hyp: dict,
           da_model: BaseModel,
           device,
           __entry_file__,
           use_entry_model_in_new_dist,
           tag=None):
    
    log_dir = get_res_save_dir(__entry_file__, tag=tag)
    tb_writer = create_tbwriter(os.path.join(log_dir, 'tb_log'), True)
    res = []
    global_avg_after_acc = 0.
    global_iter = 0
    
    if use_entry_model_in_new_dist:
        entry_da_model = da_model
    
    for domain_index, _ in enumerate(scenario.target_domains_order):
        # if domain_index == 0:
        #     scenario.next_domain()
        #     continue
        
        cur_target_domain_name = scenario.target_domains_order[scenario.cur_domain_index]
        if cur_target_domain_name in da_alg_hyp:
            da_alg_hyp = da_alg_hyp[cur_target_domain_name]
            logger.info(f'use dataset-specific hyps')
        
        logger.info(f'----- domain {domain_index}: {cur_target_domain_name}')
        
        if use_entry_model_in_new_dist:
            logger.info(f'use entry model in new dist')
            da_model = deepcopy(entry_da_model)
        else:
            logger.info(f'reuse previously retrained model')
        
        da_metrics, _ = da_alg(
            {'main': da_model}, 
            os.path.join(log_dir, f'{domain_index:02d}')
        ).run(scenario, da_alg_hyp)
        
        if domain_index > 0:
            shutil.rmtree(os.path.join(log_dir, f'{domain_index:02d}/backup_codes'))
        
        accs = da_metrics['accs']
        before_acc = accs[0]['acc']
        after_acc = accs[-1]['acc']
        
        tb_writer.add_scalars(f'accs', dict(before=before_acc, after=after_acc), domain_index)
        
        for _acc in accs:
            tb_writer.add_scalar('total_acc', _acc['acc'], _acc['iter'] + global_iter)
        global_iter += _acc['iter'] + 1
        
        scenario.next_domain()
        
        logger.info(f"----- domain {domain_index}, acc: {before_acc:.4f} -> "
                    f"{after_acc:.4f}")
        
        global_avg_after_acc += after_acc
        cur_res = da_metrics
        res += [cur_res]
        write_json(os.path.join(log_dir, 'res.json'), res, backup=False)
        
        # if domain_index == 0:
        #     print(f'DEBUG: retraining in only second domain')
        #     exit()

    global_avg_after_acc /= (domain_index + 1)
    logger.info(f'----- final metric: {global_avg_after_acc:.4f}')
    write_json(os.path.join(log_dir, f'res_{global_avg_after_acc:.4f}.json'), res, backup=False)
    
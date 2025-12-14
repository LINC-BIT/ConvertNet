from typing import Dict, List, Optional, Type, Union
from ..datasets.ab_dataset import ABDataset
# from benchmark.data.visualize import visualize_classes_in_object_detection
# from benchmark.scenario.val_domain_shift import get_val_domain_shift_transform
from ..dataset import get_dataset
import copy
from torchvision.transforms import Compose
from ..datasets.registery import static_dataset_registery
from ..build.scenario import Scenario as DAScenario
from copy import deepcopy
from utils.common.log import logger
import random
from .scenario import _ABDatasetMetaInfo, Scenario
        
        
def _check(source_datasets_meta_info: List[_ABDatasetMetaInfo], target_datasets_meta_info: List[_ABDatasetMetaInfo]):
    # requirements for simplity
    # 1. no same class in source datasets
    
    source_datasets_class = [i.classes for i in source_datasets_meta_info]
    for ci1, c1 in enumerate(source_datasets_class):
        for ci2, c2 in enumerate(source_datasets_class):
            if ci1 == ci2:
                continue
            
            c1_name = source_datasets_meta_info[ci1].name
            c2_name = source_datasets_meta_info[ci2].name
            intersection = set(c1).intersection(set(c2))
            assert len(intersection) == 0, f'{c1_name} has intersection with {c2_name}: {intersection}'

    
def build_cl_scenario(
    da_scenario: DAScenario,
    target_datasets_name: List[str],
    num_classes_per_task: int,
    max_num_tasks: int,
    data_dirs,
    sanity_check=False
):
    config = deepcopy(locals())
    
    source_datasets_idx_map = {}
    source_class_idx_max = 0
    
    for sd in da_scenario.config['source_datasets_name']:
        da_scenario_idx_map = None
        for k, v in da_scenario.all_datasets_idx_map.items():
            if k.startswith(sd):
                da_scenario_idx_map = v
                break
            
        source_datasets_idx_map[sd] = da_scenario_idx_map
        source_class_idx_max = max(source_class_idx_max, max(list(da_scenario_idx_map.values())))
    
    
    target_class_idx_start = source_class_idx_max + 1
    
    target_datasets_meta_info = [_ABDatasetMetaInfo(d, *static_dataset_registery[d][1:], None, None) for d in target_datasets_name]
    
    task_datasets_seq = []
    
    num_tasks_per_dataset = {}
    
    for td_info_i, td_info in enumerate(target_datasets_meta_info):
        
        if td_info_i >= 1:
            for _td_info_i, _td_info in enumerate(target_datasets_meta_info[0: td_info_i]):
                if _td_info.name == td_info.name:
                    # print(111)
                    # class_idx_offset = sum([len(t.classes) for t in target_datasets_meta_info[0: td_info_i]])
                    print(len(task_datasets_seq))
                    
                    task_index_offset = sum([v if __i < _td_info_i else 0 for __i, v in enumerate(num_tasks_per_dataset.values())])
                    
                    task_datasets_seq += task_datasets_seq[task_index_offset: task_index_offset + num_tasks_per_dataset[_td_info_i]]
                    print(len(task_datasets_seq))
                    break
            continue
        
        td_classes = td_info.classes
        num_tasks_per_dataset[td_info_i] = 0
        
        for ci in range(0, len(td_classes), num_classes_per_task):
            task_i = ci // num_classes_per_task
            task_datasets_seq += [_ABDatasetMetaInfo(
                f'{td_info.name}|task-{task_i}|ci-{ci}-{ci + num_classes_per_task - 1}',
                td_classes[ci: ci + num_classes_per_task],
                td_info.task_type,
                td_info.object_type,
                td_info.class_aliases,
                td_info.shift_type,
                
                td_classes[:ci] + td_classes[ci + num_classes_per_task: ],
                {cii: cii + target_class_idx_start for cii in range(ci, ci + num_classes_per_task)}
            )]
            num_tasks_per_dataset[td_info_i] += 1
            
        if ci + num_classes_per_task < len(td_classes) - 1:
            task_datasets_seq += [_ABDatasetMetaInfo(
                f'{td_info.name}-task-{task_i + 1}|ci-{ci}-{ci + num_classes_per_task - 1}',
                td_classes[ci: len(td_classes)],
                td_info.task_type,
                td_info.object_type,
                td_info.class_aliases,
                td_info.shift_type,
                
                td_classes[:ci],
                {cii: cii + target_class_idx_start for cii in range(ci, len(td_classes))}
            )]
            num_tasks_per_dataset[td_info_i] += 1
                
        target_class_idx_start += len(td_classes)
    
    if len(task_datasets_seq) < max_num_tasks:
        print(len(task_datasets_seq), max_num_tasks)
        raise RuntimeError()
    
    task_datasets_seq = task_datasets_seq[0: max_num_tasks]
    target_class_idx_start = max([max(list(td.idx_map.values())) + 1 for td in task_datasets_seq])
    
    scenario = Scenario(config, task_datasets_seq, target_class_idx_start, source_class_idx_max + 1, data_dirs)
    
    if sanity_check:
        selected_tasks_index = []
        for task_index, _ in enumerate(scenario.target_tasks_order):
            cur_datasets = scenario.get_cur_task_train_datasets()
            
            if len(cur_datasets) < 300:
                # empty_tasks_index += [task_index]
                # while True:
                    # replaced_task_index = random.randint(0, task_index - 1) # ensure no random
                replaced_task_index = task_index // 2
                assert replaced_task_index != task_index
                while replaced_task_index in selected_tasks_index:
                    replaced_task_index += 1
                    
                task_datasets_seq[task_index] = deepcopy(task_datasets_seq[replaced_task_index])
                selected_tasks_index += [replaced_task_index]
                
                logger.warning(f'replace {task_index}-th task with {replaced_task_index}-th task')
            
            # print(task_index, [t.name for t in task_datasets_seq])
            
            scenario.next_task()
            
        # print([t.name for t in task_datasets_seq])
            
        if len(selected_tasks_index) > 0:
            target_class_idx_start = max([max(list(td.idx_map.values())) + 1 for td in task_datasets_seq])
            scenario = Scenario(config, task_datasets_seq, target_class_idx_start, source_class_idx_max + 1, data_dirs)
            
            for task_index, _ in enumerate(scenario.target_tasks_order):
                cur_datasets = scenario.get_cur_task_train_datasets()
                logger.info(f'task {task_index}, len {len(cur_datasets)}')
                assert len(cur_datasets) > 0
                
                scenario.next_task()
                
            scenario = Scenario(config, task_datasets_seq, target_class_idx_start, source_class_idx_max + 1, data_dirs)

    return scenario
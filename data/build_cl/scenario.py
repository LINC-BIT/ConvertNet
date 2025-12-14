import enum
from functools import reduce
from typing import Dict, List, Tuple
import numpy as np
import copy
from utils.common.log import logger
from ..datasets.ab_dataset import ABDataset
from ..dataloader import FastDataLoader, InfiniteDataLoader, build_dataloader
from data import get_dataset, MergedDataset, Scenario as DAScenario


class _ABDatasetMetaInfo:
    def __init__(self, name, classes, task_type, object_type, class_aliases, shift_type, ignore_classes, idx_map):
        self.name = name
        self.classes = classes
        self.class_aliases = class_aliases
        self.shift_type = shift_type
        self.task_type = task_type
        self.object_type = object_type
        
        self.ignore_classes = ignore_classes
        self.idx_map = idx_map
        
    def __repr__(self) -> str:
        return f'({self.name}, {self.classes}, {self.idx_map})'


class Scenario:
    def __init__(self, config, target_datasets_info: List[_ABDatasetMetaInfo], num_classes: int, num_source_classes: int, data_dirs):
        self.config = config 
        self.target_datasets_info = target_datasets_info
        self.num_classes = num_classes
        self.cur_task_index = 0
        self.num_source_classes = num_source_classes
        self.cur_class_offset = num_source_classes
        self.data_dirs = data_dirs
        
        self.target_tasks_order = [i.name for i in self.target_datasets_info]
        self.num_tasks_to_be_learn = sum([len(i.classes) for i in target_datasets_info])

        logger.info(f'[scenario build] # classes: {num_classes}, # tasks to be learnt: {len(target_datasets_info)}, '
                    f'# classes per task: {config["num_classes_per_task"]}')

    def to_json(self):
        config = copy.deepcopy(self.config)
        config['da_scenario'] = config['da_scenario'].to_json()
        target_datasets_info = [str(i) for i in self.target_datasets_info]
        return dict(
            config=config, target_datasets_info=target_datasets_info,
            num_classes=self.num_classes
        )
        
    def __str__(self):
        return f'Scenario({self.to_json()})'
    
    def get_cur_class_offset(self):
        return self.cur_class_offset
    
    def get_cur_num_class(self):
        return len(self.target_datasets_info[self.cur_task_index].classes)
    
    def get_nc_per_task(self):
        return len(self.target_datasets_info[0].classes)
    
    def next_task(self):
        self.cur_class_offset += len(self.target_datasets_info[self.cur_task_index].classes)
        self.cur_task_index += 1
        
        print(f'now, cur task: {self.cur_task_index}, cur_class_offset: {self.cur_class_offset}')
        
    def get_cur_task_datasets(self):
        dataset_info = self.target_datasets_info[self.cur_task_index]
        dataset_name = dataset_info.name.split('|')[0]
        # print()
        
        # source_datasets_info = []
        
        res ={ **{split: get_dataset(dataset_name=dataset_name, 
                    root_dir=self.data_dirs[dataset_name], 
                    split=split, 
                    transform=None, 
                    ignore_classes=dataset_info.ignore_classes, 
                    idx_map=dataset_info.idx_map) for split in ['train']}, 
              
              **{split: MergedDataset([get_dataset(dataset_name=dataset_name, 
                    root_dir=self.data_dirs[dataset_name], 
                    split=split, 
                    transform=None, 
                    ignore_classes=di.ignore_classes, 
                    idx_map=di.idx_map) for di in self.target_datasets_info[0: self.cur_task_index + 1]]) 
                 for split in ['val', 'test']}
        }
        
        # if len(res['train']) < 200 or len(res['val']) < 200 or len(res['test']) < 200:
        #     return None
            
        
        if len(res['train']) < 1000:
            res['train'] = MergedDataset([res['train']] * 5)
            logger.info('aug train dataset')
        if len(res['val']) < 1000:
            res['val'] = MergedDataset(res['val'].datasets * 5)
            logger.info('aug val dataset')
        if len(res['test']) < 1000:
            res['test'] = MergedDataset(res['test'].datasets * 5)
            logger.info('aug test dataset')
        # da_scenario: DAScenario = self.config['da_scenario']
        # offline_datasets = da_scenario.get_offline_datasets()
        
        for k, v in res.items():
            logger.info(f'{k} dataset: {len(v)}')
        
        # new_val_datasets = [
        #     *[d['val'] for d in offline_datasets.values()],
        #     res['val']
        # ]
        # res['val'] = MergedDataset(new_val_datasets)
        
        # new_test_datasets = [
        #     *[d['test'] for d in offline_datasets.values()],
        #     res['test']
        # ]
        # res['test'] = MergedDataset(new_test_datasets)
        
        return res
    
    def get_cur_task_train_datasets(self):
        dataset_info = self.target_datasets_info[self.cur_task_index]
        dataset_name = dataset_info.name.split('|')[0]
        # print()
        
        # source_datasets_info = []
        
        res = get_dataset(dataset_name=dataset_name, 
                    root_dir=self.data_dirs[dataset_name], 
                    split='train', 
                    transform=None, 
                    ignore_classes=dataset_info.ignore_classes, 
                    idx_map=dataset_info.idx_map)
        
        return res
    
    def get_online_cur_task_samples_for_training(self, num_samples):
        dataset = self.get_cur_task_datasets()
        dataset = dataset['train']
        return next(iter(build_dataloader(dataset, num_samples, 0, True, None)))[0]
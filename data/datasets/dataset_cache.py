from typing import List, Optional, Dict
import os
import torch
from utils.common.log import logger
import hashlib


def get_dataset_cache_path(root_dir: str, 
                         classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
    
    def _hash(o):
        if isinstance(o, list):
            o = sorted(o)
        elif isinstance(o, dict):
            o = {k: o[k] for k in sorted(o)}
        elif isinstance(o, set):
            o = sorted(list(o))
        # else:
        #     print(type(o))
        
        obj = hashlib.md5()
        obj.update(str(o).encode('utf-8'))
        return obj.hexdigest()
    
    cache_key = _hash(f'zql_data_{_hash(root_dir)}_{_hash(classes)}_{_hash(ignore_classes)}_{_hash(idx_map)}.cache')
    
    # print(root_dir, classes, ignore_classes, idx_map)
    # print('cache key', cache_key)
    
    cache_file_path = os.path.join('/tmp', f'./zql_data_cache_{cache_key}.cache')
    return cache_file_path


def cache_dataset_status(status, cache_file_path, dataset_name):
    logger.debug(f'cache dataset status: {dataset_name}')
    torch.save(status, cache_file_path)
    
def read_cached_dataset_status(cache_file_path, dataset_name):
    logger.debug(f'read dataset cache: {dataset_name}')
    return torch.load(cache_file_path)

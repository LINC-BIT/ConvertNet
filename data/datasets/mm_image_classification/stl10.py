from ..data_aug import pil_image_to_tensor
from ..ab_dataset import ABDataset
from ..dataset_split import train_val_test_split
from ..dataset_cache import get_dataset_cache_path, read_cached_dataset_status, cache_dataset_status
from .mm_image_folder import MMImageFolder
from ..dataset_split import train_val_split
from torchvision.datasets import CIFAR10 as RawCIFAR10
import os
from typing import Dict, List, Optional
from torchvision.transforms import Compose
from utils.common.others import HiddenPrints
import numpy as np
from ..registery import dataset_register

    
    
@dataset_register(
    name='STL10-MM', 
    classes=['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'], 
    task_type='MM Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class STL10MM(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = pil_image_to_tensor(224)
            self.transform = transform
        
        # root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        dataset = MMImageFolder(root_dir, [c for c in classes if c not in ignore_classes], transform=transform)
        
        cache_file_path = get_dataset_cache_path(root_dir, classes, ignore_classes, idx_map)
        if os.path.exists(cache_file_path):
            dataset.samples = read_cached_dataset_status(cache_file_path, 'STL10-' + split)
        else:
            if len(ignore_classes) > 0:
                ignore_classes_idx = [classes.index(c) for c in ignore_classes]
                dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]    
            if idx_map is not None:
                dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]
            
            cache_dataset_status(dataset.samples, cache_file_path, 'STL10-' + split)
        
        dataset = train_val_test_split(dataset, split)
        return dataset
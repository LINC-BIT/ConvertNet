from ..data_aug import cifar_like_image_train_aug, cifar_like_image_test_aug
from ..ab_dataset import ABDataset
from ..dataset_split import train_val_test_split
from torchvision.datasets import ImageFolder
from typing import Dict, List, Optional
from torchvision.transforms import Compose
from utils.common.others import HiddenPrints

from ..registery import dataset_register


@dataset_register(
    name='STL10-single', 
    classes=['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'], 
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class STL10Single(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = cifar_like_image_train_aug() if split == 'train' else cifar_like_image_test_aug()
            self.transform = transform
        
        dataset = ImageFolder(root_dir, transform=transform)
        
        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]
            
        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]
        
        dataset = train_val_test_split(dataset, split)
        return dataset

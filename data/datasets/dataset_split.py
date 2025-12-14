import torch
import os
import numpy as np
from .ab_dataset import ABDataset


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys

    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0, transform=None):
    
    if isinstance(dataset, ABDataset):
        if dataset.task_type == 'Object Detection':
            return split_dataset_det(dataset, n, seed)
        if dataset.task_type == 'MM Object Detection':
            return split_dataset_det_mm(dataset, n, seed, transform=transform)
    
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert n <= len(dataset), f'{n}_{len(dataset)}'

    cache_p = f'{n}_{seed}_{len(dataset)}'
    cache_p = os.path.join(os.path.expanduser(
        '~'), '.domain_benchmark_split_dataset_cache_' + str(cache_p))
    if os.path.exists(cache_p):
        keys_1, keys_2 = torch.load(cache_p)
    else:
        keys = list(range(len(dataset)))
        np.random.RandomState(seed).shuffle(keys)
        keys_1 = keys[:n]
        keys_2 = keys[n:]
        torch.save((keys_1, keys_2), cache_p)
    
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


def train_val_split(dataset, split, rate=0.8):
    assert split in ['train', 'val']
    if split == 'train':
        return split_dataset(dataset, int(len(dataset) * rate))[0]
    else:
        return split_dataset(dataset, int(len(dataset) * rate))[1]

    
def train_val_test_split(dataset, split):
    assert split in ['train', 'val', 'test']

    train_set, test_set = split_dataset(dataset, int(len(dataset) * 0.8))
    train_set, val_set = split_dataset(train_set, int(len(train_set) * 0.8))
    
    return {'train': train_set, 'val': val_set, 'test': test_set}[split]


def split_dataset_det(dataset: ABDataset, n, seed=0):
    coco_ann_json_path = dataset.ann_json_file_path_for_split
    from .object_detection.yolox_data_util.api import coco_split, get_default_yolox_coco_dataset
    split_coco_ann_json_path = coco_split(coco_ann_json_path, ratio=n / len(dataset))[0]
    # print(n, len(dataset))
    return get_default_yolox_coco_dataset(dataset.root_dir, split_coco_ann_json_path, train=dataset.split == 'train'), None

def split_dataset_det_mm(dataset: ABDataset, n, seed=0, transform=None):
    coco_ann_json_path = dataset.ann_json_file_path_for_split
    from .object_detection.yolox_data_util.api import coco_split, get_yolox_coco_dataset_with_caption
    split_coco_ann_json_path = coco_split(coco_ann_json_path, ratio=n / len(dataset))[0]
    # print(n, len(dataset))
    return get_yolox_coco_dataset_with_caption(dataset.root_dir, split_coco_ann_json_path, transform=transform, train=dataset.split == 'train', classes=dataset.classes), None
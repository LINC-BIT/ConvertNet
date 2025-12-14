# Now this only supports en-de for T5 model
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from typing import Dict, List, Optional, Any
from utils.common.data_record import read_json
from itertools import chain
import random
import json
from datasets import load_dataset
from transformers import T5Tokenizer
# os.environ['t5_path'] = '/data/zql/concept-drift-in-edge-projects/UniversalElasticNet/new_impl/nlp/t5/translation/t5-small'

class News_commentary_taskbase(Dataset):
    def __init__(self, root_dir: str, split: str, transform: Any, 
                 classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]], type: str = 'en-de'):
        
        assert transform is None
        if type == 'en-de':
            self.task_prefix = "translate English to German: "
            root_dir = os.path.join(root_dir, 'de-en')
            
        self.type = type
        rate = 0.95
        self.tokenizer = T5Tokenizer.from_pretrained(os.environ['t5_path'])
        self.idx_map = []
        self.ignore_classes = []
        self.max_length = 512  # 设置文本的最大长度
        self.split = split
        if split == 'test':
            split = 'val'
        json_file_path = os.path.join(root_dir, f'{split}.json')
        if not os.path.exists(json_file_path): 
            tmp = os.path.join(root_dir, 'train-00000-of-00001.parquet')
            dataset = load_dataset("parquet", data_files=tmp)
            pairs = []
            for line in dataset['train']:
                pairs.append(line['translation'])
            
            random.shuffle(pairs)
            train_pairs = pairs[:int(len(pairs) * rate)]
            test_pairs = pairs[int(len(pairs) * rate):]
            train_file_path = os.path.join(root_dir, f'train.json')
            test_file_path = os.path.join(root_dir, f'val.json')
            with open(train_file_path, 'w') as f:
                json.dump(train_pairs, f)
            with open(test_file_path, 'w') as f:
                json.dump(test_pairs, f)
            
        self.pairs = read_json(json_file_path)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lang1_name, lang2_name = self.type.split('-')
        bos, eos, pad, sep = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, self.tokenizer.sep_token_id
        
        inputs = self.tokenizer.encode_plus(self.task_prefix + self.pairs[idx][lang1_name], padding = 'max_length', truncation=True, max_length = self.max_length)
        output = self.tokenizer.encode(self.pairs[idx][lang2_name], truncation=True)
        if len(output) < self.max_length:
            if self.split == 'train':
                output += [-100] * (self.max_length - len(output))
            else:
                output += [pad] * (self.max_length - len(output))
        x = {
            "input_ids": torch.tensor(inputs['input_ids']),
            "attention_mask": torch.tensor(inputs['attention_mask']),
            "labels": torch.tensor(output),
            'return_dict': True
        }
        return x

from ..ab_dataset import ABDataset
from ..registery import dataset_register

@dataset_register(
    name='News_commentary_de_en', 
    classes=['None'], 
    task_type='Translation',
    object_type=None,
    class_aliases=[],
    shift_type=None
)
class News_commentary(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return News_commentary_taskbase(root_dir, split, transform, classes, ignore_classes, idx_map)

# a = News_commentary_taskbase('/data/zql/datasets/news_commentary', 'train', None, None, None, None)
# a.__getitem__(0)
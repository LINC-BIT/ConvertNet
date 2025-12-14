import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from typing import Dict, List, Optional, Any
from utils.common.data_record import read_json
# from ..sentiment_classification.global_bert_tokenizer import get_tokenizer

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b')
# print(tokenizer.config)

# TAGS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB', '#', '$', "''", '(', ')', ',', '.', ':']
# TAGS = ['POS', "''", 'UH', 'TO', 'PDT', 'SYM', 'PRP', 'CD', 'RP', 'EX', ':', 'WDT', 'JJ', 'NN', 'JJS', 'LS', '(', 'WRB', 'WP$', 'VBZ', ')', '$', 'NNPS', 'WP', 'CC', 'VBP', 'IN', 'FW', 'RBR', 'DT', 'JJR', 'VBD', ',', 'NNP', '#', 'VB', 'RBS', 'NNS', 'VBN', 'RB', 'MD', 'PRP$', '.', 'VBG']
# TAGS = ['RBS', 'TO', 'WP$', 'UH', "''", 'JJ', 'DT', 'NN', 'SYM', ',', 'NNS', 'VB', 'VBG', 'MD', 'VBZ', 'CD', 'PDT', 'PRP$', '#', ')', '$', 'RBR', 'JJR', '(', 'VBD', 'VBN', 'WDT', 'WRB', 'CC', 'PRP', 'VBP', 'IN', 'NNP', 'POS', 'WP', 'JJS', 'RP', 'RB', 'NNPS', 'LS', 'EX', '.', 'FW', ':']

TAGS = ['JJ', 'JJR', "''", 'DT', 'CC', 'VBG', 'MD', 'WP', 'RBS', 'NNPS', 'EX', 'PDT', '``', 
        'PRP$', 'VBD', 'NNP', ')', 'RP', 'WDT', 'UH', '$', 'WRB', ',', '(', 'IN', 'VBP', 'VBN', 
        'VBZ', 'POS', 'NNS', 'PRP', '#', 'JJS', '.', 'NN', 'SYM', 'TO', 'VB', 'LS', 'WP$', 'RB', 'FW', 'CD', 'RBR', ':']

# 自定义数据集类
class UniversalASC19DomainsTokenClsDataset(Dataset):

    def __init__(self, root_dir: str, split: str, transform: Any, 
                 classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        assert transform is None
        
        self.tokenizer = tokenizer # 传入tokenizer对象
        # tokenizer.pad_token = tokenizer.eos_token
        
        self.srcs = []
        self.tgts = []
          # 设置文本的最大长度

        json_file_path = os.path.join(root_dir, f'train.json.token_cls_data.10000' if split == "train" else 'val.json.token_cls_data.2000')
        anns = read_json(json_file_path)
        
        # label_map = {'-': 0, '+': 1, 'negative': 0, 'positive': 1}
        
        # ignore_cls_indexes = [classes.index(c) for c in ignore_classes]
        # max_length = 0
        # for info in anns:
        #     max_length = max(len(info['sentence']), max_length)
        self.max_length = 256
        # print(f'seq_max_length: {self.max_length}')
        
        for info in anns:
            assert len(info['tags']) < self.max_length, f'{len(info["tags"])}'
            
            self.srcs += [info['sentence']]
            
            label = [TAGS.index(t) for t in info['tags']]
            label = label + [-100] * (self.max_length - len(label))
            self.tgts += [label]

    def __len__(self):
        return len(self.srcs)

    def __getitem__(self, idx):
        text = self.srcs[idx]
        label = self.tgts[idx]
        
        encoded_input = self.tokenizer.encode_plus(
            text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        x = {key: tensor.squeeze(0) for key, tensor in encoded_input.items()}
        # print(x['input_ids'].size())
        x['return_dict'] = False
        
        return x, torch.tensor(label)


from ..ab_dataset import ABDataset
from ..registery import dataset_register

@dataset_register(
    name='News-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class NewsTokenCls(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
    
@dataset_register(
    name='Opus-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class OpusTokenCls(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
    
@dataset_register(
    name='WMT14-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class WMT14TokenCls(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
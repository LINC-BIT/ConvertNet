import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from typing import Dict, List, Optional, Any
from utils.common.data_record import read_json
# from .global_bert_tokenizer import get_tokenizer

from transformers import GPT2Tokenizer
# gpt_neo_series_id = '1.3B_ckpt'
# os.environ['gpt_neo_series_id'] = gpt_neo_series_id

class AlpacaBase(Dataset):
    def __init__(self, root_dir: str, split: str, transform: Any, 
                 classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        assert transform is None
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(f'experiments/elasticdnn/gpt_neo/{os.environ["gpt_neo_series_id"]}') # 传入tokenizer对象
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.texts = []
        self.labels = []
        self.idx_map = []
        self.ignore_classes = []
        self.max_length = 768  # 设置文本的最大长度

        json_file_path = os.path.join(root_dir, f'data.json')
        anns = read_json(json_file_path)
        
        for v in anns:
            if len(v['input']) != 0:
                txt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{v['instruction']}

### Input:
{v['input']}

### Response:
"""
            else:
                txt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{v['instruction']}

### Response:
"""
            self.texts.append(txt)
            self.labels.append(v['output'])

    def __len__(self):
        return len(self.texts)

    def setSplit(self, split):
        self.split = split

    def __getitem__(self, idx):
        if self.split == 'val':
            self.tokenizer.padding_side = "left"
            text = self.texts[idx]
            label = self.labels[idx]
            
            encoded_input = self.tokenizer.encode_plus(
                text, max_length=self.max_length - 128, padding="max_length", truncation=True, return_tensors="pt"
            )
            label = self.tokenizer.encode_plus(
                label, max_length=128, padding="max_length", truncation=True, return_tensors="pt"
            )
            x = {key: tensor.squeeze(0) for key, tensor in encoded_input.items()}
            y = label['input_ids'].squeeze(0)
            x['labels'] = y
            x['len'] = len(text)
            x['return_dict'] = False
            return x
        else:
            self.tokenizer.padding_side = "right"
            text = self.texts[idx] + self.labels[idx] + self.tokenizer.eos_token
        
            encoded_input = self.tokenizer.encode_plus(
                text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            x = {key: tensor.squeeze(0) for key, tensor in encoded_input.items()}
            x['labels'] = x['input_ids'].clone()
            x['labels'][x['labels'] == self.tokenizer.pad_token_id] = -100
            for i, v in enumerate(x['labels']):
                if v == -100:
                    x['labels'][i] = self.tokenizer.eos_token_id
                    break
            x['return_dict'] = False
            return x

from ..ab_dataset import ABDataset
from ..registery import dataset_register

@dataset_register(
    name='Alpaca', 
    classes=[], 
    task_type='Text Generation',
    object_type=None,
    class_aliases=[],
    shift_type=None
)
class Alpaca(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return AlpacaBase(root_dir, split, transform, classes, ignore_classes, idx_map)
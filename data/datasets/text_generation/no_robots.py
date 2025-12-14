from datasets import load_dataset
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from typing import Dict, List, Optional, Any
from utils.common.data_record import read_json
from itertools import chain
# from .global_bert_tokenizer import get_tokenizer

from transformers import GPT2Tokenizer
# gpt_neo_series_id = '1.3B_ckpt'
# os.environ['gpt_neo_series_id'] = gpt_neo_series_id
class No_Robotsbase(Dataset):
    def __init__(self, root_dir: str, split: str, transform: Any, 
                 classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        assert transform is None
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(f'experiments/elasticdnn/gpt_neo/{os.environ["gpt_neo_series_id"]}')
        special_tokens = {"pad_token":"<|pad|>"}#, "sep_token":"<|sep|>", "bos_token":"<|bos|>"}
        self.tokenizer.add_special_tokens(special_tokens)
        self.tokenizer.pad_token = "<|pad|>" # 传入tokenizer对象
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.sep_token = self.tokenizer.eos_token
        self.msgs = []
        self.idx_map = []
        self.ignore_classes = []
        self.max_length = 768  # 设置文本的最大长度
        self.split = split

        dataset = load_dataset(root_dir, split=('test' if split == 'val' else split))
        for line in dataset:
            for i, msg in enumerate(line['messages']):
                if msg['role'] == 'assistant':
                    self.msgs.append(line['messages'][:i + 1])
        if self.split == 'val':
            self.msgs = self.msgs[:100]
        

    def __len__(self):
        return len(self.msgs)

    def __getitem__(self, idx):
        bos, eos, pad, sep = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, self.tokenizer.sep_token_id
        role_tti = {'user': 0, 'assistant': 1, 'system': 2, 'pad': 3}
        role_sgn = {'user': "Q: ", 'assistant': "A: "}
        context_list = [con['content'] for con in self.msgs[idx]]
        role_list = [con['role'] for con in self.msgs[idx]]
        if self.split == 'val':
            self.tokenizer.padding_side = "left"
            input_ids = []
            labels = []
            for id, utter in enumerate(context_list[:-1]):
                if role_list[id] == 'system':
                    tmp = self.tokenizer.encode(utter + '\n\n')
                else:
                    tmp = self.tokenizer.encode(role_sgn[role_list[id]]) + self.tokenizer.encode(utter + '\n\n')
                input_ids += tmp
            input_ids += self.tokenizer.encode(role_sgn[role_list[len(context_list) - 1]])
            if len(input_ids) > self.max_length - 128:
                return {'return_dict': True}
            leng = len(self.tokenizer.decode(input_ids))
            input_ids = [pad] * (self.max_length - 128 - len(input_ids)) + input_ids
            labels = self.tokenizer.encode(context_list[-1], max_length=128, padding="max_length", truncation=True)
            if len(labels) > 128:
                return {'return_dict': True}
            x = {
                "input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                'return_dict': True,
                'len': leng
            }
            return x
        else:
            self.tokenizer.padding_side = "right"
            target = context_list[-1]
            input_ids = []
            labels = []
            for id, utter in enumerate(context_list[:-1]):
                if role_list[id] == 'system':
                    tmp = self.tokenizer.encode(utter + '\n\n')
                else:
                    tmp = self.tokenizer.encode(role_sgn[role_list[id]]) + self.tokenizer.encode(utter + '\n\n')
                input_ids += tmp
            input_ids += self.tokenizer.encode(role_sgn[role_list[len(context_list) - 1]])
            labels = [-100] * len(input_ids) + self.tokenizer.encode(target) + [eos]
            # labels = input_ids + self.tokenizer.encode(target) + [eos]
            input_ids += self.tokenizer.encode(target) + [eos]
            
            # token_type_ids = [[role_tti[role_list[i]]] * (len(self.tokenizer.encode(utter)) + len(self.tokenizer.encode(role_sgn[role_list[i]]))) for i, utter in enumerate(context_list)]
            # token_type_ids += [[role_tti[role_list[-1]]]]
            # lm_labels = [[pad] * (len(list(chain(*input_ids))) - len(self.tokenizer.encode(target)) - 1)] + [self.tokenizer.encode(target)] + [eos]
            # input_ids = list(chain(*input_ids))
            if len(input_ids) > self.max_length:
                return {'return_dict': True}
            # token_type_ids = list(chain(*token_type_ids))
            attention_mask = [1] * len(input_ids) + [0] * (self.max_length - len(input_ids))
            # labels = [[-100] * (len(token_type_ids) - len(self.tokenizer.encode(target)) - 1)] + [self.tokenizer.encode(target)] + [[eos]]
            # labels = list(chain(*labels))
            # labels = input_ids.copy()
            labels += [-100] * (self.max_length - len(input_ids))
            input_ids += [pad] * (self.max_length - len(input_ids))
            # token_type_ids += [role_tti['pad']] * (self.max_length - len(token_type_ids))
            x = {
                "input_ids": torch.tensor(input_ids),
                # "token_type_ids": torch.tensor(token_type_ids),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
                'return_dict': True
            }
            return x

from ..ab_dataset import ABDataset
from ..registery import dataset_register

@dataset_register(
    name='No_robots', 
    classes=['None'], 
    task_type='Text Generation',
    object_type=None,
    class_aliases=[],
    shift_type=None
)
class No_Robots(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return No_Robotsbase(root_dir, split, transform, classes, ignore_classes, idx_map)
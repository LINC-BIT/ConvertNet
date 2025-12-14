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
import json
import random
from copy import deepcopy

from dnns.llama import AttentionPrunableLlamaForCausalLM, get_tokenizer
tokenizer = get_tokenizer()
# os.environ['llama_path'] = '/data/zql/concept-drift-in-edge-projects/UniversalElasticNet/new_impl/nlp/llama/text-generation/llama-68m'

class MultiWoz_base(Dataset):
    def __init__(self, root_dir: str, split: str, transform: Any, type: str, 
                 classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        assert transform is None
        
        self.tokenizer = tokenizer
        special_tokens = {"pad_token": "<pad>"}
        self.tokenizer.add_special_tokens(special_tokens)
        self.tokenizer.pad_token = "<pad>" # 传入tokenizer对象
        self.max_length = 768  # 设置文本的最大长度
        self.split = split

        if not os.path.exists(os.path.join(root_dir, 'train.json')):    
            with open(os.path.join(root_dir, 'data.json'), 'r') as f:
                datas = json.load(f)
            with open(os.path.join(root_dir, 'valListFile.txt'), 'r') as f:
                val_list = [line.strip() for line in f.readlines()]

            val_datas = []
            train_datas = []
            taxi_datas = []
            hotel_datas = []
            for key in datas:
                if key in val_list:
                    val_datas.append(datas[key])
                else:
                    if 'SNG' in key:
                        if len(datas[key]['goal']['taxi']) != 0 and len(taxi_datas) < 300:
                            taxi_datas.append(datas[key])
                        elif len(datas[key]['goal']['hotel']) != 0 and len(hotel_datas) < 300:
                            hotel_datas.append(datas[key])
                        else:
                            train_datas.append(datas[key])
                    else:
                        train_datas.append(datas[key])
            
            train_lines = [[t['text'].strip() for t in line['log']] for line in train_datas]
            val_lines = [[t['text'].strip() for t in line['log']] for line in val_datas]
            taxi_lines = [[t['text'].strip() for t in line['log']] for line in taxi_datas]
            hotel_lines = [[t['text'].strip() for t in line['log']] for line in hotel_datas]
            
            with open(os.path.join(root_dir, 'train.json'), 'w') as f:
                json.dump(train_lines, f)
            with open(os.path.join(root_dir, 'val.json'), 'w') as f:
                json.dump(val_lines, f)
            with open(os.path.join(root_dir, 'hotel.json'), 'w') as f:
                json.dump(hotel_lines, f)
            with open(os.path.join(root_dir, 'taxi.json'), 'w') as f:
                json.dump(taxi_lines, f)    
        
        self.msgs = []
        
        if type == 'mul':
            if split == 'train':
                with open(os.path.join(root_dir, 'train.json'), 'r') as f:
                    self.msgs = json.load(f)
            else:
                with open(os.path.join(root_dir, 'val.json'), 'r') as f:
                    tmp_msgs = json.load(f)
                for line in tmp_msgs:
                    for i, msg in enumerate(line):
                        if i % 2 == 1:
                            self.msgs.append(line[:i + 1])
                self.msgs = self.msgs[:1000]
        
        elif type == 'hotel':
            with open(os.path.join(root_dir, 'hotel.json'), 'r') as f:
                self.msgs = json.load(f)
            random.shuffle(self.msgs)
            if split == 'train':
                self.msgs = self.msgs
            else:
                tmp_msgs = deepcopy(self.msgs)
                for line in tmp_msgs:
                    for i, msg in enumerate(line):
                        if i % 2 == 1:
                            self.msgs.append(line[:i + 1])
        
        elif type == 'taxi':
            with open(os.path.join(root_dir, 'taxi.json'), 'r') as f:
                self.msgs = json.load(f)
            random.shuffle(self.msgs)
            if split == 'train':
                self.msgs = self.msgs
            else:
                tmp_msgs = deepcopy(self.msgs) 
                for line in tmp_msgs:
                    for i, msg in enumerate(line):
                        if i % 2 == 1:
                            self.msgs.append(line[:i + 1]) 
                
    def __len__(self):
        return len(self.msgs)

    def __getitem__(self, idx):
        bos, eos, pad, sep = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, self.tokenizer.sep_token_id
        role_tti = ['user', 'system', 'pad']
        role_sgn = {'user': "User:", 'system': "System:"}
        context_list = [con for con in self.msgs[idx]]
        role_list = [role_tti[id % 2] for id in range(len(self.msgs[idx]))]
        if self.split == 'val':
            self.tokenizer.padding_side = "left"
            input_ids = []
            labels = []
            for id, utter in enumerate(context_list[:-1]):
                head = self.tokenizer.encode(role_sgn[role_list[id]])[1:]
                tmp = self.tokenizer.encode(utter + '\n\n')[1:]
                input_ids += head + tmp
            input_ids = [bos] + input_ids + self.tokenizer.encode(role_sgn[role_list[len(context_list) - 1]])[1:]
            
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
            for id, utter in enumerate(context_list):
                head = self.tokenizer.encode(role_sgn[role_list[id]])[1:]
                tmp = self.tokenizer.encode(utter + '\n\n')[1:]
                input_ids += head + tmp
                if id % 2 == 0:
                    labels += [-100] * len(head + tmp)
                else:
                    labels += [-100] * len(head) + tmp
                
            input_ids = [bos] + input_ids + [eos]
            labels = [-100] + labels + [eos]
            # labels = [-100] * len(input_ids) + self.tokenizer.encode(target) + [eos]
            # labels = input_ids + self.tokenizer.encode(target) + [eos]
            # input_ids += self.tokenizer.encode(target) + [eos]
            
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
            # y = torch.LongTensor(1)
            return x


# tmp = MultiWoz_base('/data/zql/datasets/multiwoz', 'val', None, 'hotel', None, None, None)
# print(tmp.__getitem__(0))

from ..ab_dataset import ABDataset
from ..registery import dataset_register

@dataset_register(
    name='MultiWoz_MultiDomains', 
    classes=['None'], 
    task_type='Text Generation',
    object_type=None,
    class_aliases=[],
    shift_type=None
)
class MultiWoz(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return MultiWoz_base(root_dir, split, transform, 'mul', classes, ignore_classes, idx_map)
    
@dataset_register(
    name='MultiWoz_taxi', 
    classes=['None'], 
    task_type='Text Generation',
    object_type=None,
    class_aliases=[],
    shift_type=None
)
class MultiWoz_taxi(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return MultiWoz_base(root_dir, split, transform, 'taxi', classes, ignore_classes, idx_map)

@dataset_register(
    name='MultiWoz_hotel', 
    classes=['None'], 
    task_type='Text Generation',
    object_type=None,
    class_aliases=[],
    shift_type=None
)
class MultiWoz_hotel(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return MultiWoz_base(root_dir, split, transform, 'hotel', classes, ignore_classes, idx_map)


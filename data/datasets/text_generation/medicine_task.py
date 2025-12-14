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
import random
import json
# from .global_bert_tokenizer import get_tokenizer

from transformers import GPT2Tokenizer
# gpt_neo_series_id = '1.3B_ckpt'
# os.environ['gpt_neo_series_id'] = gpt_neo_series_id
class Medicine_taskbase(Dataset):
    def __init__(self, root_dir: str, split: str, transform: Any, 
                 classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        assert transform is None
        rate = 0.8
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
        json_file_path = os.path.join(root_dir, f'{split}.json')
        if not os.path.exists(json_file_path): 
            anns = read_json(os.path.join(root_dir, f'data.json'))
            random.shuffle(anns)
            train_anns = anns[:int(len(anns) * rate)]
            test_anns = anns[int(len(anns) * rate):]
            train_file_path = os.path.join(root_dir, f'train.json')
            test_file_path = os.path.join(root_dir, f'val.json')
            with open(train_file_path, 'w') as f:
                json.dump(train_anns, f)
            with open(test_file_path, 'w') as f:
                json.dump(test_anns, f)
            
        anns = read_json(json_file_path)
        self.questions = []
        self.answers = []
        
        for line in anns:
            quest = line['input']
            if 'Q: ' in quest:
                quest = quest.replace('Q: ', '')
                quest = quest.replace('A:', '')
            if 'Question: ' in quest:
                quest = quest.replace('Question: ', '')
                quest = quest.replace('Answer:', '')
            quest = quest.strip('\n') + "\nOptions:\n" + '\n'.join(line['options'])
            
            ans = line['options'][line['gold_index']]
            self.questions.append(quest)
            self.answers.append(ans)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        bos, eos, pad, sep = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, self.tokenizer.sep_token_id
        if self.split == 'val':
            self.tokenizer.padding_side = "left"
            input_ids = []
            labels = []
            input_ids = self.tokenizer.encode("Q: ") + self.tokenizer.encode(self.questions[idx] + '\n\n') + self.tokenizer.encode("A: ")
            if len(input_ids) > self.max_length - 128:
                return {'return_dict': True}
            leng = len(self.tokenizer.decode(input_ids))
            input_ids = [pad] * (self.max_length - 128 - len(input_ids)) + input_ids
            labels = self.tokenizer.encode(self.answers[idx], max_length=128, padding="max_length", truncation=True)
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
            input_ids = []
            labels = []
            input_ids = self.tokenizer.encode("Q: ") + self.tokenizer.encode(self.questions[idx] + '\n\n') + self.tokenizer.encode("A: ")
            labels = [-100] * len(input_ids) + self.tokenizer.encode(self.answers[idx]) + [eos]
            # labels = input_ids + self.tokenizer.encode(target) + [eos]
            input_ids += self.tokenizer.encode(self.answers[idx]) + [eos]
            
            if len(input_ids) > self.max_length:
                return {'return_dict': True}
            attention_mask = [1] * len(input_ids) + [0] * (self.max_length - len(input_ids))
            # labels = [[-100] * (len(token_type_ids) - len(self.tokenizer.encode(target)) - 1)] + [self.tokenizer.encode(target)] + [[eos]]
            labels += [-100] * (self.max_length - len(input_ids))
            input_ids += [pad] * (self.max_length - len(input_ids))
            x = {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
                'return_dict': True
            }
            return x

from ..ab_dataset import ABDataset
from ..registery import dataset_register

@dataset_register(
    name='Medicine_task', 
    classes=['None'], 
    task_type='Text Generation',
    object_type=None,
    class_aliases=[],
    shift_type=None
)
class Medicine_task(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return Medicine_taskbase(root_dir, split, transform, classes, ignore_classes, idx_map)

# a = Medicine_taskbase('/data/zql/datasets/medicine_task', 'train', None, None, None, None)
# a.__getitem__(0)
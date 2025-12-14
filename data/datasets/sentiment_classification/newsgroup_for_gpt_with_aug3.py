from sklearn.datasets import fetch_20newsgroups
# from pprint import pprint
# newsgroups_train = fetch_20newsgroups(subset='train')
# print(newsgroups_train.target_names)
# print(newsgroups_train['data'][0])
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
tokenizer = GPT2Tokenizer.from_pretrained(f'experiments/elasticdnn/gpt_neo/{os.environ["gpt_neo_series_id"]}')


import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action

aug = naw.RandomWordAug(action="swap")



class NewsgroupDomainsDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform: Any, 
                 classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        assert transform is None
        
        self.tokenizer = tokenizer  # 传入tokenizer对象\
        tokenizer.pad_token = tokenizer.eos_token
        
        self.texts = []
        self.labels = []
        self.max_length = 512  # 设置文本的最大长度

        # json_file_path = os.path.join(root_dir, f'{split if split != "val" else "dev"}.json')
        # anns = read_json(json_file_path)
        
        # label_map = {'-': 0, '+': 1, 'negative': 0, 'positive': 1}
        
        ignore_cls_indexes = [classes.index(c) for c in ignore_classes]
        
        # for v in anns.values():
        #     if v['polarity'] not in label_map.keys():
        #         continue
            
        #     cls = label_map[v['polarity']]
            
        #     if cls in ignore_cls_indexes:
        #         continue
            
        #     self.texts += [v['sentence']]
        #     self.labels += [idx_map[cls] if idx_map is not None else cls]
        
        if split == 'val':
            split = 'test'
        data = fetch_20newsgroups(subset=split)

        self.texts = [i for _i, i in enumerate(data['data']) if data['target'][_i] not in ignore_cls_indexes]
        self.labels = [i for i in data['target'] if i not in ignore_cls_indexes]
        self.labels = [idx_map[i] if idx_map is not None else i for i in self.labels]
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        text = aug.augment(text)
        
        encoded_input = self.tokenizer.encode_plus(
            text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        x = {key: tensor.squeeze(0) for key, tensor in encoded_input.items()}
        x['return_dict'] = False
        return x, torch.tensor(label)
    
    
from ..ab_dataset import ABDataset
from ..registery import dataset_register

@dataset_register(
    name='Newsgroup-GPT-Aug3', 
    classes=['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'], 
    task_type='Sentiment Classification',
    object_type='News',
    class_aliases=[],
    shift_type=None
)
class NewsgroupGPTAug3(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return NewsgroupDomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
    
# @dataset_register(
#     name='Newsgroup2-GPT', 
#     classes=['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'], 
#     task_type='Sentiment Classification',
#     object_type='News',
#     class_aliases=[],
#     shift_type=None
# )
# class Newsgroup2GPT(ABDataset):    
#     def create_dataset(self, root_dir: str, split: str, transform, 
#                        classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
#         return NewsgroupDomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
    
# @dataset_register(
#     name='Newsgroup3', 
#     classes=['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'], 
#     task_type='Sentiment Classification',
#     object_type='News',
#     class_aliases=[],
#     shift_type=None
# )
# class Newsgroup3(ABDataset):    
#     def create_dataset(self, root_dir: str, split: str, transform, 
#                        classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
#         return NewsgroupDomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
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

        json_file_path = os.path.join(root_dir, f'{split if split != "val" else "dev"}.json.token_cls_data.for_opt')
        anns = read_json(json_file_path)
        
        # label_map = {'-': 0, '+': 1, 'negative': 0, 'positive': 1}
        
        # ignore_cls_indexes = [classes.index(c) for c in ignore_classes]
        # max_length = 0
        # for info in anns:
        #     max_length = max(len(info['sentence']), max_length)
        self.max_length = 256
        # print(f'seq_max_length: {self.max_length}')
        
        for info in anns:
            if len(info['tags']) < self.max_length:
                self.srcs += [info['sentence']]
                
                label = [TAGS.index(t) for t in info['tags']]
                label = label + [-100] * (self.max_length - len(label))
                self.tgts += [label]
            else:
                self.srcs += [info['sentence'][0: self.max_length]]
                label = [TAGS.index(t) for t in info['tags'][0: self.max_length]]
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
    name='HL5Domains-ApexAD2600Progressive-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class HL5Domains_ApexAD2600Progressive_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='HL5Domains-CanonG3-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class HL5Domains_CanonG3_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class HL5Domains_CreativeLabsNomadJukeboxZenXtra40GB_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='HL5Domains-NikonCoolpix4300-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class HL5Domains_NikonCoolpix4300_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='HL5Domains-Nokia6610-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class HL5Domains_Nokia6610_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    

@dataset_register(
    name='Liu3Domains-Computer-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Liu3Domains_Computer_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='Liu3Domains-Router-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Liu3Domains_Router_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='Liu3Domains-Speaker-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Liu3Domains_Speaker_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    

# import os 
# for domain in os.listdir('/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing9Domains/asc'):
#     print(f"""
# @dataset_register(
#     name='Ding9Domains-{domain}', 
#     classes=TAGS, 
#     task_type='POS Tagging',
#     object_type='Generic',
#     class_aliases=[],
#     shift_type=None
# )
# class Ding9Domains_{domain}_GPTNeo(ABDataset):    
#     def create_dataset(self, root_dir: str, split: str, transform, 
#                        classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
#         return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
#           """)

@dataset_register(
    name='Ding9Domains-DiaperChamp-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_DiaperChamp_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-Norton-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_Norton_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-LinksysRouter-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_LinksysRouter_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-MicroMP3-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_MicroMP3_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-Nokia6600-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_Nokia6600_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-CanonPowerShotSD500-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_CanonPowerShotSD500_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-ipod-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_ipod_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-HitachiRouter-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_HitachiRouter_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-CanonS100-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_CanonS100_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    


@dataset_register(
    name='SemEval-Laptop-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class SemEval_Laptop_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='SemEval-Rest-TokenCls-OPT', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class SemEval_Rest_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
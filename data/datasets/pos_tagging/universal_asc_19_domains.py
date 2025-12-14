import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from typing import Dict, List, Optional, Any
from utils.common.data_record import read_json
from ..sentiment_classification.global_bert_tokenizer import get_tokenizer


TAGS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB', '#', '$', "''", '(', ')', ',', '.', ':']


# 自定义数据集类
class UniversalASC19DomainsTokenClsDataset(Dataset):

    def __init__(self, root_dir: str, split: str, transform: Any, 
                 classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        assert transform is None
        
        self.tokenizer = get_tokenizer()  # 传入tokenizer对象
        self.srcs = []
        self.tgts = []
        self.max_length = None  # 设置文本的最大长度

        json_file_path = os.path.join(root_dir, f'{split if split != "val" else "dev"}.json.token_cls_data')
        anns = read_json(json_file_path)
        
        # label_map = {'-': 0, '+': 1, 'negative': 0, 'positive': 1}
        
        # ignore_cls_indexes = [classes.index(c) for c in ignore_classes]
        
        for info in anns:
            self.srcs += [info['sentence']]
            
            label = [TAGS.index(t) for t in info['tags']]
            label = label + [-100] * (512 - len(label))
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
        x['return_dict'] = False
        
        return x, torch.tensor(label)


from ..ab_dataset import ABDataset
from ..registery import dataset_register

@dataset_register(
    name='HL5Domains-ApexAD2600Progressive-TokenCls', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class HL5Domains_ApexAD2600Progressive(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='HL5Domains-CanonG3-TokenCls', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class HL5Domains_CanonG3(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB-TokenCls', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class HL5Domains_CreativeLabsNomadJukeboxZenXtra40GB(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='HL5Domains-NikonCoolpix4300-TokenCls', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class HL5Domains_NikonCoolpix4300(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='HL5Domains-Nokia6610-TokenCls', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class HL5Domains_Nokia6610(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    

@dataset_register(
    name='Liu3Domains-Computer-TokenCls', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Liu3Domains_Computer(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='Liu3Domains-Router-TokenCls', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Liu3Domains_Router(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='Liu3Domains-Speaker-TokenCls', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Liu3Domains_Speaker(ABDataset):    
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
# class Ding9Domains_{domain}(ABDataset):    
#     def create_dataset(self, root_dir: str, split: str, transform, 
#                        classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
#         return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
#           """)

@dataset_register(
    name='Ding9Domains-DiaperChamp-TokenCls', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_DiaperChamp(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-Norton-TokenCls', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_Norton(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-LinksysRouter-TokenCls', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_LinksysRouter(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-MicroMP3-TokenCls', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_MicroMP3(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-Nokia6600-TokenCls', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_Nokia6600(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-CanonPowerShotSD500-TokenCls', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_CanonPowerShotSD500(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-ipod-TokenCls', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_ipod(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-HitachiRouter-TokenCls', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_HitachiRouter(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-CanonS100-TokenCls', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_CanonS100(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    


@dataset_register(
    name='SemEval-Laptop-TokenCls', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class SemEval_Laptop(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='SemEval-Rest-TokenCls', 
    classes=TAGS, 
    task_type='POS Tagging',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class SemEval_Rest(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTokenClsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
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


# 自定义数据集类
class UniversalASC19DomainsDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform: Any, 
                 classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        assert transform is None
        
        self.tokenizer = tokenizer # 传入tokenizer对象
        tokenizer.pad_token = tokenizer.eos_token
        
        self.texts = []
        self.labels = []
        self.max_length = 512  # 设置文本的最大长度

        json_file_path = os.path.join(root_dir, f'{split if split != "val" else "dev"}.json')
        anns = read_json(json_file_path)
        
        label_map = {'-': 0, '+': 1, 'negative': 0, 'positive': 1}
        
        ignore_cls_indexes = [classes.index(c) for c in ignore_classes]
        
        for v in anns.values():
            if v['polarity'] not in label_map.keys():
                continue
            
            cls = label_map[v['polarity']]
            
            if cls in ignore_cls_indexes:
                continue
            
            self.texts += [v['sentence']]
            self.labels += [idx_map[cls] if idx_map is not None else cls]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoded_input = self.tokenizer.encode_plus(
            text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        x = {key: tensor.squeeze(0) for key, tensor in encoded_input.items()}
        x['return_dict'] = False
        return x, torch.tensor(label)


from ..ab_dataset import ABDataset
from ..registery import dataset_register

@dataset_register(
    name='HL5Domains-ApexAD2600Progressive-GPTNeo', 
    classes=['negative', 'positive'], 
    task_type='Sentiment Classification',
    object_type='Emotion',
    class_aliases=[],
    shift_type=None
)
class HL5Domains_ApexAD2600Progressive_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='HL5Domains-CanonG3-GPTNeo', 
    classes=['negative', 'positive'], 
    task_type='Sentiment Classification',
    object_type='Emotion',
    class_aliases=[],
    shift_type=None
)
class HL5Domains_CanonG3_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB-GPTNeo', 
    classes=['negative', 'positive'], 
    task_type='Sentiment Classification',
    object_type='Emotion',
    class_aliases=[],
    shift_type=None
)
class HL5Domains_CreativeLabsNomadJukeboxZenXtra40GB_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='HL5Domains-NikonCoolpix4300-GPTNeo', 
    classes=['negative', 'positive'], 
    task_type='Sentiment Classification',
    object_type='Emotion',
    class_aliases=[],
    shift_type=None
)
class HL5Domains_NikonCoolpix4300_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='HL5Domains-Nokia6610-GPTNeo', 
    classes=['negative', 'positive'], 
    task_type='Sentiment Classification',
    object_type='Emotion',
    class_aliases=[],
    shift_type=None
)
class HL5Domains_Nokia6610_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    

@dataset_register(
    name='Liu3Domains-Computer-GPTNeo', 
    classes=['negative', 'positive'], 
    task_type='Sentiment Classification',
    object_type='Emotion',
    class_aliases=[],
    shift_type=None
)
class Liu3Domains_Computer_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='Liu3Domains-Router-GPTNeo', 
    classes=['negative', 'positive'], 
    task_type='Sentiment Classification',
    object_type='Emotion',
    class_aliases=[],
    shift_type=None
)
class Liu3Domains_Router_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='Liu3Domains-Speaker-GPTNeo', 
    classes=['negative', 'positive'], 
    task_type='Sentiment Classification',
    object_type='Emotion',
    class_aliases=[],
    shift_type=None
)
class Liu3Domains_Speaker_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    

# import os 
# for domain in os.listdir('/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing9Domains/asc'):
#     print(f"""
# @dataset_register(
#     name='Ding9Domains-{domain}', 
#     classes=['negative', 'positive'], 
#     task_type='Sentiment Classification',
#     object_type='Emotion',
#     class_aliases=[],
#     shift_type=None
# )
# class Ding9Domains_{domain}(ABDataset):    
#     def create_dataset(self, root_dir: str, split: str, transform, 
#                        classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
#         return UniversalASC19DomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
#           """)

@dataset_register(
    name='Ding9Domains-DiaperChamp-GPTNeo', 
    classes=['negative', 'positive'], 
    task_type='Sentiment Classification',
    object_type='Emotion',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_DiaperChamp_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-Norton-GPTNeo', 
    classes=['negative', 'positive'], 
    task_type='Sentiment Classification',
    object_type='Emotion',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_Norton_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-LinksysRouter-GPTNeo', 
    classes=['negative', 'positive'], 
    task_type='Sentiment Classification',
    object_type='Emotion',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_LinksysRouter_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-MicroMP3-GPTNeo', 
    classes=['negative', 'positive'], 
    task_type='Sentiment Classification',
    object_type='Emotion',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_MicroMP3_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-Nokia6600-GPTNeo', 
    classes=['negative', 'positive'], 
    task_type='Sentiment Classification',
    object_type='Emotion',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_Nokia6600_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-CanonPowerShotSD500-GPTNeo', 
    classes=['negative', 'positive'], 
    task_type='Sentiment Classification',
    object_type='Emotion',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_CanonPowerShotSD500_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-ipod-GPTNeo', 
    classes=['negative', 'positive'], 
    task_type='Sentiment Classification',
    object_type='Emotion',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_ipod_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-HitachiRouter-GPTNeo', 
    classes=['negative', 'positive'], 
    task_type='Sentiment Classification',
    object_type='Emotion',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_HitachiRouter_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-CanonS100-GPTNeo', 
    classes=['negative', 'positive'], 
    task_type='Sentiment Classification',
    object_type='Emotion',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_CanonS100_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    


@dataset_register(
    name='SemEval-Laptop-GPTNeo', 
    classes=['negative', 'positive'], 
    task_type='Sentiment Classification',
    object_type='Emotion',
    class_aliases=[],
    shift_type=None
)
class SemEval_Laptop_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='SemEval-Rest-GPTNeo', 
    classes=['negative', 'positive'], 
    task_type='Sentiment Classification',
    object_type='Emotion',
    class_aliases=[],
    shift_type=None
)
class SemEval_Rest_GPTNeo(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
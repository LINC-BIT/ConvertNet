import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from typing import Dict, List, Optional, Any
from utils.common.data_record import read_json
from ..sentiment_classification.global_bert_tokenizer import get_tokenizer


# 自定义数据集类
class UniversalASC19DomainsTranslationDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform: Any, 
                 classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        assert transform is None
        
        self.tokenizer = get_tokenizer()  # 传入tokenizer对象
        self.srcs = []
        self.tgts = []
        self.max_length = None  # 设置文本的最大长度

        json_file_path = os.path.join(root_dir, f'{split if split != "val" else "dev"}.json.translate_data')
        anns = read_json(json_file_path)
        
        # label_map = {'-': 0, '+': 1, 'negative': 0, 'positive': 1}
        
        # ignore_cls_indexes = [classes.index(c) for c in ignore_classes]
        
        for info in anns:
            self.srcs += [info['src']]
            self.tgts += [info['dst']]

    def __len__(self):
        return len(self.srcs)

    def __getitem__(self, idx):
        src = self.srcs[idx]
        tgt = self.tgts[idx]
        
        encoded_src = self.tokenizer(
            src, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        encoded_tgt = self.tokenizer(
            tgt, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        x = {key: tensor.squeeze(0) for key, tensor in encoded_src.items()}
        
        y = encoded_tgt['input_ids'][0]
        y = torch.LongTensor([(int(l) if l != self.tokenizer.pad_token_id else -100) for l in y])
        
        return x, y


from ..ab_dataset import ABDataset
from ..registery import dataset_register

@dataset_register(
    name='HL5Domains-ApexAD2600Progressive-Tr', 
    classes=['unknown'], 
    task_type='Machine Translation',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class HL5Domains_ApexAD2600Progressive(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTranslationDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='HL5Domains-CanonG3-Tr', 
    classes=['unknown'], 
    task_type='Machine Translation',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class HL5Domains_CanonG3(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTranslationDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB-Tr', 
    classes=['unknown'], 
    task_type='Machine Translation',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class HL5Domains_CreativeLabsNomadJukeboxZenXtra40GB(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTranslationDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='HL5Domains-NikonCoolpix4300-Tr', 
    classes=['unknown'], 
    task_type='Machine Translation',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class HL5Domains_NikonCoolpix4300(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTranslationDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='HL5Domains-Nokia6610-Tr', 
    classes=['unknown'], 
    task_type='Machine Translation',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class HL5Domains_Nokia6610(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTranslationDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    

@dataset_register(
    name='Liu3Domains-Computer-Tr', 
    classes=['unknown'], 
    task_type='Machine Translation',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Liu3Domains_Computer(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTranslationDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='Liu3Domains-Router-Tr', 
    classes=['unknown'], 
    task_type='Machine Translation',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Liu3Domains_Router(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTranslationDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='Liu3Domains-Speaker-Tr', 
    classes=['unknown'], 
    task_type='Machine Translation',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Liu3Domains_Speaker(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTranslationDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    

# import os 
# for domain in os.listdir('/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing9Domains/asc'):
#     print(f"""
# @dataset_register(
#     name='Ding9Domains-{domain}', 
#     classes=['unknown'], 
#     task_type='Machine Translation',
#     object_type='Generic',
#     class_aliases=[],
#     shift_type=None
# )
# class Ding9Domains_{domain}(ABDataset):    
#     def create_dataset(self, root_dir: str, split: str, transform, 
#                        classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
#         return UniversalASC19DomainsTranslationDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
#           """)

@dataset_register(
    name='Ding9Domains-DiaperChamp-Tr', 
    classes=['unknown'], 
    task_type='Machine Translation',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_DiaperChamp(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTranslationDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-Norton-Tr', 
    classes=['unknown'], 
    task_type='Machine Translation',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_Norton(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTranslationDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-LinksysRouter-Tr', 
    classes=['unknown'], 
    task_type='Machine Translation',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_LinksysRouter(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTranslationDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-MicroMP3-Tr', 
    classes=['unknown'], 
    task_type='Machine Translation',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_MicroMP3(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTranslationDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-Nokia6600-Tr', 
    classes=['unknown'], 
    task_type='Machine Translation',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_Nokia6600(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTranslationDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-CanonPowerShotSD500-Tr', 
    classes=['unknown'], 
    task_type='Machine Translation',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_CanonPowerShotSD500(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTranslationDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-ipod-Tr', 
    classes=['unknown'], 
    task_type='Machine Translation',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_ipod(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTranslationDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-HitachiRouter-Tr', 
    classes=['unknown'], 
    task_type='Machine Translation',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_HitachiRouter(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTranslationDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
          

@dataset_register(
    name='Ding9Domains-CanonS100-Tr', 
    classes=['unknown'], 
    task_type='Machine Translation',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class Ding9Domains_CanonS100(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTranslationDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    


@dataset_register(
    name='SemEval-Laptop-Tr', 
    classes=['unknown'], 
    task_type='Machine Translation',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class SemEval_Laptop(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTranslationDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
    
@dataset_register(
    name='SemEval-Rest-Tr', 
    classes=['unknown'], 
    task_type='Machine Translation',
    object_type='Generic',
    class_aliases=[],
    shift_type=None
)
class SemEval_Rest(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform, 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        return UniversalASC19DomainsTranslationDataset(root_dir, split, transform, classes, ignore_classes, idx_map)
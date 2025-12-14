from transformers import AutoTokenizer
from utils.common.log import logger
import os

tokenizer = None 
bert_model_tag = os.environ['bert_path'] if 'bert_path' in os.environ.keys() else '/data/zql/concept-drift-in-edge-projects/UniversalElasticNet/data/datasets/sentiment_classification/mobilebert-uncased'


def get_tokenizer():
    global tokenizer
    
    
    if tokenizer is None:
        logger.info(f'init bert tokenizer for sen cls (using {bert_model_tag})')
        tokenizer = AutoTokenizer.from_pretrained(bert_model_tag)
    return tokenizer

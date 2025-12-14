import os


# root_dir = '/data/zql/datasets/opus_boo'
# root_dir = os.path.join(root_dir, 'de-en')
# lang1_path = os.path.join(root_dir, 'commoncrawl.de-en.en')

from utils.common.data_record import read_json, write_json
import requests
import random
import hashlib
import tqdm
import time
import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# https://blog.csdn.net/weixin_44826203/article/details/107484634

# from data.datasets.sentiment_classification.global_bert_tokenizer import get_tokenizer
# bert_tokenizer = get_tokenizer()
from transformers import GPT2Tokenizer, AutoTokenizer
bert_tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b', use_fast=False)

pos_tag_set = set()

def gen_label(lang1_path, max_num_sentences=100000):
    texts = []
    anns = read_json(lang1_path)
    # with open(lang1_path, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         texts += [line.strip()]
    for ann in anns:
        texts += [ann['en']]
    # texts = list(set(texts))
        
    res_json = []
    pbar = tqdm.tqdm(max_num_sentences, total=max_num_sentences)
    
    for text in texts:
        # text = "When I put it to use for my daughter 's graduation party in longer lengths , the speaker wire worked as expected , even out of doors . I like it . "

        # tag for whole text
        words_whole_text = bert_tokenizer._tokenize(text)
        pos_tags_trial = nltk.pos_tag(words_whole_text)
        
        # tag for splited sentences
        sentences = nltk.sent_tokenize(text)
        
        pos_tags = []
        for sentence in sentences:
            # words = nltk.word_tokenize(sentence)
            words = bert_tokenizer._tokenize(sentence)
            pos_tags += nltk.pos_tag(words)
        
        if len(pos_tags_trial) != len(pos_tags):
            pos_tags = pos_tags_trial
        
        if len(pos_tags) > 256:
            continue
        
        res_json += [{
            'sentence': text,
            'tags': [t[1] for t in pos_tags]
        }]
        
        for tag in pos_tags:
            pos_tag_set.add(tag[1])
        
        pbar.update()
        if len(res_json) > max_num_sentences:
            break
        
    write_json(lang1_path + f'.token_cls_data.{max_num_sentences}', res_json, backup=False)
        
        
# gen_label('/data/zql/datasets/news_commentary/de-en/train.json', 10000)
# gen_label('/data/zql/datasets/news_commentary/de-en/val.json', 2000)
# gen_label('/data/zql/datasets/opus_books/de-en/train.json', 10000)
# gen_label('/data/zql/datasets/opus_books/de-en/val.json', 2000)
# gen_label('/data/zql/datasets/wmt14/de-en/train.json', 10000)
# gen_label('/data/zql/datasets/wmt14/de-en/val.json', 2000)


gen_label('/data/zql/datasets/news_commentary/de-en/train.json', 10000)
gen_label('/data/zql/datasets/news_commentary/de-en/val.json', 2000)
gen_label('/data/zql/datasets/opus_books/de-en/train.json', 10000)
gen_label('/data/zql/datasets/opus_books/de-en/val.json', 2000)
gen_label('/data/zql/datasets/wmt14/de-en/train.json', 10000)
gen_label('/data/zql/datasets/wmt14/de-en/val.json', 2000)

print(pos_tag_set)
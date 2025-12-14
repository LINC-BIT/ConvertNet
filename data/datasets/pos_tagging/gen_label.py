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
from transformers import AutoTokenizer
bert_tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b', use_fast=False)

pos_tag_set = set()


def get_sentences_len(sen_cls_json_path):
    anns = read_json(sen_cls_json_path)
    avg_len = 0
    n = 0
    for v in anns.values():
        avg_len += len(v['sentence'].split(' '))
        n += 1
    print(avg_len / n)
        

def gen_label_from_sen_cls_json(sen_cls_json_path):
    texts = []
    anns = read_json(sen_cls_json_path)
    for v in anns.values():
        texts += [v['sentence']]
        assert '\n' not in texts[-1]
        
    texts = list(set(texts))
        
    res_json = []
    
    
    # for text in tqdm.tqdm(texts):
    #     text = "When I put it to use for my daughter 's graduation party in longer lengths , the speaker wire worked as expected , even out of doors . I like it . "
    #     words = bert_tokenizer._tokenize(text)
    #     pos_tags = nltk.pos_tag(words)
        
    #     print(text, pos_tags, len(pos_tags))
        
    #     exit()
    

    for text in tqdm.tqdm(texts):
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
            
        res_json += [{
            'sentence': text,
            'tags': [t[1] for t in pos_tags]
        }]
        
        for tag in pos_tags:
            pos_tag_set.add(tag[1])
        
        write_json(sen_cls_json_path + '.token_cls_data.for_opt', res_json, backup=False)
    
if __name__ == '__main__':
    # res = translate('I am a doctor.\nHello world!')
    # print(res)
    import os
    
    data_dir_paths = {
        **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing5Domains/asc/{k.split("-")[1]}' 
             for k in ['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB',
                              'HL5Domains-NikonCoolpix4300', 'HL5Domains-Nokia6610']},
        
        **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing3Domains/asc/{k.split("-")[1]}' 
             for k in ['Liu3Domains-Computer', 'Liu3Domains-Router', 'Liu3Domains-Speaker']},
        
        **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing9Domains/asc/{k.split("-")[1]}' 
             for k in [f'Ding9Domains-{d}' for d in os.listdir('/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing9Domains/asc')]},
        
        **{f'SemEval-{k[0].upper()}{k[1:]}': f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/XuSemEval/asc/14/{k}' 
             for k in ['laptop', 'rest']},
    }
    
    json_paths = []
    for p in data_dir_paths.values():
        # json_paths += [os.path.join(p, f'{split}.json') for split in ['train', 'dev', 'test']]
        json_paths += [os.path.join(p, f'{split}.json') for split in ['train']]
        
    assert all([os.path.exists(p) for p in json_paths])
    
    # print(len(json_paths))
    # exit()
    
    for p in tqdm.tqdm(json_paths):
        print(p)
        # gen_label_from_sen_cls_json(p)
        get_sentences_len(p)
        
    # print(pos_tag_set)
    
    # """
    # ['FW', 'NNPS', '$', 'RBR', 'DT', 'VBG', 'EX', '.', 'JJS', 'RB', 'RP', 'JJR', '#', 'IN', 'VBZ', 'VB', 'NNP', 'WRB', 'JJ', 'POS', 'WP', 'RBS', 'VBN', 'UH', 'PRP$', 'NN', 'VBD', '(', 'NNS', 'WDT', 'MD', ',', 'SYM', 'TO', 'VBP', 'LS', 'PDT', 'CD', ')', ':', "''", 'PRP', 'CC']
    # """
    
    # tags = ['FW', 'NNPS', '$', 'RBR', 'DT', 'VBG', 'EX', '.', 'JJS', 'RB', 'RP', 'JJR', '#', 'IN', 'VBZ', 'VB', 'NNP', 'WRB', 'JJ', 'POS', 'WP', 'RBS', 'VBN', 'UH', 'PRP$', 'NN', 'VBD', '(', 'NNS', 'WDT', 'MD', ',', 'SYM', 'TO', 'VBP', 'LS', 'PDT', 'CD', ')', ':', "''", 'PRP', 'CC']
    # print(sorted(tags))
    
    # """
    # ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB', '#', '$', "''", '(', ')', ',', '.', ':']
    # """
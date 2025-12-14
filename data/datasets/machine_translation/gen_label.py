from utils.common.data_record import read_json, write_json
import requests
import random
import hashlib
import tqdm
import time


session = requests.Session()


def translate(sentence):
    app_id = '20221004001369410'
    salt = str(random.randint(1000000000, 9999999999))
    key = 'XEsBS6babmp9wz5bcoEs'
    
    sign = hashlib.md5(f'{app_id}{sentence}{salt}{key}'.encode('utf8')).hexdigest()
    
    response = requests.get(
        'https://fanyi-api.baidu.com/api/trans/vip/translate',
        params={
            'q': sentence,
            'from': 'en',
            'to': 'zh',
            'appid': app_id,
            'salt': salt,
            'sign': sign
        }
    ).json()
    
    if 'trans_result' not in response.keys():
        print(response)
        raise RuntimeError
    
    return response['trans_result'][0]['src'], response['trans_result'][0]['dst']


def gen_label_from_sen_cls_json(sen_cls_json_path):
    # generate Chinese translation
    
    texts = []
    anns = read_json(sen_cls_json_path)
    for v in anns.values():
        texts += [v['sentence']]
        assert '\n' not in texts[-1]
        
    texts = list(set(texts))
        
    res_json = []

    for text in tqdm.tqdm(texts):
        time.sleep(1.2)
        
        src_text, dst_text = translate(text)
        res_json += [{
            'src': src_text,
            'dst': dst_text
        }]
        
        write_json(sen_cls_json_path + '.translate_data', res_json, backup=False)
        
    
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
        json_paths += [os.path.join(p, f'{split}.json') for split in ['train', 'dev', 'test']]
        
    assert all([os.path.exists(p) for p in json_paths])
    
    # print(len(json_paths))
    # exit()
    
    for p in tqdm.tqdm(json_paths[23:]):
        print(p)
        gen_label_from_sen_cls_json(p)
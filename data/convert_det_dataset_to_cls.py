from data import ABDataset
from utils.common.data_record import read_json, write_json
from PIL import Image
import os 
from utils.common.file import ensure_dir
import numpy as np
from itertools import groupby
from skimage import morphology, measure
from PIL import Image
from scipy import misc
import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import shutil


def convert_det_dataset_to_det(coco_ann_json_path, data_dir, target_data_dir, min_img_size=224):
    
    coco_ann = read_json(coco_ann_json_path)
    
    img_id_to_path = {}
    for img in coco_ann['images']:
        img_id_to_path[img['id']] = os.path.join(data_dir, img['file_name'])
    
    classes_imgs_id_map = {}
    for ann in tqdm.tqdm(coco_ann['annotations'], total=len(coco_ann['annotations']), dynamic_ncols=True):
        img_id = ann['image_id']
        img_path = img_id_to_path[img_id]
        img = Image.open(img_path)

        bbox = ann['bbox']
        if bbox[2] < min_img_size or bbox[3] < min_img_size:
            continue
        
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        
        class_idx = str(ann['category_id'])
        if class_idx not in classes_imgs_id_map.keys():
            classes_imgs_id_map[class_idx] = 0
        target_cropped_img_path = os.path.join(target_data_dir, class_idx, 
                                               f'{classes_imgs_id_map[class_idx]}.{img_path.split(".")[-1]}')
        classes_imgs_id_map[class_idx] += 1
        
        ensure_dir(target_cropped_img_path)
        img.crop(bbox).save(target_cropped_img_path)
        
        
        
if __name__ == '__main__':
    convert_det_dataset_to_det(
        coco_ann_json_path='/data/zql/datasets/coco2017/train2017/coco_ann.json',
        data_dir='/data/zql/datasets/coco2017/train2017',
        target_data_dir='/data/zql/datasets/coco2017_for_cls_task',
        min_img_size=224
    )
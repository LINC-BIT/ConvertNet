from curses import raw
from .data_augment import TrainTransform, ValTransform
from .datasets.coco import COCODataset
from .datasets.mm_coco import MM_COCODataset
from .datasets.mosaicdetection import MosaicDetection
from utils.common.others import HiddenPrints
import os
import json
from tqdm import tqdm 
from utils.common.log import logger

from .norm_categories_index import ensure_index_start_from_1_and_successive


def get_default_yolox_coco_dataset(data_dir, json_file_path, img_size=416, train=True):
    logger.info(f'[get yolox dataset] "{json_file_path}"')

    if train:
        with HiddenPrints():
            dataset = COCODataset(
                data_dir=data_dir,
                json_file=json_file_path,
                name='',
                img_size=(img_size, img_size),
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=0.5,
                    hsv_prob=1.0),
                cache=False,
            )
            # dataset = COCODataset(
            #     data_dir=data_dir,
            #     json_file=json_file_path,
            #     name='',
            #     img_size=(img_size, img_size),
            #     preproc=ValTransform(legacy=False),
            # )
            
        dataset = MosaicDetection(
            dataset,
            mosaic=True,
            img_size=(img_size, img_size),
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=0.5,
                hsv_prob=1.0),
            degrees=10.0,
            translate=0.1,
            mosaic_scale=(0.1, 2),
            mixup_scale=(0.5, 1.5),
            shear=2.0,
            enable_mixup=True,
            mosaic_prob=1.0,
            mixup_prob=1.0,
            only_return_xy=True
        )
        
    else:
        with HiddenPrints():
            dataset = COCODataset(
                data_dir=data_dir,
                json_file=json_file_path,
                name='',
                img_size=(img_size, img_size),
                preproc=ValTransform(legacy=False),
            )

    # print(json_file_path, len(dataset))
            
    return dataset

def get_yolox_coco_dataset_with_caption(data_dir, json_file_path, img_size=416, transform=None, train=True, classes=None):
    logger.info(f'[get yolox dataset] "{json_file_path}"')

    if train:
        with HiddenPrints():
            dataset = COCODataset(
                data_dir=data_dir,
                json_file=json_file_path,
                name='',
                img_size=(img_size, img_size),
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=0.5,
                    hsv_prob=1.0),
                cache=False,
            )
            # dataset = COCODataset(
            #     data_dir=data_dir,
            #     json_file=json_file_path,
            #     name='',
            #     img_size=(img_size, img_size),
            #     preproc=ValTransform(legacy=False),
            # )
        coco = dataset.coco
        dataset = MosaicDetection(
            dataset,
            mosaic=True,
            img_size=(img_size, img_size),
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=0.5,
                hsv_prob=1.0),
            degrees=10.0,
            translate=0.1,
            mosaic_scale=(0.1, 2),
            mixup_scale=(0.5, 1.5),
            shear=2.0,
            enable_mixup=True,
            mosaic_prob=1.0,
            mixup_prob=1.0,
            only_return_xy=True
        )
        dataset = MM_COCODataset(dataset, transform=transform, split='train', coco=coco, classes=classes)
    else:
        with HiddenPrints():
            dataset = COCODataset(
                data_dir=data_dir,
                json_file=json_file_path,
                name='',
                img_size=(img_size, img_size),
                preproc=ValTransform(legacy=False),
            )
        dataset = MM_COCODataset(dataset, transform=transform, split='val', coco=dataset.coco, classes=classes)
    # print(json_file_path, len(dataset))
            
    return dataset

import hashlib

def _hash(o):
    if isinstance(o, list):
        o = sorted(o)
    elif isinstance(o, dict):
        o = {k: o[k] for k in sorted(o)}
    elif isinstance(o, set):
        o = sorted(list(o))
    # else:
    #     print(type(o))
    
    obj = hashlib.md5()
    obj.update(str(o).encode('utf-8'))
    return obj.hexdigest()


DEBUG = True


def remap_dataset(json_file_path, ignore_classes, category_idx_map):
    # k and v in category_idx_map indicates the index of categories, not 'id' of categories
    ignore_classes = sorted(list(ignore_classes))
    # print(ignore_classes, category_idx_map)

    if len(ignore_classes) == 0 and category_idx_map is None:
        return json_file_path
    
    # hash_str = '_'.join(ignore_classes) + str(category_idx_map)
    hash_str = _hash(f'yolox_dataset_cache_{_hash(ignore_classes)}_{_hash(category_idx_map)}')
    cached_json_file_path = f'{json_file_path}.{hash(hash_str)}'
    
    # TODO:
    if os.path.exists(cached_json_file_path):
        if DEBUG:
            os.remove(cached_json_file_path)
        else:
            logger.info(f'get cached dataset in {cached_json_file_path}')
            return cached_json_file_path
    
    with open(json_file_path, 'r') as f:
        raw_ann = json.load(f)
    id_to_idx_map = {c['id']: i for i, c in enumerate(raw_ann['categories'])}
        
    ignore_classes_id = [c['id'] for c in raw_ann['categories'] if c['name'] in ignore_classes]
    raw_ann['categories'] = [c for c in raw_ann['categories'] if c['id'] not in ignore_classes_id]
    raw_ann['annotations'] = [ann for ann in raw_ann['annotations'] if ann['category_id'] not in ignore_classes_id]
    ann_img_map = {ann['image_id']: 1 for ann in raw_ann['annotations']}
    raw_ann['images'] = [img for img in raw_ann['images'] if img['id'] in ann_img_map.keys()]
    
    # print(category_idx_map, id_to_idx_map)
    # NOTE: category idx starts from 0 or 1? 1
    # NOTE: reshuffle "categories"
    new_categories = [{"id": i, "name": f"dummy-{i}"} for i in range(int(os.environ['_ZQL_NUMC']))]
    for c in raw_ann['categories']:
        # print(c)
        # print(id_to_idx_map, c['id'], category_idx_map)
        new_idx = category_idx_map[id_to_idx_map[c['id']]]
        new_categories[new_idx] = c
        c['id'] = new_idx
    raw_ann['categories'] = new_categories
    for ann in raw_ann['annotations']:
        ann['category_id'] = category_idx_map[id_to_idx_map[ann['category_id']]]
        if 'segmentation' in ann:
            del ann['segmentation']
    
    with open(cached_json_file_path, 'w') as f:
        json.dump(raw_ann, f)
        
    return cached_json_file_path


def coco_split(ann_json_file_path, ratio=0.8):
    if os.path.exists(ann_json_file_path + f'.{ratio}.split1') and not DEBUG:
        return ann_json_file_path + f'.{ratio}.split1', ann_json_file_path + f'.{ratio}.split2'
    
    with open(ann_json_file_path, 'r') as f:
        raw_ann = json.load(f)

    import copy 
    import torch 
    res_ann1, res_ann2 = copy.deepcopy(raw_ann), copy.deepcopy(raw_ann)

    images = raw_ann['images']

    cache_images_path = ann_json_file_path + '.tmp-cached-shuffled-images'
    if True:
        import random
        random.shuffle(images)
        torch.save(images, cache_images_path)
    else:
        images = torch.load(cache_images_path)

    images1, images2 = images[0: int(len(images) * ratio)], images[int(len(images) * ratio): ]
    images1_id, images2_id = {i['id']: 0 for i in images1}, {i['id']: 0 for i in images2}
    ann1 = [ann for ann in raw_ann['annotations'] if ann['image_id'] in images1_id.keys()]
    ann2 = [ann for ann in raw_ann['annotations'] if ann['image_id'] in images2_id.keys()]

    res_ann1['images'] = images1
    res_ann1['annotations'] = ann1
    res_ann2['images'] = images2
    res_ann2['annotations'] = ann2 

    from utils.common.data_record import write_json
    write_json(ann_json_file_path + f'.{ratio}.split1', res_ann1, indent=0, backup=False)
    write_json(ann_json_file_path + f'.{ratio}.split2', res_ann2, indent=0, backup=False)

    return ann_json_file_path + f'.{ratio}.split1', ann_json_file_path + f'.{ratio}.split2'


def coco_train_val_test_split(ann_json_file_path, split):
    train_ann_p, test_ann_p = coco_split(ann_json_file_path)
    if split == 'test':
        return test_ann_p
    train_ann_p, val_ann_p = coco_split(train_ann_p)
    return train_ann_p if split == 'train' else val_ann_p


def coco_train_val_split(train_ann_p, split):
    train_ann_p, val_ann_p = coco_split(train_ann_p)
    return train_ann_p if split == 'train' else val_ann_p


def visualize_coco_dataset(dataset, num_images, res_save_p, cxcy):
    from torchvision.transforms import ToTensor
    from torchvision.utils import make_grid
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt
    import numpy as np
    
    def draw_bbox(img, bbox, label, f):
        # if f:
        #     img = np.uint8(img.transpose(1, 2, 0))
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox, outline=(255, 0, 0), width=6)
        draw.text((bbox[0], bbox[1]), label)
        return np.array(img)

    d = dataset.dataset
    if d.__class__.__name__ == 'MosaicDetection':
        d = d._dataset
    class_ids = d.class_ids # category_id
    def get_cname(label):
        return d.coco.loadCats(class_ids[int(label)])[0]['name']

    def cxcywh2xyxy(bbox):
        cx, cy, w, h = bbox
        x1, y1 = cx - w/2, cy - h/2
        x2, y2 = cx + w/2, cy + h/2
        return x1, y1, x2, y2

    xs = []
    import random
    for image_i in range(num_images):
        x, y = dataset[random.randint(0, len(dataset) - 1)][:2]
        x = np.uint8(x.transpose(1, 2, 0))

        for label_i, label_info in enumerate(y):
            if sum(label_info[1:]) == 0: # pad label
                break

            label, bbox = label_info[0], label_info[1:]

            if cxcy:
                bbox = cxcywh2xyxy(bbox)

            x = draw_bbox(x, bbox, str(label) + '-' + get_cname(label), label_i == 0)
        # print(x.shape)
        xs += [x]

    xs = [ToTensor()(x) for x in xs]
    grid = make_grid(xs, normalize=True, nrow=2)
    plt.axis('off')
    img = grid.permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.savefig(res_save_p, dpi=300)
    plt.clf()

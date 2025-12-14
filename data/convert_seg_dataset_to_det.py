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


def convert_seg_dataset_to_det(seg_imgs_path, seg_labels_path, root_dir, target_coco_ann_path, ignore_classes_idx, thread_i, min_img_size=224, label_after_hook=lambda x: x):
    """
    Reference: https://blog.csdn.net/lizaijinsheng/article/details/119889946

    NOTE: 
    Background class should not be considered. 
    However, if a seg dataset has only one valid class, so that the generated cls dataset also has only one class and 
    the cls accuracy will be 100% forever. But we do not use the generated cls dataset alone, so it is ok.
    """
    assert len(seg_imgs_path) == len(seg_labels_path)
    
    classes_imgs_id_map = {}
    
    coco_ann = {
        'categories': [],
        "type": "instances",
        'images': [],
        'annotations': []
    }
    
    image_id = 0
    ann_id = 0
    
    pbar = tqdm.tqdm(zip(seg_imgs_path, seg_labels_path), total=len(seg_imgs_path), 
                                                   dynamic_ncols=True, leave=False, desc=f'thread {thread_i}')
    for seg_img_path, seg_label_path in pbar:

        try:
            seg_img = Image.open(seg_img_path)
            seg_label = Image.open(seg_label_path).convert('L')
            seg_label = np.array(seg_label)
            seg_label = label_after_hook(seg_label)
        except Exception as e:
            print(e)
            print(f'file {seg_img_path} error, skip')
            exit()
        # seg_img = Image.open(seg_img_path)
        # seg_label = Image.open(seg_label_path).convert('L')
        # seg_label = np.array(seg_label)
        
        image_coco_info = {'file_name': os.path.relpath(seg_img_path, root_dir), 'height': seg_img.height, 'width': seg_img.width,
                 'id':image_id}
        image_id += 1
        coco_ann['images'] += [image_coco_info]
            
        this_img_classes = set(seg_label.reshape(-1).tolist())
        # print(this_img_classes)
        
        for class_idx in this_img_classes:
            if class_idx in ignore_classes_idx:
                continue
            
            if class_idx not in classes_imgs_id_map.keys():
                classes_imgs_id_map[class_idx] = 0

            mask = np.zeros((seg_label.shape[0], seg_label.shape[1]), dtype=np.uint8)
            mask[seg_label == class_idx] = 1
            mask_without_small = morphology.remove_small_objects(mask, min_size=10, connectivity=2)
            label_image = measure.label(mask_without_small)

            for region in measure.regionprops(label_image):
                bbox = region.bbox # (top, left, bottom, right)
                bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]  # (left, top, right, bottom)
                
                width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if width < min_img_size or height < min_img_size:
                    continue
                
                # target_cropped_img_path = os.path.join(target_cls_data_dir, str(class_idx), 
                #                                        f'{classes_imgs_id_map[class_idx]}.{seg_img_path.split(".")[-1]}')
                # ensure_dir(target_cropped_img_path)
                # seg_img.crop(bbox).save(target_cropped_img_path)    
                # print(target_cropped_img_path)
                # exit()
                
                ann_coco_info = {'area': width*height, 'iscrowd': 0, 'image_id':
                   image_id - 1, 'bbox': [bbox[0], bbox[1], width, height],
                   'category_id': class_idx, 
                   'id': ann_id, 'ignore': 0,
                   'segmentation': []}
                ann_id += 1
                
                coco_ann['annotations'] += [ann_coco_info]
                
                classes_imgs_id_map[class_idx] += 1
                
                pbar.set_description(f'# ann: {ann_id}')

    coco_ann['categories'] = [
        {'id': ci, 'name': f'class_{c}_in_seg'} for ci, c in enumerate(classes_imgs_id_map.keys())
    ]
    c_to_ci_map = {c: ci for ci, c in enumerate(classes_imgs_id_map.keys())}
    for ann in coco_ann['annotations']:
        ann['category_id'] = c_to_ci_map[
            ann['category_id']
        ]
    
    write_json(target_coco_ann_path, coco_ann, indent=0, backup=True)
    write_json(os.path.join(root_dir, 'coco_ann.json'), coco_ann, indent=0, backup=True)

    num_cls_imgs = 0
    for k, v in classes_imgs_id_map.items():
        # print(f'# class {k}: {v + 1}')
        num_cls_imgs += v
    # print(f'total: {num_cls_imgs}')
    
    return classes_imgs_id_map
    

from concurrent.futures import ThreadPoolExecutor



# def convert_seg_dataset_to_cls_multi_thread(seg_imgs_path, seg_labels_path, target_cls_data_dir, ignore_classes_idx, num_threads):
#     if os.path.exists(target_cls_data_dir):
#         shutil.rmtree(target_cls_data_dir)
    
#     assert len(seg_imgs_path) == len(seg_labels_path)
#     n = len(seg_imgs_path) // num_threads
    
#     pool = ThreadPoolExecutor(max_workers=num_threads)
#     # threads = []
#     futures = []
#     for thread_i in range(num_threads):
#         # thread = threading.Thread(target=convert_seg_dataset_to_cls, 
#         #                           args=(seg_imgs_path[thread_i * n: (thread_i + 1) * n], 
#         #                                 seg_labels_path[thread_i * n: (thread_i + 1) * n], 
#         #                                 target_cls_data_dir, ignore_classes_idx))
#         # threads += [thread]
#         future = pool.submit(convert_seg_dataset_to_cls, *(seg_imgs_path[thread_i * n: (thread_i + 1) * n], 
#                                         seg_labels_path[thread_i * n: (thread_i + 1) * n], 
#                                         target_cls_data_dir, ignore_classes_idx, thread_i))
#         futures += [future]
    
#     futures += [
#         pool.submit(convert_seg_dataset_to_cls, *(seg_imgs_path[(thread_i + 1) * n: ], 
#                                         seg_labels_path[(thread_i + 1) * n: ], 
#                                         target_cls_data_dir, ignore_classes_idx, thread_i))
#     ]
    
#     for f in futures:
#         f.done()
    
#     res = []
#     for f in futures:
#         res += [f.result()]
#         print(res[-1])
    
#     res_dist = {}
#     for r in res:
#         for k, v in r.items():
#             if k in res_dist.keys():
#                 res_dist[k] += v 
#             else:
#                 res_dist[k] = v
    
#     print('results:')
#     print(res_dist)
    
#     pool.shutdown()



# import random
# def random_crop_aug(target_dir):
#     for class_dir in os.listdir(target_dir):
#         class_dir = os.path.join(target_dir, class_dir)
        
#         for img_path in os.listdir(class_dir):
#             img_path = os.path.join(class_dir, img_path)

#             img = Image.open(img_path)
            
#             w, h = img.width, img.height
            
#             for ri in range(5):
#                 img.crop(
#                     [
#                         random.randint(0, w // 5),
#                         random.randint(0, h // 5),
#                         random.randint(w // 5 * 4, w),
#                         random.randint(h // 5 * 4, h)
#                     ]
#                 ).save(
#                     os.path.join(os.path.dirname(img_path), f'randaug_{ri}_' + os.path.basename(img_path))
#                 )
#                 # print(img_path)
#                 # exit()


def post_ignore_classes(coco_ann_json_path):
    # from data.datasets.object_detection.yolox_data_util.api import remap_dataset
    # remap_dataset(coco_ann_json_path, [], {})
    pass
    
    

if __name__ == '__main__':
    # SuperviselyPerson
    # root_dir = '/data/zql/datasets/supervisely_person_full_20230635/Supervisely Person Dataset'
    
    # images_path, labels_path = [], []
    # for p in os.listdir(root_dir):
    #     if p.startswith('ds'):
    #         p1 = os.path.join(root_dir, p, 'img')
    #         images_path += [(p, os.path.join(p1, n)) for n in os.listdir(p1)]
    # for dsi, img_p in images_path:
    #     target_p = os.path.join(root_dir, p, dsi, img_p.split('/')[-1])
    #     labels_path += [target_p]
    # images_path = [i[1] for i in images_path]
    
    # target_coco_ann_path = '/data/zql/datasets/supervisely_person_for_det_task/coco_ann.json'
    # if os.path.exists(target_coco_ann_path):
    #     os.remove(target_coco_ann_path)
    # convert_seg_dataset_to_det(
    #     seg_imgs_path=images_path,
    #     seg_labels_path=labels_path,
    #     root_dir=root_dir,
    #     target_coco_ann_path=target_coco_ann_path,
    #     ignore_classes_idx=[0, 2],
    #     # num_threads=8
    #     thread_i=0
    # )
    
    # random_crop_aug('/data/zql/datasets/supervisely_person_for_cls_task')
    
    
    # GTA5
    # root_dir = '/data/zql/datasets/GTA-ls-copy/GTA5'
    # images_path, labels_path = [], []
    # for p in os.listdir(os.path.join(root_dir, 'images')):
    #     p = os.path.join(root_dir, 'images', p)
    #     if not p.endswith('png'):
    #         continue
    #     images_path += [p]
    #     labels_path += [p.replace('images', 'labels_gt')]

    # target_coco_ann_path = '/data/zql/datasets/gta5_for_det_task/coco_ann.json'
    # if os.path.exists(target_coco_ann_path):
    #     os.remove(target_coco_ann_path)
    
    # """
    # [
    #     'road', 'sidewalk', 'building', 'wall',
    #     'fence', 'pole', 'light', 'sign',
    #     'vegetation', 'terrain', 'sky', 'people', # person
    #     'rider', 'car', 'truck', 'bus', 'train',
    #     'motocycle', 'bicycle'
    # ]
    # """
    # need_classes_idx = [13, 15]
    # convert_seg_dataset_to_det(
    #     seg_imgs_path=images_path,
    #     seg_labels_path=labels_path,
    #     root_dir=root_dir,
    #     target_coco_ann_path=target_coco_ann_path,
    #     ignore_classes_idx=[i for i in range(20) if i not in need_classes_idx],
    #     thread_i=0
    # )
    
    # from data.datasets.object_detection.yolox_data_util.api import remap_dataset
    # new_coco_ann_json_path = remap_dataset('/data/zql/datasets/GTA-ls-copy/GTA5/coco_ann.json', [-1], {0: 0, 1:-1, 2:-1, 3: 1, 4:-1, 5:-1})
    # print(new_coco_ann_json_path)
    
    # cityscapes
    # root_dir = '/data/zql/datasets/cityscape/'
    
    # def _get_target_suffix(mode: str, target_type: str) -> str:
    #     if target_type == 'instance':
    #         return '{}_instanceIds.png'.format(mode)
    #     elif target_type == 'semantic':
    #         return '{}_labelIds.png'.format(mode)
    #     elif target_type == 'color':
    #         return '{}_color.png'.format(mode)
    #     else:
    #         return '{}_polygons.json'.format(mode)

    
    # images_path, labels_path = [], []
    # split = 'train'
    # images_dir = os.path.join(root_dir, 'leftImg8bit', split)
    # targets_dir = os.path.join(root_dir, 'gtFine', split)
    # for city in os.listdir(images_dir):
    #     img_dir = os.path.join(images_dir, city)
    #     target_dir = os.path.join(targets_dir, city)
    #     for file_name in os.listdir(img_dir):
    #         target_types = []
    #         for t in ['semantic']:
    #             target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
    #                                             _get_target_suffix('gtFine', t))
    #             target_types.append(os.path.join(target_dir, target_name))

    #         images_path.append(os.path.join(img_dir, file_name))
    #         labels_path.append(target_types[0])
            
    # # print(images_path[0: 5], '\n', labels_path[0: 5])
    
    # target_coco_ann_path = '/data/zql/datasets/cityscape/coco_ann.json'
    # # if os.path.exists(target_dir):
    # #     shutil.rmtree(target_dir)
        
    # need_classes_idx = [26, 28]
    # convert_seg_dataset_to_det(
    #     seg_imgs_path=images_path,
    #     seg_labels_path=labels_path,
    #     root_dir=root_dir,
    #     target_coco_ann_path=target_coco_ann_path,
    #     ignore_classes_idx=[i for i in range(80) if i not in need_classes_idx],
    #     # num_threads=8
    #     thread_i=0
    # )
    
    # import shutil
    
    # ignore_target_dir = '/data/zql/datasets/cityscapes_for_cls_task_ignored'
    
    # ignore_label = 255
    # raw_idx_map_in_y_transform = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
    #         3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
    #         7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
    #         14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
    #         18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    #         28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
    # ignore_classes_idx = [k for k, v in raw_idx_map_in_y_transform.items() if v == ignore_label]
    # ignore_classes_idx = sorted(ignore_classes_idx)
    
    # for class_dir in os.listdir(target_dir):
    #     if int(class_dir) in ignore_classes_idx:
    #         continue
    #         shutil.move(
    #             os.path.join(target_dir, class_dir),
    #             os.path.join(ignore_target_dir, class_dir)
    #         )
    #     else:
    #         shutil.move(
    #             os.path.join(target_dir, class_dir),
    #             os.path.join(target_dir, str(raw_idx_map_in_y_transform[int(class_dir)]))
    #         )
    #         continue
    #     print(class_dir)
    # exit()
    
    
    
    # baidu person
    # root_dir = '/data/zql/datasets/baidu_person/clean_images/'
    
    # images_path, labels_path = [], []
    # for p in os.listdir(os.path.join(root_dir, 'images')):
    #     images_path += [os.path.join(root_dir, 'images', p)]
    #     labels_path += [os.path.join(root_dir, 'profiles', p.split('.')[0] + '-profile.jpg')]
    
    # target_dir = '/data/zql/datasets/baiduperson_for_cls_task'
    # # if os.path.exists(target_dir):
    # #     shutil.rmtree(target_dir)
        
    # def label_after_hook(x):
    #     x[x > 1] = 1
    #     return x    
    
    # convert_seg_dataset_to_det(
    #     seg_imgs_path=images_path,
    #     seg_labels_path=labels_path,
    #     root_dir=root_dir,
    #     target_coco_ann_path='/data/zql/datasets/baidu_person/clean_images/coco_ann_zql.json',
    #     ignore_classes_idx=[1],
    #     # num_threads=8
    #     thread_i=1,
    #     min_img_size=224,
    #     label_after_hook=label_after_hook
    # )
    
    
    # from data.visualize import visualize_classes_in_object_detection
    # from data import get_dataset
    # d = get_dataset('CityscapesDet', '/data/zql/datasets/cityscape/', 'val', None, [], None)
    # visualize_classes_in_object_detection(d, {'car': 0, 'bus': 1}, {}, 'debug.png')
    
    # d = get_dataset('GTA5Det', '/data/zql/datasets/GTA-ls-copy/GTA5', 'val', None, [], None)
    # visualize_classes_in_object_detection(d, {'car': 0, 'bus': 1}, {}, 'debug.png')
    
    # d = get_dataset('BaiduPersonDet', '/data/zql/datasets/baidu_person/clean_images/', 'val', None, [], None)
    # visualize_classes_in_object_detection(d, {'person': 0}, {}, 'debug.png')
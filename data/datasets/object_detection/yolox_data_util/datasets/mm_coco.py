import torch
from .datasets_wrapper import Dataset
from .coco_classes import COCO_CLASSES
from maskrcnn_benchmark.structures.bounding_box import BoxList

def cxcywh2xyxy(bboxes):
    #bboxes:(cx,cy,w,h)
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5
    bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
    return bboxes

class MM_COCODataset(Dataset):
    def __init__(self, dataset, transform, split, coco=None, classes=None):
        self.cocods = dataset
        self.coco = coco
        self.transform = transform
        self.split = split
        catIds = self.coco.getCatIds()
        cats = self.coco.loadCats(catIds)
        if classes is None:
            self.cls_names = [cat['name'] for cat in cats]
        else:
            self.cls_names = classes

    def __getitem__(self, index):
        if self.split == 'train':
            img, target = self.cocods[index]
        else:
            img, target, img_size, idx = self.cocods[index]

        img = self.transform(torch.Tensor(img))

        # self.json_category_id_to_contiguous_id = {
        #     v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        # }
        # self.contiguous_category_id_to_json_id = {
        #     v: k for k, v in self.json_category_id_to_contiguous_id.items()
        # }

        num_bbox = 0
        for item in target:
            if item[1] == 0 and item[2] == 0 and item[3] == 0 and item[4] == 0:
                break
            num_bbox += 1
            
        # [0:'zebra', 
        # 1:'backpack', 
        # 2:'umbrella', 
        # 3:'kite', 
        # 4:'cup', 
        # 5:'banana', 
        # 6:'orange', 
        # 7:'broccoli', 
        # 8:'pizza', 
        # 9:'laptop', 
        # 10:'mouse', 
        # 11:'microwave', 
        # 12:'toaster', 
        # 13:'refrigerator', 
        # 14:'vase']
        if self.split == 'train':
            new_target = BoxList(cxcywh2xyxy(target[:num_bbox, 1:]), self.cocods.input_dim, mode='xyxy')
        else:
            new_target = BoxList(target[:num_bbox, 1:], self.cocods.img_size, mode='xyxy')
        
        labels = target[:num_bbox, 0]
        my_labels = []

        caption_string = ". ".join(self.cls_names) + ". "
        tokens_positive = []

        for label in labels:
            word = self.cls_names[int(label)]
            st = caption_string.find(word)
            my_labels.append(label + 1)
            tokens_positive.append([[st, st + len(word)]])

        new_target.add_field('labels', torch.LongTensor(my_labels))
        new_target.add_field('caption', caption_string)
        new_target.add_field('tokens_positive', tokens_positive)
        if self.split == 'train':
            return img, new_target
        else:
            return img, new_target, img_size, idx
    
    def __len__(self):
        return self.cocods.__len__()
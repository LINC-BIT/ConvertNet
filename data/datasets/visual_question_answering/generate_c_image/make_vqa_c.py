import os
import numpy as np
from PIL import Image
from data.datasets.visual_question_answering.generate_c_image.imagenet_c import corrupt


imgs_dir = '/data/zql/datasets/vqav2/train2014'

for img_p in os.listdir(imgs_dir):
    img = np.array(Image.open(os.path.join(imgs_dir, img_p)))
    print(img.shape)

    c_img = corrupt(img, severity=1, corruption_name='gaussian_noise')
    print(c_img.shape)
    
    break
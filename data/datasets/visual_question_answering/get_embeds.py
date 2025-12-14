from .utils import Config
from .modeling_frcnn import GeneralizedRCNN
from .processing_image import Preprocess
from PIL import Image
from torchvision.transforms import Resize
import torch

def get_visual_embeds(image_path):
    config = Config.from_pretrained('data/datasets/visual_question_answering/frcnn')
    frcnn = GeneralizedRCNN.from_pretrained('data/datasets/visual_question_answering/frcnn',config = config)
    image_preprocess = Preprocess(config)
    # image = Image.open('2.jpg').convert('RGB')
    # image = Resize((224,224))(image)
    images, sizes, scales_yx = image_preprocess(image_path)
    output_dict = frcnn(images,sizes,scales_yx=scales_yx,padding="max_detections",max_detections=config.max_detections,return_tensors = "pt")
    features = output_dict.get("roi_features")
    return features
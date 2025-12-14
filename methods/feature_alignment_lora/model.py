from typing import List
import torch
from methods.base.model import BaseModel
import tqdm
from torch import nn
import torch.nn.functional as F
from abc import abstractmethod

from utils.dl.common.model import LayerActivation
from .mmd import mmd_rbf
from ..feature_alignment import FeatureAlignmentModel


class FeatureAlignmentLoraModel(FeatureAlignmentModel):
    pass
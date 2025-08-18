import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.append(project_root)

import torch
from trainers.trainer import trainer
from models.inception3 import Inception3
from models.inception_resnetv2 import InceptionResNetV2
from models.resnet import resnet50, resnet34, wide_resnet50_2
from models.googlenet import GoogLeNet
from models.squeezenet import SqueezeNet
from models.wideresnet import wide_resnet50_2_v2, wide_resnet34_2_v2
from models.densenet import densenet121, densenet161, densenet169, densenet201

model = densenet201
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
trainer(device=device, model=model, need_seed=True)
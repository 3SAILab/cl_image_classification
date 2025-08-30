import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.append(project_root)

import torch
from trainers.trainer import trainer
# from models.alexnet import AlexNet
# from models.inception3 import Inception3
# from models.inception_resnetv2 import InceptionResNetV2
# from models.resnet import resnet50, resnet34, wide_resnet50_2, resnext50_32x4d, resnext101_32x8d, se_resnet50, cbam_resnet50, eca_resnet50
# from models.googlenet import GoogLeNet
# from models.squeezenet import SqueezeNet
# from models.wideresnet import wide_resnet50_2_v2, wide_resnet34_2_v2
# from models.densenet import densenet121, densenet161, densenet169, densenet201
# from models.xception import Xception
# from models.mobilenet import MobileNetV1
# from models.mobilenetv2 import MobileNetV2
# from models.effnet import EFFNet
# from models.shufflenetv1 import ShuffleNetV1
# from models.shufflenetv2 import shufflenetv2_x2_0
# from models.efficientnetv1 import efficientnet_b0, efficientnet_b3
# from models.mobilenetv3 import mobilenetv3_small, mobilenetv3_large
# from models.hardnet import hardnet68, hardnet85
# from models.ghostnetv1 import ghostnetv1
# from models.resnest import resnest50
from models.new_densenet import new_densenet161

model = new_densenet161
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
data_name = "my_data"
log_name = "{}_{}".format(model.__name__, 1)
trainer(device=device, model=model, data_name=data_name, log_name=log_name, need_seed=True, alpha=0.4, nlp=False)
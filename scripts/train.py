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
# from models.resnet import resnet50, resnet34, wide_resnet50_2, resnext50_32x4d
# from models.googlenet import GoogLeNet
# from models.squeezenet import SqueezeNet
# from models.wideresnet import wide_resnet50_2_v2, wide_resnet34_2_v2
# from models.densenet import densenet121, densenet161, densenet169, densenet201
# from models.xception import Xception
# from models.new_densenet import new_densenet161
from models.vit import vit_base_patch16_224, vit_large_patch16_224
from models.vggnet import vgg

model = vgg
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
log_name = "{}_{}".format(model.__name__, 1)
weights = None
trainer(device=device, model=model, log_name=log_name, need_seed=True, alpha=0.4, weights=weights, freeze_layers=False)
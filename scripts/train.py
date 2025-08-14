import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.append(project_root)

import torch
from trainers.trainer import trainer
from models.googlenet import GoogLeNet

model = GoogLeNet
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
trainer(device=device, batch_size=32, model=model, epochs=30)
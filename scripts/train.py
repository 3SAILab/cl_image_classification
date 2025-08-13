import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.append(project_root)

from trainers.trainer import trainer
from models.goolenet import GoogLeNet

model = GoogLeNet
trainer(cuda=2, batch_size=32, model=model, epochs=3, num_classes=5)
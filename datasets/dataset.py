import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.append(project_root)

from torchvision import datasets
from datasets.transform import transform

def train_dataset(root):
    return datasets.ImageFolder(root=root,
                                transform=transform["train"])

def val_dataset(root):
    return datasets.ImageFolder(root=root,
                                transform=transform["val"])

def test_dataset(img):
    return transform["val"](img)
                               
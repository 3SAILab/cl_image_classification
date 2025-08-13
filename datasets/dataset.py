import sys

sys.path.append("/mnt/driver_g/chenlong/image_classification_project")

from torchvision import datasets
from datasets.transform import transform

def train_dataset(root):
    return datasets.ImageFolder(root=root,
                                transform=transform["train"])

def val_dataset(root):
    return datasets.ImageFolder(root=root,
                                transform=transform["val"])

def test_dataset(root):
    return datasets.ImageFolder(root=root,
                                transform=transform["val"])
                               
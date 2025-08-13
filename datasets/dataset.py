from torchvision import datasets
from transforms import transform

def train_dataset(root):
    return datasets.ImageFolder(root=root,
                                transform=transform["train"])

def val_dataset(root):
    return datasets.ImageFolder(root=root,
                                transform=transform["val"])

def test_dataset(root):
    return datasets.ImageFolder(root=root,
                                transform=transform["val"])
                               
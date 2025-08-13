import sys

sys.path.append("/mnt/driver_g/chenlong/image_classification_project")

from trainers.trainer import trainer
from models.GoogLeNet import GoogLeNet

model = GoogLeNet
trainer(cuda=2, batch_size=32, model=model, epochs=3, num_classes=5)
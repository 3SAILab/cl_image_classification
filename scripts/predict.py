import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.append(project_root)

import torch
from datasets import dataset
from PIL import Image
import json
import matplotlib.pyplot as plt
import yaml

from models.googlenet import GoogLeNet

def predict(device, model, image_path):
    print("using {} device.".format(device))

    img = Image.open(image_path)
    img = dataset.test_dataset(img)
    img = torch.unsqueeze(img,dim=0)

    json_path = "configs/class_indices.json"
    assert os.path.exists(json_path),"file: '{}' dose not exist.".format(json_path)
    with open(json_path,"r") as f:
        class_indict = json.load(f)

    config_path = "configs/config.yaml"
    assert os.path.exists(config_path),"file {} is not exist.".format(config_path)
    with open(config_path,"r") as f:
        config = yaml.safe_load(f)

    model_config = config[model.__name__]

    net = model(model_config).to(device)
    weights_path = "checkpoints/{}_LSR.pth".format(model.__name__)
    assert os.path.exists(weights_path),"file: '{}' dose not exist.".format(weights_path)
    net.load_state_dict(torch.load(weights_path, map_location=device))

    net.eval()
    with torch.no_grad():
        output = torch.squeeze(net(img.to(device))).cpu()
        predict = torch.softmax(output,dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()

if __name__ == "__main__":
    image_path = "data/test/roses2.jpg"
    assert os.path.exists(image_path),"file {} is not exist.".format(image_path)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    predict(device=device, model=GoogLeNet, image_path=image_path)
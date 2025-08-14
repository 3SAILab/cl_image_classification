import torch
from datasets import dataset
from PIL import Image
import os
import json
import matplotlib.pyplot as plt

from models.googlenet import GoogLeNet

def predict(model, image_path, num_classes):
    print("using {} device.".format(device))

    img = Image.open(image_path)
    img = dataset.test_dataset(img)
    img = torch.unsqueeze(img,dim=0)

    json_path = "../configs/class_indices.json"
    assert os.path.exists(json_path),"file: '{}' dose not exist.".format(json_path)
    with open(json_path,"r") as f:
        class_indict = json.load(f)

    model = model(num_classes=num_classes, aux_logits=False).to(device)
    weights_path = "../checkpoints/{}.pth".format(model.__name__)
    assert os.path.exists(weights_path),"file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
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
    image_path = "../data/test/roses.jpg"
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    predict(device=device, model=GoogLeNet, image_path=image_path, num_classes=5)
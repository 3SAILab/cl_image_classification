import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.append(project_root)

import torch.nn as nn
import torch
import os
import json
from datasets import dataset,dataloader
import sys
from tqdm import tqdm
import torch.optim as optim
from utils.visualization import plot_loss_curves,plot_accuracy_curves


def trainer(cuda, batch_size, model, epochs, num_classes):
    device = torch.device('cuda:{}'.format(cuda) if torch.cuda.is_available() else 'cpu')
    print("using {} device.".format(device))

    data_root = os.path.abspath(os.getcwd())
    image_path = os.path.join(data_root, "data")
    assert os.path.exists(image_path),"file {} does not exist.".format(image_path)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0,8])
    print("using {} dataloader workers every process.".format(nw))

    train_dataset = dataset.train_dataset(os.path.join(image_path, "train"))
    train_num = len(train_dataset)
    train_loader =  dataloader.train_loader(train_dataset, batch_size,nw)

    val_dataset =  dataset.val_dataset(os.path.join(image_path, "val"))
    val_num = len(val_dataset)
    val_loader =  dataloader.val_loader(val_dataset, batch_size,nw)

    print("using {} images for training, using {} images for validation.".format(train_num, val_num))

    net = model(num_classes=num_classes, aux_logits=True, init_weights=True)

    net.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.0001)

    best_acc = 0.0
    save_path = "checkpoints/{}.pth".format(model.__name__)
    train_steps = len(train_loader)
    loss_list = []
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file = sys.stdout)
        for step,data in enumerate(train_bar):
            images,labels = data
            optimizer.zero_grad()
            if model.__name__ == "GoogLeNet":
                logits, aux_logits1, aux_logits2 = net(images.to(device))
                loss0 = loss_function(logits, labels.to(device))
                loss1 = loss_function(aux_logits1, labels.to(device))
                loss2 = loss_function(aux_logits2, labels.to(device))
                loss = loss0 + loss1*0.3 + loss2*0.3
            else:
                outputs = net(images.to(device))
                loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch [{}/{}] loss: {:.3f}".format(epoch + 1,
                                                                       epochs,
                                                                       loss)

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                images, labels = val_data
                outputs = net(images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, labels.to(device)).sum().item()

        val_accuracy = acc / val_num
        print("epoch: {} train_loss: {:.3f} val_accuracy: {:.3f}".format(epoch + 1,
                                                                         running_loss / train_steps,
                                                                         val_accuracy))

        loss_list.append(running_loss / train_steps)

        if best_acc < val_accuracy:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)
    
    print("Finished Training.")

    loss_save_path = "results/{}_loss.jpg".format(model.__name__)
    plot_loss_curves(loss_list, epochs, loss_save_path)

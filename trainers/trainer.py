import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.append(project_root)

import torch
import os
import json
from datasets import dataset, dataloader, transform
import sys
from tqdm import tqdm
import torch.optim as optim
from utils.visualization import plot_training_curves
import yaml
import numpy as np
import random
import time


def trainer(device, model, need_seed=False):
    # 选择设备
    print("using {} device.".format(device))

    # 设置数据集路径
    data_root = os.path.abspath(os.getcwd())
    image_path = os.path.join(data_root, "data", "my_data")
    assert os.path.exists(image_path),"file {} does not exist.".format(image_path)

    # 导入相关参数
    config_path = os.path.join(data_root, "configs", "config.yaml")
    assert os.path.exists(config_path),"file {} is not exist.".format(config_path)
    with open(config_path,"r") as f:
        config = yaml.safe_load(f)

    model_config = config[model.__name__]
    train_config = config['train']
    batch_size = train_config.get('batch_size')
    epochs = train_config.get('epochs')

    # 是否固定种子
    if need_seed:
        seed = train_config.get('seed')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # 设置并行进程数
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0,8])
    print("using {} dataloader workers every process.".format(nw))

    # 数据加载和预处理
    train_dataset = dataset.train_dataset(os.path.join(image_path, "train"))
    train_num = len(train_dataset)
    train_loader =  dataloader.train_loader(train_dataset, batch_size,nw)

    val_dataset =  dataset.val_dataset(os.path.join(image_path, "val"))
    val_num = len(val_dataset)
    val_loader =  dataloader.val_loader(val_dataset, batch_size,nw)

    print("using {} images for training, using {} images for validation.".format(train_num, val_num))

    # 类别标签写入json文件
    image_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in image_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open("configs/my_class_indices.json","w") as json_file:
        json_file.write(json_str)

    # 导入模型
    net = model(model_config)
    net.to(device)
    total_params = sum(p.numel() for p in net.parameters())

    # 设置损失函数和优化器
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(net.parameters(), lr = 0.0001)

    # 训练
    best_acc = 0.0
    save_name = "{}.pth".format(model.__name__)
    save_path = os.path.join(data_root, "checkpoints", save_name)
    train_steps = len(train_loader)
    val_steps = len(val_loader)
    loss_list = []
    accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []
    total_train_start = time.perf_counter()
    
    for epoch in range(epochs):
        net.train()
        epoch_train_start = time.perf_counter()
        running_loss = 0.0
        running_acc = 0.0
        train_bar = tqdm(train_loader, file = sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # 数据增强mixup
            images, labels_a, labels_b, lam = transform.mixup_data(images, labels, device, alpha=0.4)

            optimizer.zero_grad()
            if model.__name__ == "GoogLeNet":
                logits, aux_logits1, aux_logits2 = net(images)
                train_predict_y = torch.max(logits, dim=1)[1]
                loss0 = transform.mixup_criterion(loss_function, logits, labels_a, labels_b, lam)
                loss1 = transform.mixup_criterion(loss_function, aux_logits1, labels_a, labels_b, lam)
                loss2 = transform.mixup_criterion(loss_function, aux_logits2, labels_a, labels_b, lam)
                loss = loss0 + loss1*0.3 + loss2*0.3
            elif model.__name__ == "Inception3":
                logits, aux_logits = net(images)
                train_predict_y = torch.max(logits, dim=1)[1]
                loss0 = transform.mixup_criterion(loss_function, logits, labels_a, labels_b, lam)
                loss1 = transform.mixup_criterion(loss_function, aux_logits, labels_a, labels_b, lam)
                loss = loss0 + loss1*0.3
            else:
                outputs = net(images)
                train_predict_y = torch.max(outputs, dim=1)[1]

                loss = transform.mixup_criterion(loss_function, outputs, labels_a, labels_b, lam)

            acc = (lam * train_predict_y.eq(labels_a.to(device)).sum().item()
                    + (1 - lam) * train_predict_y.eq(labels_b.to(device)).sum().item())
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += acc
            train_bar.desc = "train epoch [{}/{}] loss: {:.3f}".format(epoch + 1,
                                                                       epochs,
                                                                       loss)

        net.eval()
        val_acc = 0.0
        val_l = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                images, labels = val_data
                outputs = net(images.to(device))
                val_l += loss_function(outputs, labels.to(device)).item()
                predict_y = torch.max(outputs, dim=1)[1]
                val_acc += torch.eq(predict_y, labels.to(device)).sum().item()

        epoch_train_end = time.perf_counter()
        val_accuracy = val_acc / val_num
        val_loss = val_l / val_steps
        print("epoch: {} train_loss: {:.3f} val_accuracy: {:.3f} time: {:.5f}".format(epoch + 1,
                                                                         running_loss / train_steps,
                                                                         val_accuracy,
                                                                         epoch_train_end - epoch_train_start))

        loss_list.append(running_loss / train_steps)
        accuracy_list.append(running_acc / train_num)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)

        if best_acc < val_accuracy:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)
    
    total_train_end = time.perf_counter()
    print("Finished Training.")

    # 计算训练时间
    total_train_time = total_train_end - total_train_start
    print("Total Training Time: {:.5f}".format(total_train_time))

    # 保存训练日志
    train_log_name = "{}.txt".format(model.__name__)
    train_log_path = os.path.join(data_root, "logs", train_log_name)
    
    log_data = {
        "Parameters": total_params,
        "Train Epochs": epochs,
        "Batch Size": batch_size,
        "Initial Learning Rate": 0.0001,
        "Seed": seed,
        "Best Accuracy": best_acc,
        "Train Time": total_train_time,
        "Loss List": loss_list,
        "Accuracy List": accuracy_list,
        "Val Loss List": val_loss_list,
        "Val Accuracy List": val_accuracy_list
    }

    with open(train_log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)

    # 绘制训练损失及准确率曲线图
    plot_training_curves(log_data['Loss List'], log_data['Accuracy List'], log_data['Val Loss List'], log_data['Val Accuracy List'],
                         title="Image Classification Model Performance", name=model.__name__)

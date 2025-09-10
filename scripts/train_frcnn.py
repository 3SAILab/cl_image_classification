import sys
import os

from torch.optim import lr_scheduler

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.append(project_root)

from models.detection.frcnn import FasterRCNN
from trainers.frcnn_trainer import FasterRCNNTrainer, get_lr_scheduler, set_optimizer_lr, weights_init
from datasets.frcnn_dataloader import FRCNNDataset, frcnn_dataset_collate, worker_init_fn

import torch
from torch.utils.data import DataLoader
import datetime
from functools import partial
import numpy as np
import torch.optim as optim
import yaml
import cv2

def train():
    # 选择设备
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print("using {} device.".format(device))

    # 设置数据集路径
    data_root = os.path.abspath(os.getcwd())
    image_path = os.path.join(data_root, "data", "VOCdevkit", "VOC2012")
    assert os.path.exists(image_path),"file {} does not exist.".format(image_path)

    # 导入相关配置参数
    config_path = os.path.join(data_root, "configs", "config.yaml")
    assert os.path.exists(config_path),"file {} is not exist.".format(config_path)
    with open(config_path,"r") as f:
        config = yaml.safe_load(f)

    train_config = config['voc_train']
    lr = train_config.get('lr')
    backbone = train_config.get('backbone')
    pretrained = train_config.get('pretrained')
    Init_Epoch = train_config.get('Init_Epoch')
    Freeze_Epoch = train_config.get('Freeze_Epoch')
    Freeze_batch_size = train_config.get('Freeze_batch_size')
    UnFreeze_Epoch = train_config.get("UnFreeze_Epoch")
    Unfreeze_batch_size = train_config.get("Unfreeze_batch_size")
    Freeze_Train = train_config.get("Freeze_Train")
    seed = train_config.get('seed')

    class_path = os.path.join(data_root, "configs", "voc_classes.txt")
    assert os.path.exists(class_path),"file {} is not exist.".format(class_path)
    with open(class_path, "r") as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]
    num_classes = len(classes)

    # 导入模型
    model = FasterRCNN(
        mode="train",
        num_classes=num_classes,
        feat_stride = 16,
        anchor_scales = [1, 2, 4],
        ratios = [0.5, 1, 2],
        backbone=backbone,
        pretrained=pretrained
    ).to(device)

    # 加载预训练权重
    model_path = os.path.join(data_root, "checkpoints", "vgg16-397923af.pth")
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print(f'>>加载预训练权重文件：{model_path}')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)

        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict and torch.shape(model_dict[k]) == torch.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)

        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )

    # 数据集加载
    train_annotation_path = os.path.join(image_path, "ImageSets", "Main", "train.txt")
    val_annotation_path = os.path.join(image_path, "ImageSets", "Main", "val.txt")
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    UnFreeze_flag = False
    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0,8])
    print("using {} dataloader workers every process.".format(nw))

    train_dataset = FRCNNDataset(train_lines, [600, 600], train=True)
    val_dataset = FRCNNDataset(val_lines, [600, 600], train=False)
    gen = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True,
    drop_last=True, collate_fn=frcnn_dataset_collate, worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
    gen_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True,
    drop_last=True, collate_fn=frcnn_dataset_collate, worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

    # 训练
    save_name = "frcnn.pth"
    save_path = os.path.join(data_root, "checkpoints", save_name)
    loss_list = []
    loss_val_list = []
    trainer = FasterRCNNTrainer(model, optimizer).to(device)
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size
            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            for param in model.extractor.parameters():
                param.requires_grad = True

            gen = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True,
                            drop_last=True, collate_fn=frcnn_dataset_collate, worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
            gen_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True,
                                drop_last=True, collate_fn=frcnn_dataset_collate, worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

            print(f'>>> Epoch {epoch}: Unfreeze backbone and change batch size to {batch_size}')
            UnFreeze_flag = True

        model.train()
        total_loss = 0
        print(f'>>> Start training epoch {epoch + 1}/{UnFreeze_Epoch}')

        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            images, bboxes, labels = batch
            images = images.to(device)
            scale = 1.0

            losses = trainer.train_step(images, bboxes, labels, scale)

            rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss, total = losses
            total_loss += total.item()

            if iteration % 50 == 0:
                print(f'Epoch {epoch + 1}, Iter {iteration}/{epoch_step} || '
                    f'RPN Loc: {rpn_loc_loss.item():.4f}, RPN Cls: {rpn_cls_loss.item():.4f}, '
                    f'ROI Loc: {roi_loc_loss.item():.4f}, ROI Cls: {roi_cls_loss.item():.4f}, '
                    f'Total: {total.item():.4f}')

        scheduler.step()
        

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for iteration, batch in enumerate(gen_val):
                if iteration >= epoch_step_val:
                    break
                images, bboxes, labels = batch
                images = images.to(device)
                scale = 1.0
                losses = trainer.forward(images, bboxes, labels, scale)
                val_loss += losses[-1].item()

        avg_val_loss = val_loss / epoch_step_val
        print(f'>>> Epoch {epoch + 1} | Train Loss: {total_loss / epoch_step:.4f} | Val Loss: {avg_val_loss:.4f}')

        loss_list.append(total_loss / epoch_step)
        loss_val_list.append(avg_val_loss)

        # 保存
        if epoch == 0 or avg_val_loss < min(loss_val_list[:-1] or [float('inf')]):
            torch.save(model.state_dict(), save_path)
            print(f'>>> Save best model to {save_path}')

# 数据检查
def visualize_sample(index):
    # 加载数据
    data_root = os.path.abspath(os.getcwd())
    image_path = os.path.join(data_root, "data", "VOCdevkit", "VOC2012")
    train_annotation_path = os.path.join(image_path, "ImageSets", "Main", "train.txt")
    val_annotation_path = os.path.join(image_path, "ImageSets", "Main", "val.txt")
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()

    train_dataset = FRCNNDataset(train_lines, [600, 600], train=True)
    val_dataset = FRCNNDataset(val_lines, [600, 600], train=False)

    image, bboxes, labels = train_dataset[index]
    img = image.transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    class_path = os.path.join(data_root, "configs", "voc_classes.txt")
    with open(class_path, "r") as f:
        class_names = [c.strip() for c in f.readlines()]

    for i, box in enumerate(bboxes):
        print(box)
        x1, y1, x2, y2 = box

        # 转为整数，防止浮点坐标
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

        # 防止越界 
        h, w = img.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # 获取标签和颜色
        label = int(labels[i])
        class_name = class_names[label] if label < len(class_names) else f"cls{label}"
        color = (0, 255, 0)  

         # 画框 + 标签
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, class_name, (x1, max(5, y1 - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        save_path = f"debug_sample_{index}.jpg"
        cv2.imwrite(save_path, img)
        print(f"✅ 图片已保存到: {save_path}")

if __name__ == '__main__':
    train()

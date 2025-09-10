import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import random
import os
import xml.etree.ElementTree as ET


class FRCNNDataset(Dataset):
    def __init__(self, annotation_lines, input_shape = [600, 600], train = True):
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.train              = train

        data_root = os.path.abspath(os.getcwd())
        self.data_path = os.path.join(data_root, "data", "VOCdevkit", "VOC2012", "JPEGImages")
        self.annotation_path = os.path.join(data_root, "data", "VOCdevkit", "VOC2012", "Annotations")

        class_path = os.path.join(data_root, "configs", "voc_classes.txt")
        with open(class_path, "r") as f:
            self.class_name = [c.strip() for c in f.readlines()]

        self.class_to_idx = {name:idx for idx, name in enumerate(self.class_name)}

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length
        image_id = self.annotation_lines[index].strip()  
        image, box, label = self.get_random_data(image_id, self.input_shape[0:2], random=self.train)
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        return image, box, label

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, image_id, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        image_path = os.path.join(self.data_path, image_id + ".jpg")
        image = Image.open(image_path)
        image = cvtColor(image)
        iw, ih = image.size
        h, w = input_shape

        xml_path = os.path.join(self.annotation_path, image_id + ".xml")
        boxes, labels = self.parse_voc_xml(xml_path) 

        if len(boxes) == 0:
            box = np.zeros((0, 4), dtype=np.float32)
            label = np.zeros((0,), dtype=np.float32)
        else:
            box = boxes.astype(np.float32)
            label = labels.astype(np.float32)

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # 调整真实框
            if len(box) > 0:
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                mask = np.logical_and(box_w > 1, box_h > 1)
                box = box[mask]
                label = label[mask]

            return image_data, box, label
                
        #   对图像进行缩放并且进行长和宽的扭曲
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #   将图像多余的部分加上灰条
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #   翻转图像
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #   对图像进行色域变换
        #   计算色域变换的参数
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #   将图像转到HSV上
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #   应用变换
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #   对真实框进行调整
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return image_data, box, label

    def parse_voc_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.class_to_idx:
                continue  # 跳过未知类别

            label = self.class_to_idx[name]

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        if len(boxes) == 0:
            return np.zeros((0, 4)), np.zeros((0,))
        return np.array(boxes), np.array(labels)

def frcnn_dataset_collate(batch):
    images = []
    bboxes = []
    labels = []
    for img, box, label in batch:
        images.append(img)
        bboxes.append(box)
        labels.append(label)
    images = torch.from_numpy(np.array(images))
    return images, bboxes, labels

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

def preprocess_input(image):
    image /= 255.0
    return image

def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

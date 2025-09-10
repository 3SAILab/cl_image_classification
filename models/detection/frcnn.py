import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.append(project_root)

import torch.nn as nn

from detection.classifier import VGG16RoIHead, Resnet50RoIHead
from detection.vgg16 import decom_vgg16
from detection.rpn import RegionProposalNetwork
from detection.resnet50 import resnet50

class FasterRCNN(nn.Module):
    def __init__(self, num_classes,
                    mode="training",
                    feat_stride=16,
                    anchor_scales=[8, 16, 32],
                    ratios = [0.5, 1, 2],
                    backbone = 'vgg',
                    pretrained = False):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride
        if backbone == 'vgg':
            self.extractor, classifier = decom_vgg16(pretrained)
            # 构建建议框网络
            self.rpn = RegionProposalNetwork(
                512, 512,
                ratios = ratios,
                anchor_scales = anchor_scales,
                feat_stride = self.feat_stride,
                mode = mode
            )
            # 构建分类器网络
            self.head = VGG16RoIHead(
                n_class = num_classes + 1,
                roi_size = 7,
                spatial_scale = 1,
                classifier = classifier
            )
        elif backbone == 'resnet50':
            self.extractor, classifier = resnet50(pretrained)
            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios          = ratios,
                anchor_scales   = anchor_scales,
                feat_stride     = self.feat_stride,
                mode            = mode
            )
            self.head = Resnet50RoIHead(
                n_class         = num_classes + 1,
                roi_size        = 14,
                spatial_scale   = 1,
                classifier      = classifier
            )
            
    def forward(self, x, scale=1, mode="forward"):
        if mode == "forward":
            # 输入图片大小
            img_size = x.shape[2:]
            # 利用主干网络提取特征
            base_feature = self.extractor.forward(x)
            # 获得建议框
            _, _, rois, roi_indices, _ = self.rpn.forward(base_feature, img_size, scale)
            # 获得calssifier的分类结果和回归结果
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores, rois, roi_indices
        
        elif mode == "extractor":
            base_feature = self.extractor.forward(x)
            return base_feature

        elif mode == "rpn":
            base_feature, img_size = x
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
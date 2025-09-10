import warnings

import torch
from torch import nn
from torchvision.ops import RoIPool

warnings.filterwarnings("ignore")

class VGG16RoIHead(nn.Module):
    def __init__(
        self,
        n_class,     # 类别总数，num_classes + 1
        roi_size,    # 特征图尺寸
        spatial_scale,   # 缩放比例
        classifier  
        ):
        super(VGG16RoIHead, self).__init__()
        self.classifier = classifier
        # 对RoIPooling的结果进行回归预测
        self.cls_loc = nn.Linear(4096, n_class * 4)
        # 对RoIPooling的结果进行分类预测
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        roi_indices = roi_indices.to(x.device)
        rois = rois.to(x.device)
        rois = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)

        # 将RoI坐标从原始图像空间映射到特征图空间
        # 按比例缩放
        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]

        # [N * num_rois, 5]
        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1).to(x.device)

        # 利用建议框对公用特征层进行截取
        pool = self.roi(x, indices_and_rois)

        # 利用classifier进行网络特征提取
        pool = pool.view(pool.size(0), -1) # [N * num_rois, C * roi_size * roi_size]
        fc7 = self.classifier(pool) # [N * num_rois, 4096]

        roi_cls_locs = self.cls_loc(fc7) # [N * num_rois, n_class * 4]
        roi_scores = self.score(fc7) # [N * num_rois, n_class]

        roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.size(1)) # [N, num_rois, n_class * 4]
        roi_scores = roi_scores.view(n, -1, roi_scores.size(1))  # [N, num_rois, n_class]
        return roi_cls_locs, roi_scores

class Resnet50RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(Resnet50RoIHead, self).__init__()
        self.classifier = classifier
        self.cls_loc = nn.Linear(2048, n_class * 4)
        self.score = nn.Linear(2048, n_class)
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        roi_indices = roi_indices.to(x.device)
        rois = rois.to(x.device)
        rois = torch.flatten(rois, 0, 1)

        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1).to(x.device)

        pool = self.roi(x, indices_and_rois)
        fc7 = self.classifier(pool)
        fc7 = fc7.view(fc7.size(0), -1)

        roi_cls_locs    = self.cls_loc(fc7)
        roi_scores      = self.score(fc7)
        roi_cls_locs    = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores      = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import nms

from utils.anchor import generate_anchor_base, _enumerate_shifted_anchor
from utils.utils_bbox import loc2bbox

class ProposalCreator():
    def __init__(
        self,
        mode,   # 预测还是训练
        nms_iou=0.7, # 非极大抑制的iou大小
        n_train_pre_nms=12000,    # 训练nms前的框数量
        n_train_post_nms=600,     # 训练nms后的框数量
        n_test_pre_nms=3000,      # 预测nms前的框数量
        n_test_post_nms=300,      # 预测nms后的框数量
        min_size=16
    ):
        self.mode = mode
        self.nms_iou = nms_iou
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1):
        if self.mode == "training":
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # 将先验框转为tensor
        anchor = torch.from_numpy(anchor).type_as(loc)
        # 将RPN结果转为建议框
        roi = loc2bbox(anchor, loc)
        # 防止建议框超出图像边缘
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min = 0, max = img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min = 0, max = img_size[0])

        # 建议框的最小值不能小于16
        # min_size缩放统一
        min_size = self.min_size * scale
        # 选出所有大于最小值的索引
        keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1] >= min_size)))[0]
        # 保留对应的框
        roi = roi[keep, :]
        score = score[keep]

        # 根据得分进行排序，取出建议框
        order = torch.argsort(score, descending=True)
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # 对建议框进行非极大抑制（使用官方的）
        keep = nms(roi, score, self.nms_iou)
        # 不足n_post_nms随机重复补充
        if len(keep) < n_post_nms:
            if len(keep) == 0:
                print("[WARNING] No valid proposals after NMS. Returning a default ROI.")
                default_box = [0, 0, min(16, img_size[1]), min(16, img_size[0])]
                roi = torch.tensor([default_box] * n_post_nms, device=roi.device, dtype=roi.dtype)
                return roi
            else:
                index_extra = np.random.choice(range(len(keep)), size=(n_post_nms - len(keep)), replace=True)
                keep = torch.cat([keep, keep[index_extra]])
        keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi

class RegionProposalNetwork(nn.Module):
    def __init__(
        self,
        in_channels=512,
        mid_channels=512,
        ratios = [0.5, 1, 2], # anchor的宽高比例
        anchor_scales = [8, 16, 32], # anchor基础尺度
        feat_stride = 16, # 特征图与原始图像比例
        mode = "training",
        ):
        super(RegionProposalNetwork, self).__init__()

        # 生成基础先验框,shape[9, 4] (9个框，4个点位置)
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        n_anchor = self.anchor_base.shape[0] # anchor数量

        # 3x3卷积
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)

        # 分类预测，是否包含目标
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1)
        
        # 回归预测，调整先验框
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1)

        # 特征点间距步长
        self.feat_stride = feat_stride

        # 非极大抑制，去除重叠，超出边框的
        self.proposal_layer = ProposalCreator(mode)

        # 初始化权重
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1):
        n, _, h, w = x.shape

        x = F.relu(self.conv1(x))

        # 分支1.分类预测
        rpn_scores = self.score(x)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2) # [B, anchor_num, 2]

        # 计算前景（目标）概率
        rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous().view(n, -1) # [B, anchor_num]

        # 分支2.回归预测
        rpn_locs = self.loc(x)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        # 生成先验框
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)
        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale = scale)
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi.unsqueeze(0))
            roi_indices.append(batch_index.unsqueeze(0))

        # [B, 300, 4]
        rois = torch.cat(rois, dim=0).type_as(x)
        # [B, 300]
        roi_indices = torch.cat(roi_indices, dim=0).type_as(x)
        # [1, H*W*9, 4]
        anchor = torch.from_numpy(anchor).unsqueeze(0).float().to(x.device)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).nul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()




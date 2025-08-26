import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.append(project_root)

import torch
import torch.nn as nn
from models.squeezeexcitation import SqueezeExcitation
from models.cbam import CBAM
from models.eca import ECA

def conv3x3(in_channels, out_channels, stride=1, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=False
    )

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False
    )

class BasicBlock(nn.Module):
    expansion = 1 #主分支的卷积核个数是否相同
    def __init__(
        self, 
        in_channel, 
        out_channel, 
        stride=1, 
        downsample=None,
        groups = 1, 
        base_width=64,
        with_se=False):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.stride=stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(
        self, 
        in_channel, 
        out_channel, 
        stride=1, 
        downsample=None,
        groups=1, 
        base_width=64,
        with_se = False,
        with_cbam = False,
        with_eca = False
        ):
        super(Bottleneck,self).__init__()
        width = int(out_channel * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(in_channel, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, out_channel * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if with_se:
            self.se = SqueezeExcitation(out_channel*self.expansion,
                                        (out_channel*self.expansion)//16)
        else:
            self.se = None

        if with_cbam:
            self.cbam = CBAM(out_channel*self.expansion)
        else:
            self.cbam = None

        if with_eca:
            self.eca = ECA(out_channel*self.expansion)
        else:
            self.eca = None

        self.downsample = downsample
        self.stride=stride

    def forward(self, x):
        indentity = x
        if self.downsample is not None:
            indentity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se:
            out = self.se(out)

        if self.cbam:
            out = self.cbam(out)

        if self.eca:
            out = self.eca(out)

        out += indentity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self, 
        block, 
        block_num, #block_num参数列表
        num_classes=1000, 
        include_top=True,
        groups=1, 
        width_per_group=64,
        with_se=False,
        with_cbam=False,
        with_eca=False
        ): 
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.base_width = width_per_group
        self.groups = groups
        self.se = with_se
        self.cbam = with_cbam
        self.eca = with_eca

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channel, channel * block.expansion, stride),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        layers.append(
            block(
                self.in_channel, channel, stride, downsample, self.groups, self.base_width, self.se, self.cbam
                )
        )
        self.in_channel = channel*block.expansion

        for _ in range(1, block_num):
            layers.append(
                block(
                    self.in_channel, channel, groups=self.groups, base_width=self.base_width, with_se=self.se, with_cbam=self.cbam, with_eca=self.eca
                    )
            )

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x
        
def resnet34(model_config):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=model_config.get('num_classes'), include_top=model_config.get('include_top'))

def resnet101(model_config):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=model_config.get('num_classes'), include_top=model_config.get('include_top'))

def resnet50(model_config):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=model_config.get('num_classes'), include_top=model_config.get('include_top'))

def wide_resnet50_2(model_config):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=model_config.get('num_classes'), width_per_group=model_config.get("width_per_group"))

def resnext50_32x4d(model_config):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=model_config.get('num_classes'), groups=model_config.get("groups"),
                  width_per_group=model_config.get("width_per_group"))

def resnext101_32x8d(model_config):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=model_config.get('num_classes'), groups=model_config.get("groups"),
                  width_per_group=model_config.get("width_per_group"))

def se_resnet50(model_config):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=model_config.get('num_classes'), with_se=model_config.get("with_se"))

def cbam_resnet50(model_config):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=model_config.get('num_classes'), with_cbam=model_config.get("with_cbam"))

def eca_resnet50(model_config):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=model_config.get('num_classes'), with_eca=model_config.get("with_eca"))
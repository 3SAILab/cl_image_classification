import torch
from torch._functorch._aot_autograd.schemas import InputAliasInfo
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, dilation=1, groups=1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, reps, stride=1, start_with_relu=True, grow_first=True):
        '''
        reps: 深度可分离卷积堆叠数量
        start_with_relu: 是否在block开头加relu
        grow_first: 是否先增加通道数
        '''
        super(Block, self).__init__()

        if out_channels != in_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

        rep = []

        channels = in_channels
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_channels, out_channels, kernel_size=3,
                                       stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_channels))
            channels = out_channels
        
        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(channels, channels, kernel_size=3, stride=1,
                                       padding=1, bias=False))
            rep.append(nn.BatchNorm2d(channels))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_channels, out_channels, kernel_size=3,
                                       stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_channels))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=True)

        if stride != 1:
            rep.append(nn.MaxPool2d(3, stride, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, input):
        x = self.rep(input)

        if self.downsample:
            dentity = self.downsample(input)
        else:
            dentity = input

        x += dentity
        return x

class Xception(nn.Module):
    def __init__(self, model_config):
        super(Xception, self).__init__()
        self.num_classes = model_config.get("num_classes")

        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3=Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # Middle flow
        self.block4=Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5=Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6=Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7=Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8=Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9=Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10=Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11=Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, self.num_classes)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = nn.ReLU(inplace=True)(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    
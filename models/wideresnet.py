from re import X
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1 #主分支的卷积核个数是否相同
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, base_width=64): #downsample下采样参数
        super(BasicBlock,self).__init__()
        width = int(out_channel * (base_width / 64.0))
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(width)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
       
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out += identity

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, base_width=64):
        super(Bottleneck,self).__init__()
        width = int(out_channel * (base_width / 64.0))
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(width)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width,
                               kernel_size=3, stride=stride, bias=False,padding=1)
        
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)
        
        self.downsample = downsample

    def forward(self, x):
        indentity = x
        if self.downsample is not None:
            indentity = self.downsample(x)
        
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)
        
        out += indentity

        return out

class ResNet(nn.Module):
    def __init__(self, block, block_num, num_classes=1000, include_top=True, width_per_group=64): #block_num参数列表
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.base_width = width_per_group

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
                nn.Conv2d(self.in_channel, channel*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel*block.expansion)
            )

        layers = []
        layers.append(block(self.in_channel, channel, stride, downsample, self.base_width))
        self.in_channel = channel*block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,channel))

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

def wide_resnet50_2_v2(model_config):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=model_config.get('num_classes'), include_top=model_config.get('include_top'),
                  width_per_group=model_config.get("width_per_group"))

def wide_resnet34_2_v2(model_config):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=model_config.get('num_classes'), include_top=model_config.get('include_top'),
                  width_per_group=model_config.get("width_per_group"))


import torch
import torch.nn as nn 
from torch import Tensor
from collections import OrderedDict
import torch.nn.functional as F

# 密集层
class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size):
        '''
        num_input_features: 输入特征通道数
        growth_rate: 固定学习量，即当前层的输出特征通道数
        bn_size: 瓶颈层的缩放因子
        '''
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size*growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size*growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size*growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        
        self.dropout = nn.Dropout(p=0)

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(concated_features))
        return bottleneck_output

    def forward(self, input):
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.bn2(bottleneck_output)))
        new_features = self.dropout(new_features)
        return new_features

# 密集块
class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate,
                               growth_rate=growth_rate,
                               bn_size=bn_size)
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

# 过渡层
class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1,
                              stride=1, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseNet(nn.Module):
    def __init__(self, num_block, num_init_features=64, bn_size=4, growth_rate=32, num_classes=1000):
        super(DenseNet, self).__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("bn0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        num_features = num_init_features
        for i, num_layers in enumerate(num_block):
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(num_block) - 1:
                trans = Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2
        
        self.features.add_module("bn5", nn.BatchNorm2d(num_features))

        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def densenet121(model_config):
    return DenseNet((6, 12, 24, 16), num_classes=model_config.get("num_classes"))

def densenet161(model_config):
    return DenseNet((6, 12, 36, 24), 96, growth_rate=48, num_classes=model_config.get("num_classes"))

def densenet169(model_config):
    return DenseNet((6, 12, 32, 32), num_classes=model_config.get("num_classes"))

def densenet201(model_config):
    return DenseNet((6, 12, 48, 32), num_classes=model_config.get("num_classes"))
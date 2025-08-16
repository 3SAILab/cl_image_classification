import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=0.5)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.downsample = None
        if stride != 1 or out_channel != in_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1,
                                            stride=stride, bias=False)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.dropout(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out += identity

        return out


class WideResNet(nn.Module):
    def __init__(self, depth, k, num_classes=1000):
        super(WideResNet, self).__init__()
        self.in_channel = 16

        n = (depth-4)/6

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(BasicBlock, 16*k, n)
        self.layer2 = self._make_layer(BasicBlock, 32*k, n, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64*k, n, stride=2)
        self.bn1 = nn.BatchNorm2d(64*k, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64*k, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block, out_channel, block_num, stride=1):
        strides = [stride] + [1]*(int(block_num)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channel, out_channel, stride))
            self.in_channel = out_channel

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn1(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def wideresnet40(model_config):
    return WideResNet(depth=40, k=4, num_classes=model_config.get("num_classes"))



import torch.nn as nn
import torch

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(SeparableConv2d, self).__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x

class MobileNetV1(nn.Module):
    def __init__(self, model_config):
        super(MobileNetV1, self).__init__()
        self.num_classes = model_config.get("num_classes")
        self.alpha = model_config.get("alpha")

        channel_list = [32, 64, 128, 256, 512, 1024]
        channel_list = [int(c*self.alpha) for c in channel_list]
        

        self.conv1 = nn.Conv2d(3, channel_list[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channel_list[0])
        self.conv2 = SeparableConv2d(channel_list[0], channel_list[1])
        self.conv3 = SeparableConv2d(channel_list[1], channel_list[2], stride=2)
        self.conv4 = SeparableConv2d(channel_list[2], channel_list[2])
        self.conv5 = SeparableConv2d(channel_list[2], channel_list[3], stride=2)
        self.conv6 = SeparableConv2d(channel_list[3], channel_list[3])
        self.conv7 = SeparableConv2d(channel_list[3], channel_list[4], stride=2)
        self.conv8 = nn.Sequential(
            SeparableConv2d(channel_list[4], channel_list[4]),
            SeparableConv2d(channel_list[4], channel_list[4]),
            SeparableConv2d(channel_list[4], channel_list[4]),
            SeparableConv2d(channel_list[4], channel_list[4]),
            SeparableConv2d(channel_list[4], channel_list[4])
        )
        self.conv9 = SeparableConv2d(channel_list[4], channel_list[5], stride=2)
        self.conv10 = SeparableConv2d(channel_list[5], channel_list[5], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel_list[5], self.num_classes, bias=False)

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        x = self.relu(self.bn(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
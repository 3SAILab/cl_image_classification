import torch 
import torch.nn as nn
import torch.nn.functional as F

class channel_shuffle(nn.Module):
    def __init__(self, groups):
        super(channel_shuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        assert channels % self.groups == 0
        group_channels = channels // self.groups
        x = x.view(batch_size, self.groups, group_channels, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, channels, height, width)
        return x

class shufflenetunit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super(shufflenetunit, self).__init__()
        self.stride = stride

        if stride == 2:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups),
                nn.BatchNorm2d(in_channels),
            )
            self.main_out_channels = out_channels - in_channels
        else:
            self.shortcut = nn.Sequential()
            self.main_out_channels = out_channels

        mid_channels = max(groups, self.main_out_channels // 4)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, groups=groups),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            channel_shuffle(groups),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, self.main_out_channels, kernel_size=1, groups=groups),
            nn.BatchNorm2d(self.main_out_channels), 
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.bottleneck(x)

        if self.stride == 1:
            out += shortcut
        else:
            out = torch.cat([shortcut, out], dim=1)

        return F.relu(out)

# 仅实现论文中g3版本，说是最平衡的
class ShuffleNetV1(nn.Module):
    def __init__(self, model_config):
        super(ShuffleNetV1, self).__init__()
        self.num_classes = model_config.get("num_classes")
        self.groups = model_config.get("groups")

        out_channels_per_stage = [240, 480, 960] 
        repeat_blocks = [4, 8, 4]

        self.stage_out_channels = [24] + out_channels_per_stage

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        self.stage = nn.Sequential()
        in_channels = 24
        for i, (out_channels, repeats) in enumerate(zip(self.stage_out_channels[1:], repeat_blocks)):
            layers = []
            layers.append(shufflenetunit(in_channels, out_channels, stride=2, groups=self.groups))
            in_channels = out_channels
            for _ in range(repeats - 1):
                layers.append(shufflenetunit(in_channels, out_channels, stride=1, groups=self.groups))
                # in_channels = out_channels
            self.stage.add_module(f'stage_{i+2}', nn.Sequential(*layers))

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.stage_out_channels[-1], self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

import torch.nn as nn
import torch

def Conv2dNormActivation(in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, activation_layer=nn.ReLU):
    if padding is None:
        padding = (kernel_size - 1) // 2
        
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        activation_layer(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        expand_ratio # 第一个1x1卷积层升维的比例
        ):
        super(InvertedResidual, self).__init__()
        self.stride=stride
        
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(
                Conv2dNormActivation(in_channels, hidden_dim, kernel_size=1, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    activation_layer=nn.ReLU6,
                ),
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = out_channels
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, model_config):
        super(MobileNetV2, self).__init__()
        self.num_classes = model_config.get("num_classes")
        self.width_mult = model_config.get("width_mult")
        self.round_nearest = model_config.get("round_nearest")
        input_channel = 32
        last_channel = 1280

        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # 构建第一层卷积层
        input_channel = self._make_divisible(input_channel * self.width_mult, self.round_nearest)
        self.last_channel = self._make_divisible(last_channel * max(1.0, self.width_mult), self.round_nearest)
        features = [
            Conv2dNormActivation(3, input_channel, stride=2, activation_layer=nn.ReLU6)
        ]
        # 构建倒残差块
        for t, c, n, s in inverted_residual_setting:
            output_channel = self._make_divisible(c * self.width_mult, self.round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # 构建最后一层卷积层
        features.append(
            Conv2dNormActivation(
                input_channel, self.last_channel, kernel_size=1, activation_layer=nn.ReLU6
            )
        )
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.last_channel, self.num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_divisible(self, v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v



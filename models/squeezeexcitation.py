import torch.nn as nn

class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        input_channels,
        squeeze_channels,  # 第一个全连接层压缩后的通道数
        activation=nn.ReLU,
        scale_activation=nn.Sigmoid
        ):
        super(SqueezeExcitation, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1) # squeeze操作的全局平均池化
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def forward(self, input):
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return scale * input
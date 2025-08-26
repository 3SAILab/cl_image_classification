import torch.nn as nn
import torch.nn.functional as f
import math

class ECA(nn.Module):
    def __init__(
        self,
        channels=None,          # 输入特征图的通道数
        kernel_size=3,          # 初始卷积核大小
        gamma=2,                # 控制自适应kernel_size的超参数，下同
        beta=1,
        gate_layer=nn.Sigmoid  # 最终门控函数
        ):
        super(ECA, self).__init__()
        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.gate = gate_layer()

    def forward(self, x):
        y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
        y = self.conv(y)
        y = self.gate(y).view(x.shape[0], -1, 1, 1)
        return x * y.expand_as(x)
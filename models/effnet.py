import torch
import torch.nn as nn

class EffBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_rate=4, depth_multiplier=2):
        super(EffBlock, self).__init__()
        
        expanded_channels = in_channels * expansion_rate
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True) 
        )
        
        self.depthwise_1x3 = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels * depth_multiplier, 
                     kernel_size=(1, 3), stride=1, padding=(0, 1), groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels * depth_multiplier),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.pool_1x2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 0))
        
        self.depthwise_3x1 = nn.Sequential(
            nn.Conv2d(expanded_channels * depth_multiplier, expanded_channels * depth_multiplier, 
                     kernel_size=(3, 1), stride=1, padding=(1, 0), groups=expanded_channels * depth_multiplier, bias=False),
            nn.BatchNorm2d(expanded_channels * depth_multiplier),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        
        self.project_and_downsample = nn.Sequential(
            nn.Conv2d(expanded_channels * depth_multiplier, out_channels, 
                     kernel_size=(2, 1), stride=(2, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):

        x = self.expand(x)

        x = self.depthwise_1x3(x)

        x = self.pool_1x2(x)  
        
        x = self.depthwise_3x1(x)
        
        x = self.project_and_downsample(x)
        
        return x

class EFFNet(nn.Module):
    def __init__(self, model_config):
        super(EFFNet, self).__init__()
        self.num_classes = model_config.get("num_classes")
        self.first_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )
        
        self.effnet_block1 = EffBlock(32, 64, expansion_rate=4)  
        self.effnet_block2 = EffBlock(64, 128, expansion_rate=4)  
        self.effnet_block3 = EffBlock(128, 256, expansion_rate=4)  
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(256, self.num_classes)

    def forward(self, x):
        x = self.first_layer(x)
        
        x = self.effnet_block1(x)
        x = self.effnet_block2(x)
        x = self.effnet_block3(x)
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  
        x = self.classifier(x)
        
        return x

        
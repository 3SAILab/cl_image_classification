import torch.nn as nn
import torch
import torch.nn.functional as F

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.data.size(0),-1)

class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.add_module('layer1',ConvLayer(in_channels, out_channels, kernel))
        self.add_module('layer2',DWConvLayer(out_channels, out_channels, stride=stride))
        
    def forward(self, x):
        return super().forward(x)

class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels,  stride=1,  bias=False):
        super().__init__()
        out_ch = out_channels
        
        groups = in_channels
        kernel = 3
        #print(kernel, 'x', kernel, 'x', out_channels, 'x', out_channels, 'DepthWise')
        
        self.add_module('dwconv', nn.Conv2d(groups, groups, kernel_size=3,
                                          stride=stride, padding=1, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(groups))
    def forward(self, x):
        return super().forward(x)  

class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = 1
        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
        self.add_module('conv', nn.Conv2d(in_channels, out_ch, kernel_size=kernel,          
                                          stride=stride, padding=kernel//2, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_ch))
        self.add_module('relu', nn.ReLU6(True))                                          
    def forward(self, x):
        return super().forward(x)


class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
          return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
          dv = 2 ** i
          if layer % dv == 0:
            k = layer - dv
            link.append(k)
            if i > 0:
                out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
          ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
          in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0 # if upsample else in_channels
        for i in range(n_layers):
          outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
          self.links.append(link)
          use_relu = residual_out
          if dwconv:
            layers_.append(CombConvLayer(inch, outch))
          else:
            layers_.append(ConvLayer(inch, outch))
          
          if (i % 2 == 0) or (i == n_layers - 1):
            self.out_channels += outch
        #print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)
        
    def forward(self, x):
        layers_ = [x]
        
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:            
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
            
        t = len(layers_)
        out_ = []
        for i in range(t):
          if (i == 0 and self.keepBase) or \
             (i == t-1) or (i%2 == 1):
              out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out
        
class HarDNet(nn.Module):
    def __init__(self, params, num_classes=1000, depth_wise=False):
        super().__init__()
        self.params = params
        self.depth_wise = depth_wise
        self.num_classes = num_classes
        
        # 构建网络组件
        self.features = self._build_features()
        self.classifier = self._build_classifier()
        
    def _build_features(self):
        """构建特征提取部分"""
        features = nn.ModuleList()
        params = self.params
        
        # 第一层：标准Conv3x3，Stride=2
        features.append(ConvLayer(
            in_channels=3, 
            out_channels=params['first_ch'][0], 
            kernel=3,
            stride=2,  
            bias=False
        ))
  
        # 第二层
        features.append(ConvLayer(
            in_channels=params['first_ch'][0], 
            out_channels=params['first_ch'][1], 
            kernel=params['second_kernel']
        ))
        
        # 下采样层
        if params['max_pool']:
            features.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            features.append(DWConvLayer(
                in_channels=params['first_ch'][1], 
                out_channels=params['first_ch'][1], 
                stride=2
            ))

        # 构建所有HarDNet blocks
        ch = params['first_ch'][1]
        for i in range(len(params['n_layers'])):
            # 添加HarDBlock
            blk = HarDBlock(
                ch, 
                params['gr'][i], 
                params['grmul'], 
                params['n_layers'][i], 
                dwconv=self.depth_wise
            )
            ch = blk.get_out_ch()
            features.append(blk)
            
            # 添加1x1卷积调整通道数
            features.append(ConvLayer(ch, params['ch_list'][i], kernel=1))
            ch = params['ch_list'][i]
            
            # 添加下采样
            if params['downSamp'][i] == 1:
                if params['max_pool']:
                    features.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    features.append(DWConvLayer(ch, ch, stride=2))
        
        return features
    
    def _build_classifier(self):
        """构建分类器部分"""
        ch = self.params['ch_list'][-1]
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Dropout(self.params['drop_rate']),
            nn.Linear(ch, self.num_classes)
        )
        
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.classifier(x)
        return x

def hardnet68(model_config):
    params = {
        'first_ch': [32, 64],
            'second_kernel': 3,
            'max_pool': True,
            'grmul': 1.7,
            'drop_rate': 0.1,
            'ch_list': [128, 256, 320, 640, 1024],
            'gr': [14, 16, 20, 40, 160],
            'n_layers': [8, 16, 16, 16, 4],
            'downSamp': [1, 0, 1, 1, 0]
    }
    return HarDNet(params, model_config.get("num_classes"))

def hardnet85(model_config):
    params = {
        'first_ch': [48, 96],
            'second_kernel': 3,
            'max_pool': True,
            'grmul': 1.7,
            'drop_rate': 0.2,
            'ch_list': [192, 256, 320, 480, 720, 1280],
            'gr': [24, 24, 28, 36, 48, 256],
            'n_layers': [8, 16, 16, 16, 16, 4],
            'downSamp': [1, 0, 1, 0, 1, 0]
    }
    return HarDNet(params, model_config.get("num_classes"))

def hardnet68ds(model_config):
    params = {
         'first_ch': [32, 64],
            'second_kernel': 1,
            'max_pool': False,
            'grmul': 1.7,
            'drop_rate': 0.05,
            'ch_list': [128, 256, 320, 640, 1024],
            'gr': [14, 16, 20, 40, 160],
            'n_layers': [8, 16, 16, 16, 4],
            'downSamp': [1, 0, 1, 1, 0]
    }
    return HarDNet(params, model_config.get("num_classes"), True)
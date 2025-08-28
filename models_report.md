# AlexNet
### 一、网络架构
![Alt](https://i-blog.csdnimg.cn/direct/378bf6d10c5944648c802813ada462f0.png)  
8个学习层：5个卷积层+3个全连接层（部分卷积层后有最大池化）

### 二、重点
#### 1.ReLU做激活函数 $\color{green}{\checkmark}$
- 传统tanh和sigmoid函数容易饱和，训练慢。而ReLU是max(0,x) (输入正输出x,负输出0)。
- ReLU输入中的负值指该特征与所学特征负相关。
- 主要好处是：  
①防止梯度消失，因为在x>0的时候梯度恒为1，深层训练更稳定  
②计算快，只用判断输入是否为正就好  
③降低过拟合风险，负值都输出0，相当于失活部分神经元，使模型不用学习许多不必要的特征
#### 2.多GPU训练
- 把网络拆到两个GPU上以进行大模型训练。  
- ps1: pytorch官方未实现，技术过于复杂通常不用。  
#### 3.LRN局部响应归一化
- 每个神经元的输出会被周围神经元的输出归一化，最终强的突出，弱的抑制。目的是防止过拟合，增强模型泛化能力。  
公式：  
$$
b_{x,y}^i = \frac{a_{x,y}^i}{\left(k+\alpha\sum_{j=\max(0,i-\frac{n}{2})}^{\min(N-1,i+\frac{n}{2})}(a_{x,y}^j)^2\right)^\beta}
$$
- ps:作用有限，基本不用，pytorch官方未实现  
#### 4.重叠池化 $\color{green}{\checkmark}$
- 用kernel_size=3,stride=2的池化层，使相邻池化区域重叠（因为kernel_size>stride），减少过拟合。  
#### 5.Dropout防止过拟合 $\color{green}{\checkmark}$
- 全连接层训练时，使部分神经元随机失活，文章以及pytorch官方都是p=0.5。
### 三、注意点
#### 1.初始化权重
- 文章中使用均值=0，方差=0.01的高斯随机变量，bias在2，4，5卷积层=1，其余=0。

- pytorch官方使用kaiming_uniform_(均匀分布初始化)+bais均匀初始化（ps:最新版AlexNet中只有导入预训练权重的代码，随机初始化权重在nn.Conv和nn.Linear中，目录是~\anaconda3\envs\env_name\Lib\site-packages\torch\nn\modules）。

- 学习视频中是自定义随机初始化权重，即kaiming_normol_（正态分布初始化）+bais=0，会覆盖官方的。  
#### 2.优化器
- 文章中采用SGD+momentum,每一层固定学习率0.01，效果不好再手动调整，按10倍往下降。
- 学习视频中用Adam，对学习率敏感度不高，自适应学习率，不用调很多参数，收敛速度快。
# VGGNet
### 一、网络架构及配置
架构：
![Alt](https://i-blog.csdnimg.cn/direct/4611d9f33f53459aa1c49d365019589d.png)
输入：224 x 224 的RGB图像，减去平均RGB值  
卷积层：仅由 3 x 3 和 1 x 1 构成  
最大池化层：5个  
全连接层：3个  

配置：
![Alt](https://i-blog.csdnimg.cn/direct/44fa5e74cef34c34834635fd27ae73eb.png)

### 二、重点
#### 1.感受野
- 概念：一个神经元能看到的输入区域。  
公式：  
$$
F(i)=(F(i+1)-1)*Stride+Ksize
$$
$F(i)$为第$i$层感受野，$Stride$为第$i$层步距，$Ksize$为卷积核大小或池化核大小。 
![Alt](https://i-blog.csdnimg.cn/direct/cc38347d64e94149bfc4b05e0b113eaf.png) 
感受野计算是从最后一层往前推，看能在原图上“看”到多大范围。  
Feature map: F=1   
Conv3x3(3): F=(1-1)x1+3=3  
Conv3x3(2): F=(3-1)x1+3=5  
Conv3x3(1): F=(5-1)x1+3=7  
Conv7x7(1): F=(1-1)x1+7=7  
由此可知用3x3卷积核叠3层后与7x7卷积核感受野相同。
#### 2.小卷积核代替大卷积核好处
- ①同样感受野的情况下参数减少，计算更高效。  
②小卷积需要增加深度来达到相应感受野，而深度增加便于学习到更复杂的特征。  
③由于叠了多个卷积层，ReLU更多，而ReLU是有助于提高模型的判别能力的。  
### 三、注意点
#### 1.初始化权重
- 文章中首先使用较浅层的A网络随机初始化权重进行训练，然后用它的前四个卷积层和后三个全连接层的权重初始化后面的网络，中间则仍用随机。  
但文章发布后他们使用了Xavier初始化，相关论文：<https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>  
它的核心是平衡各层方差，缓解梯度消失或爆炸。  
- pytorch官方的卷积层用了kaiming_normal_初始化权重，其余都设为常量，这个随机初始化权重的函数直接写在vgg类中。
#### 2.测试时全连接层替换为卷积层
- 全连接层需要固定输入大小，因此需要批量剪裁图片，替换为卷积层后省去这一操作，提高测试速度。同时卷积操作也可以提取更多特征信息。  
#### 3.网络实现 
- 文章中一共有5种网络结构A-E，pytorch官方中未实现C，同时加入了带BatchNorm的版本，BN对每一层的输入进行归一化，相关论文：<https://arxiv.org/pdf/1502.03167>
# GoogLeNet/Inception_v1  
### 一、网络架构
![Alt](https://i-blog.csdnimg.cn/blog_migrate/6ed5a4d39d44ea256f0a34c4e555a814.png)  
### 二、重点  
#### 1.Inception模块
- 组合不同尺寸的卷积提取多种特征，最后拼接，能够提取到更多尺度的特征。
- 其中大尺寸卷积过多可能导致计算量爆炸，因此引入1x1卷积降维，在大卷积之前较少输入通道数。  
#### 2.平均池化代替全连接层
- 文章中说引用这篇论文的结论：<https://arxiv.org/pdf/1312.4400>  
在这里应该主要是为了减少过拟合风险，全连接层需要大量训练参数，而平均池化层不用。
#### 3.辅助分类器
- 由于网络深度较大，可能导致梯度消失，无法高效的进行反向梯度传播，因此在中间加上两个辅助分类器，使浅层的网络也能接受较强信号来训练。
- 提供额外的正则化，防止过拟合。
### 三、注意点
#### 1.优化器
- 文章中用0.9动量的异步随机梯度下降（将任务分配到不同节点，个节点独立工作），每8个epoch降低4%，数据集不大时还是用Adam。
# Inception_v2
### 一、改进点
#### 1.提出Batch Normalization（批标准化）方法
- 在深层网络训练时，由于学习到的权重等参数改变，每一层的输入分布都会改变，因此每一层都要重新适应新的分布，这不仅降低了效率，还不稳定。
- BN对每一层的输入数据做标准化处理，使均值为0，方差为1，让数据分布更稳定，同时加入两个可学习参数，让网络能灵活调整这个分布。
- 主要好处是：  
①训练速度加快。  
②调参变简单，对权重初始化的要求降低，由于公式内部已有偏移量，因此不需要设置bias。  
③起到一定程度上正则化作用，减少过拟合，可以去掉dropout。  
④能运用饱和激活函数如sigmoid且不会引起梯度消失。
#### 2.5x5卷积替换为两个3x3卷积
- 减少参数  
# ResNet
### 一、网络架构  
![Alt](https://i-blog.csdnimg.cn/blog_migrate/75a8be53f53b21a4b5fa433fdf8e9cd6.png)  
从左到右分别是VGG-19,普通34层网络，34层残差网络（虚线是当维度不匹配的时候，用另一种映射方式）  
![Alt](https://i-blog.csdnimg.cn/direct/c60d5f0a4e964994b524d9008b5f57ce.png)  
左边是18和34层用的残差结构，右边是50，101，152层的  
配置：
![Alt](https://i-blog.csdnimg.cn/blog_migrate/fc311cfc3719e005c75aa728f8913e3e.png)
### 二、重点
#### 1.shortcut connection
- 绕过一层或多层变换，直接加到后续层，使其不在深层中丢失。
- 主要有3种方式：恒等、投影（1x1卷积实现）、零填充。后两个可以解决维度不匹配的问题。实验证明恒等+零填充足够解决问题，虽然全部用投影效果略好但好的不多，且参数过多。
#### 2.**深度残差学习**框架解决退化问题
- 核心思想是由传统的直接映射变为残差映射，原映射变为F(x)=H(x)+x。  
普通网络让每一层学H(x)，残差网络让它学与输入x的差值，就是在原有基础上优化，最差就是不进步，这样训练也能更快。
#### 3.两种残差块设计
- 基础块（BasicBlock）和瓶颈块（Bottleneck），前者用于18、34层，后者用于50、101、152层，主要是在深层训练时参数过多，用1x1卷积降维然后再升维可以较少计算量。
#### 4.使用BN（批归一化）
- 加速训练，简化调参，一定程度正则化效果。
### 三、注意点  
#### 1.Bottlenect差异
- 文章中1x1卷积stride=2,3x3卷积stride=1，pytorch官方用1x1卷积stride=1，3x3卷积stride=2，这样能在top-1上提升0.5%的准确率，官方文档：<https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch>
# Inception_v3
### 一、改进点
#### 1.分解卷积
- 大卷积分解为小卷积，例如5x5卷积分为两个3x3的。  
- 不对称卷积分解，例如7x7变为1x7和7x1。  
更加节省计算成本，使用特征图尺寸为12x12到20x20，早期层表现较差。
#### 2.引入LSR（标签平滑正则化）
- 一种正则化手段，不让模型过度自信。
- 公式：  
$$
y_{\text{smooth}}(k) = 
\begin{cases}
1 - \epsilon & \text{if } k = \text{true label} \\
\frac{\epsilon}{K - 1} & \text{otherwise}
\end{cases}
$$

其中：
$ \epsilon $：平滑系数（如 $ 0.1 $）
$ K $：总类别数
- 例如one-hot编码后[0,0,1,0],使用LSR后变成[0.025,0.025,0.9,0.025],可以较少过拟合，提升泛化能力。
#### 3.多种inception模块
- 根据不同层级的特征图优化
#### 4.优化辅助分类器
### 二、补充
#### 1.pytorch官方将inception_v2和v3结合实现。
#### 2.googlenet及inception3对比
- 用pytorch官方的代码，修改部分参数以适配224x224输入图像，固定seed=42训练，最终inception3准确率提升3.5%，训练时间延长，参数量翻倍（判断是网络更深，结构更复杂的缘故）
#### 3.有无LSR对比
- 用googlenet模型，固定seed=42，有LSR比没有准确率提升1.6%。
# Inception_v4
### 一、改进点
#### 1.实现更简洁的Inception_v4架构（无残差）
#### 2.inception架构与残差连接结合
- 用更轻量的inception块，后接1x1卷积扩展维度，用来匹配输入维度与残差相加。
- 设计了v1,v2两个版本，前者成本与v3相当，后者与v4相当。两者主要是模块结构的不同，经过实验v2表现更好。
#### 3.残差缩放
- 当网络中卷积核的数量超过1000时，容易出现训练不稳定，在最后全局池化层之前出现很多0，且无法通过降低学习率及增加额外批归一化来解决。
- 在将残差添加到前一层激活值之前，对残差进行缩放（乘一个常数，一般在0.1-0.3），可以有效稳定训练过程。主要是通过调整残差信号的强度，防止它过大或者过小来稳定深层训练。
# SqueezeNet
### 一、网络架构
![Alt](https://i-blog.csdnimg.cn/blog_migrate/87b844cb635931a262be3f63316fe3c6.png)
从左到右是完善过程，中间加了shortcut，右边加了匹配不同维度的shortcut。
### 二、重点
#### 1.压缩策略
- 大多数网络都在往深而精改进，但现实生活中由于计算平台的限制，无法提供足够的算力支持，因此想要构架一种轻量的小型网络，用于：  
①更有效的分布式训练。  
②可以将新模型导出到客户端时开销更小。  
③可行的嵌入式部署。
- 使用3种策略：  
①将3x3卷积替换成1x1卷积。  
②减少3x3卷积的通道数。  
③下采样后置，使卷积层具有较大的activation maps，提升精度，但会增加计算量。
#### 2.fire模块（2部分组成）
- squeeze部分，一组连续的1x1卷积。限制后续3x3卷积输入通道数量。
- expand部分，一组连续的1x1卷积和一组连续的3x3卷积拼接组成。
#### 3.无全连接层
### 三、注意点
#### 1.pytorch官方复现squeezenet有两个版本，1_0第一层卷积是7x7，1_1变为3x3,并且改动部分池化层的位置。
# Wide ResNet
### 一、网络架构
![Alt](https://i-blog.csdnimg.cn/blog_migrate/20de119b10593a7960252ea739a26c13.png)
左边两个是resnet用的残差块，右边两个是本模型用的残差块。
### 二、重点
#### 1.优选残差块
- 不使用bottleneck,因为会加大网络深度。文章中经过实验选用了精度最高的残差块（2个3x3卷积）
#### 2.增加通道数来加宽卷积层
- 主要的好处是：  
①在不用加深网络的情况下学习到更多特征。  
②提高特征重用效率，即前面层提取的特征被后续层有效利用的程度。resnet用shortcut实现每一层只需要学习跟上一层的差值，可能导致很多层没学到新东西，在划水，虽然深度增加，但效率很低。而增加通道数可以更有利于每一层学到新东西。
- 文章引入深度因子和宽度因子，通过实验选出了最优的组合。
#### 3.dropout插入残差块的卷积之间
- 更宽的网络更容易过拟合，因此需要dropout起正则化作用。
- 如果在最后加会太晚，在两个卷积层之间，防止后一个卷积记住前一个的特定模型，以此防止过拟合。
### 三、注意点
#### 1.pytorch官方将wideresnet放在resnet同文件中，使用resnet框架实现，其中两点未复现：  
①bn->relu->conv的顺序（用50层做一次对比实验显示，在自己收集的数据集上，这个顺序比传统的提升3.5%准确率）  
②不使用瓶颈层（文章中研究的都是浅层模型，官方实现深层模型使用了瓶颈层）
# DenseNet
### 一、网络架构
![Alt](https://i-blog.csdnimg.cn/blog_migrate/b626b419aadf20467aace9ce01efb483.png)
![Alt](https://i-blog.csdnimg.cn/blog_migrate/3e5439db041d9cedb31da90388f98061.png)  
配置：  
![Alt](https://i-blog.csdnimg.cn/blog_migrate/7cb083ed2b9779ac387973300edf0d3a.png)
### 二、重点
#### 1.采用密集连接
- 每层与所有前驱层直接连接，每层输入为所有前驱层特征图的拼接，输出传递给所有后继层。
- 主要好处是：    
①大幅缓解梯度消失问题，浅层也有多条路径可以回传，使网络更稳定。  
②优化信息传递，特征不需要逐层传递，而是直接被后续层引用，避免在传递过程中被污染。  
③每一层都参与学习，减少冗余，在resnet中，可能一些层学不到新东西划水，但densenet强迫每一层都要学新东西。
#### 2.采用特征拼接代替求和
- 将特征沿通道维度拼接，保留所有原始特征。
- 相较于resnet的求和不增加通道数，特征拼接将浅层的特征直接拿来给后续层用，防止浅层信息缺失污染；由于是每一层都需要将学到的新东西拼接进去，即使学到的很少，也会在反向传播当中造成影响，不存在划水现象。
- 参数量少，每一层固定学习量，即channel=32，不用像其他网络一样设置很多通道数，同时还用了1x1卷积降维。
#### 3.过渡层
- 每一层输出的尺寸不同，因此将整个网络划分为多个block，每个之间加上过渡层用来匹配输入输出尺寸。
- 引入一个压缩因子，防止通道数增长太过庞大。
### 三、注意点
#### 1.pytorch官方未实现DenseNet264,新加了一个DenseNet161，其中的growth_rate=48。官方默认设置dropout(p=0),也就是不启用，因为bn已经有很强的正则化。在自己数据集上实验显示161效果最好，但都体现出过拟合现象（？）
# Xception
### 一、网络架构
![Alt](https://i-blog.csdnimg.cn/blog_migrate/12cd63949bad5aeae1ca5f10030888a1.png)  
![Alt](https://i-blog.csdnimg.cn/blog_migrate/c8d7028ebe80463fa7b4bf215b6d6a79.png)![Alt](https://i-blog.csdnimg.cn/blog_migrate/0d9d08d8eb8eff0b9f291211b3994868.png)  
左图是inception部分结构，右图是Xception部分结构
### 二、重点
#### 1.将inception定义为普通卷积层与深度可分离卷积层的“中间态”
- 深度卷积：用于提取空间特征，channel不变，H/W改变。  
逐点卷积：用于提取通道特征，H/W不变，channel改变。  
深度可分离卷积：先深度卷积然后再逐点卷积，让参数同时只用关注一个维度的信息。
- inception与普通卷积的差别在于它将通道分为3-4个独立段，用不同卷积核处理通道相关性，虽能减少冗余，但仍存在“多通道共享部分空间参数的问题”，而用深度可分离卷积，一个卷积核只用处理一个通道。
#### 2.全由深度可分离卷积层构建
- 深度可分离卷积层使相关性完全解耦，提高参数利用率，例如3通道图像，可能红绿通道对应树叶特征，与空间无关，但普通卷积会强行关联它们，造成了无关参数增加。
#### 3.深度可分离卷积层中不加ReLU
- 文章中对比了加与不加的效果，结果是不加激活层更好，他们认为是深度卷积是单通道处理，加激活层可能丢失信息。
# ResNeXt
### 一、网络架构
![Alt](https://i-blog.csdnimg.cn/blog_migrate/66b5be5cdb66887ee0160cd99f622f89.png)  
三种等效模块形式
### 二、重点
#### 1.简介模块化的架构
- 将inception与resnet融合的同时，不像inception_v4那样设计过多的模块。
- 提出三种等效的模块形式：  
①聚合残差变换（直接对多个变换结果求和）  
优点：理解简单，变换路径清晰 缺点：需要显示定义多个分支（branch1,branch2...），计算效率不够高。  
②早期拼接（变换前拼接输入，再通过卷积融合）  
优点：拼接+单卷积，一定程度简化分支管理 缺点：拼接操作可能带来额外开销。  
③分组卷积（通过分组卷积实现聚合变换）  
优点：最简洁，计算效率最高（文章中首选）
#### 2.分组卷积
- 文章中提出“基数”，可以理解为通道分开的组数量，相比增加宽度和深度，增加它不会增加参数量。模型主要通过分组卷积实现融合inception,resnet优点的同时高度模块化。
- ps:个人理解，关于深度卷积和分组卷积，前者应该是后者的极端形式。Xception是利用深度可分离卷积进行相关性解耦，这样虽可以减少参数量，但在某些情况下也许跨通道特征和空间特征需要结合起来看，所以再某些数据集上也许resnext的准确度会更高。
# MobileNetV1
### 一、网络架构
![Alt](https://i-blog.csdnimg.cn/blog_migrate/42c65ab4806969a79d87df69d8640deb.png)
### 二、重点
#### 1.深度可分离卷积
- Xception注重于用它提升表达能力，只替换了inception模块，而该模型注重于用它降低参数量和计算量，使模型更加轻量化。  
普通卷积参数：$$D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F$$  
深度可分离卷积参数：$$D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F$$  
两者对比：$\frac{D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F}{D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F}=\frac{1}{N}+\frac{1}{D_K^2}$
#### 2.增加两个参数$\alpha$,$\rho$
- 宽度乘数$\alpha$：每层均匀瘦化一个网络，取(0,1]，通常取1,0.75,0.25,0.5  
分辨率乘数$\rho$: 缩减输入图像及内部特征图分辨率，取(0,1]，通常设置输入分辨率为224，192，160，128  
减少计算量：  
$\frac{D_K \cdot D_K \cdot \alpha M \cdot \rho D_F \cdot \rho D_F + \alpha M \cdot \alpha N \cdot \rho D_F \cdot \rho D_F}{D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F}=\frac{\alpha \rho}{N}+\frac{\alpha^2 \rho^2}{D_K^2}$
# ShuffleNetV1
### 一、网络架构
![Alt](https://i-blog.csdnimg.cn/blog_migrate/82a7e2122390a0de4adaafa3b9565bb0.png)  
最左边是ResNext的block，右边两个是该模型用的block
![Alt](https://i-blog.csdnimg.cn/blog_migrate/59c10ca551d01a9e92d32b45ad741559.png)
### 二、重点
#### 1.逐点分组卷积
- 传统的点卷积虽然能高效地进行通道维度的变换，但计算占比过高。逐点分组卷积将输入通道分组，使每个卷积核只在相对应的输入通道操作，减少计算量。
#### 2.通道重排
- 分组卷积导致了通道信息隔绝，因此把特征图的通道顺序打乱重组，让不同组的信息能够流通，这样的通道操作同时没有计算成本。
# SENet
### 一、网络架构
![Alt](https://i-blog.csdnimg.cn/blog_migrate/c786637932b64bb9f979b89d9c57693c.png)  
嵌入不同网络的SE模块  
![Alt](https://i-blog.csdnimg.cn/blog_migrate/fcef831baace632673370ef1909690d1.png)  
配置
### 二、重点
#### 1.提出SE模块
- 传统卷积会平等对待所有通道，即使该通道提取的特征对当前图像识别无用，SE块可以多关注有用的，抑制无用的，也就是自适应通道校准。这个模块分为两个部分。
- squeeze部分（全局信息嵌入）：由于每个特征通道只包含局部信息，无法得知在全局是否重要，所以先给每个通道一个全局总结值，用全局平均池化实现。
- excitation部分（自适应重校准）：获得全局总结值后需要判断重要性，通过两个全连接层和激活函数实现。  
这里需要满足：  
①能够学习通道间的非线性交互（relu引入非线性）  
②能够强调非互斥关系，确保强调多个通道（sigmoid）  
至于用两个全连接层，一个目的是为了降维再升维减少计算量，一个目的是两个激活函数不能用在一个全连接层后面而它们都是必须的。
### 三、补充
- 自己数据集上实验显示，se模块加入后确实提高了resnet的收敛速度以及泛化能力。  
- pytorch官方使用1x1卷积层代替了全连接层，可能是底层做了优化用起来效果比较好。
# MobileNetV2
### 一、改进点
#### 1.倒残差结构
- 关于V1遇到的问题：dw卷积有很多卷积核废掉，文章中认为是relu激活函数对低维特征信息造成大量损失导致，解决方案①升维②最后不做relu激活。
- 残差结构是先降维再升维，倒残差结构先升维再降维。相比v1不用残差结构，它能提升精度，相比普通残差结构，它节省了内存，因为两头小，中间虽大但可以拆着算。  
![Alt](https://i-blog.csdnimg.cn/blog_migrate/026b08442e6a8ab8e955fb2a157968cf.png)
#### 2.Bottleneck最后用Linear
- bottleneck中最后pw卷积的relu都换成线性变换，防止丢失过多信息。
#### 3.使用ReLU6激活函数
- 为了适配移动端的低精度储存，避免输出值过大而导致失真。
- ReLU6将输出值截断在0-6之间，具体如下图：  
![Alt](https://i-blog.csdnimg.cn/blog_migrate/dab14b83606b1a58a8fc75789939dd9e.png)
# EFFNet
### 一、网络架构
![Alt](https://i-blog.csdnimg.cn/blog_migrate/3958fb77a243cb46a8280592ffc85ec6.png)
![Alt](https://i-blog.csdnimg.cn/blog_migrate/0358d996fa7fcff00ee3e8fda0184487.png)
### 二、重点
#### 1.创新的卷积块设计
- 分离式卷积和池化。将深度可分离卷积分为1x3和3x1（inception_v3也用过），中间加上最大池化。拆分卷积可以减少计算量，增加最大池化有利于信息的保留（不像mobilenet直接一步到位）。
- 压缩通道的瓶颈因子=2。不像mobilenet=8,shufflenet=4。
- 去掉残差连接和分组卷积。这两个都是为了缓解在深层网络上的问题，但移动端的网络一般很浅，这样反而会丢失信息，降低准确率。
#### 2.优化首层
- 首层也用了创新的卷积块，mobilenet和shufflenet认为它占比少不优化，但在小模型上它占比不少，优化能较少计算量。
#### 3.使用Leaky ReLU  
- 解决负输入零梯度的问题，扩大函数范围  
公式：  
$$
\text{LeakyReLU}(x) = \begin{cases}
x, & \text{if } x \geq 0 \\
\alpha x, & \text{if } x < 0
\end{cases}
$$  
![Alt](https://i-blog.csdnimg.cn/blog_migrate/a9983094bb6ff066bcdb281da3839be3.png)
### 三、补充
#### 1.目前实验出来参数最少的模型，之前写错在block用了扩展（x4)而非缩减(//2)结果准确率高达0.65，且参数量依然比shufflenet第一位数（或许是LeakyReLU的关系？）
# ShuffleNetV2
### 一、改进点
#### 1.基于shufflev1的不足提出四个准则：
- G1等通道宽度最小化MAC（内存访问成本）->v1用了输入输出通道不同的瓶颈层
- G2过度组卷积增加MAC->v1大量使用了1x1卷积（逐点分组卷积）
- G3网络碎片化降低并行度->分太多组
- G4逐元素操作不可忽视->通道相等时的shortcut用add运算，使用太多
#### 2.基于不足给出的改进
- 基本单元，增加**通道分割**操作，分两个部分；  
1x1卷积不分组；  
不用add用concat；  
shuffle操作换位置。
- 用于空间下采样单元，不增加**通道分割**操作；  
左边结构改变，最后特征图空间大小减半，通道数翻倍。 
![Alt](https://i-blog.csdnimg.cn/blog_migrate/d0d41f274fdf3ac46a6e35223af00676.png)
### 二、补充
#### 1.ShuffleNetV2并未用倒残差结构，pytorch官方虽然给block命名为InvertedResidual,但实现的还是跟原文中一样的结构。
# CBAM
### 一、网络架构
![Alt](https://i-blog.csdnimg.cn/blog_migrate/3a7b95bb0632d5ce906bf8598d958f77.jpeg)  
注意力机制的两个模块，CBAM是一个可插入任意网络的模块。
### 二、重点
#### 1.注意力模块  
- 先用通道注意力：  
做平均池化和最大池化，将空间维度变成1x1，最后做sigmoid激活获得权重（1个值）乘回原特征图。
- 后用空间注意力：  
在通道方向上做平均池化和最大池化（也就是算空间上某个点在所有通道中的平均值和最大值），得到两个1channel的特征图，拼接两个特征图，最后用一个7x7卷积变为1channel的权重特征图，然后乘回原特征图的每一个通道。
# EfficientNetV1
### 一、网络架构
![Alt](https://i-blog.csdnimg.cn/blog_migrate/fedf834b7f9930367261a8a989b8d611.png)  
MBConv模块
![Alt](https://i-blog.csdnimg.cn/blog_migrate/96a8f3c53becdd4b99c68357cbe1d745.png)  
配置
### 二、重点
#### 1.复合缩放
- 传统的网络缩放宽度、深度以及分辨率时都是单一调节，且需要人工调节，麻烦而且精度不高。
- 文章利用复合系数$\phi$来均匀宽度$\alpha$、深度$\beta$和分辨率$\gamma$，它们的初始值是通过NAS（神经网络架构搜索）确定的。
#### 2.基于MobileNetV2的基础架构（EfficientNet-B0）
- 使用MBConv模块，包含倒残差结构、深度可分离卷积、引入SE模块
- 这是最基础的模型，后面延伸出一系列B1-B7。
### 三、注意点
#### 1.激活函数用的是swish
# MobileNetV3
### 一、改进点
#### 1.引入h-swish和h-sigmoid
- h-swish公式：  
$$
\text{h-swish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}
$$  
从公式可以看出h-swish计算简单，跟ReLU一样能缓解梯度消失的问题，且扩大了范围，非单调性也提升了表达能力（能够拟合更复杂的曲线）
![Alt](https://i-blog.csdnimg.cn/blog_migrate/ceeaaa427996ceaedec53a7fa19afd30.png)
- h-sigmoid公式：  
$$
\text{h-sigmoid}(x) = \begin{cases}
0 & \text{if } x \leq -2.5 \\
0.2x + 0.5 & \text{if } -2.5 < x < 2.5 \\
1 & \text{if } x \geq 2.5
\end{cases}
$$  
近似sigmoid但计算简单。
#### 2.加入SE模块（注意力机制）
- 在尽量不增加开销的情况下，提升模型对关键特征的关注度，从而提升模型性能。
#### 3.重新设计耗时层结构
- 减少第一个卷积层的卷积核个数
- 精简Last stage  
![Alt](https://i-blog.csdnimg.cn/blog_migrate/98473d61a422599e03c7597aadaa6fe3.jpeg)
### 二、补充
#### 1.和EfficientNet主要区别
- block中少了dropout
- 用的激活函数是h-swish和h-sigmoid
- 第一个和最后一个卷积层不一样
# HarDNet
### 一、网络架构
![Alt](https://i-blog.csdnimg.cn/blog_migrate/98a7810f30f5d364e69c0be43f5f81f4.png)
### 二、重点
#### 1.稀疏化连接模式
- 传统的DenseNet每一层都与前面所有层连接，导致后期输入暴涨，中间特征图的内存流量大量限制推理时间。
- 该模型的连接模式是让k层连接到$k-2^n$层（只考虑能整除$2^n$层的密集连接）,例如第4层只连接0，2，3层。大大减少连接数量。
#### 2.通道平均加权
- DenseNet还有一个问题是输入通道数过多但真正学习的太少，导致搬运过多数据却只进行了少量的计算。
- 为了平衡输入/输出通道比例，设计了一个公式  
$C(l)=k*m^n$  
其中l是本层，k是基础增长率，m是一个大于1的常数，n由l决定。  
简单理解就是输入通道数多的层输出通道数也多，但它们之间是平衡的。
#### 3.倒置过度模块和瓶颈层优化
- DenseNet中采用“Conv1x1+2x2平均池化”做过渡模块，该模型调整为先分别平均池化和最大池化，拼接之后用一个Conv1x1完成通道压缩。为了不让卷积层的计算量过高。
- 每四个3x3卷积才才插入一个瓶颈层。
# ECA-Net
### 一、网络架构
![Alt](https://i-blog.csdnimg.cn/blog_migrate/a5400edd34917920b5f0d04d67a769b1.png)  
与SE模块对比图，ECA同样是一个即插即用的模块。
### 二、重点
#### 1.避免维度降低
- SE模块用了两个全连接层做瓶颈层，这破坏了通道与其权重间的直接对应关系，使性能下降。
- ECA直接去掉全连接层，直接在全局平均池化后的特征上操作，保证信息直接传递。
#### 2.局部跨通道交互
- 每个通道只与其相邻的几个通道进行交互（文章认为全局不是必须的，这样就能学到有效的注意力），降低了计算复杂度。
- 用1d卷积（即一维卷积）实现，它的卷积核大小为k，决定交互范围，这样之引入了k个参数。
#### 3.自适应卷积核大小
- 提出一个计算k的公式，避免繁琐的手动调参。
- 先定义C与k的非线性关系：  
$$
C = 2^{(\gamma \cdot k - b)}
$$  
其中$\gamma=2$,$b=1$为实验确定的固定参数。  
接着提出公式：  
$$ 
k = \psi(C) = \left| \frac{\log_2(C)}{\gamma} + \frac{b}{\gamma} \right|_{odd} 
$$  
odd表示取它的最近奇数。
# GhostNetV1
### 一、网络架构
![Alt](https://i-blog.csdnimg.cn/direct/23a8108c569a472186f665b3f72642ea.png)  
Ghost module  
![Alt](https://i-blog.csdnimg.cn/direct/f5064ee876e247658f5bc20b27267047.png)  
Ghost bottleneck  
![Alt](https://i-blog.csdnimg.cn/direct/88bc7aa9dc3b420fa632136384e97bb3.png)  
配置
### 二、重点
#### 1.Ghost模块（廉价操作生成冗余特征）
- 传统的CNN中间特征图存在大量冗余特征，即重复或近似信息多，这些可以通过廉价操作近似生成，以此提出了Ghost模块。
- 主要分两步：  
①少量标准卷积生成m个“本源”特征图  
②对每个特征图用s个线性操作（文中用深度卷积，最后一个是恒等映射）  
最终形成m·s个特征图。
#### 2.主要架构
- Ghost瓶颈块堆叠两个Ghost模块，stride=1/2做不同处理，如图。
- 主体参考MobielNetV3,其中瓶颈块替换成Ghost瓶颈。
# ResNeSt
### 一、网络架构
![Alt](https://i-blog.csdnimg.cn/direct/affb47c912a54c6e9ff2a735fa7ef099.png)
### 二、重点
#### 1.split-attention-block(分裂注意力模块)
- ResNeXt分组卷积，缺失跨通道相关性；SE通道注意力机制，忽略空间特征；SK让网络自适应不同感受野的卷积核，但多尺度特征融合不够灵活，推理慢。因此结合这几个的优点来改进。
- 步骤：  
①split分裂：输入特征分成多个分支，进行独立卷积。  
②transform变换：每个分支输出特征。  
③aggregate聚合：将所有分支的特征按元素相加，得到聚合特征。  
④attention注意力：对聚合后的特征进行全局平均池化->全连接层->生成每个分支注意力权重。  
⑤distribute分发：注意力权重（一个）分配回每个分支原始输出加权。  
⑥end:各分支特征相加融合输出。
# Transformer（简单版理解）
### 一、网络架构
![Alt](https://i-blog.csdnimg.cn/blog_migrate/50b27b7b0dfa4e53d66acd930cf0c103.png)  
左编码器，右解码器（多了掩码）
### 二、重点
#### 1.自注意力机制(self-Attention)
- 目的：让序列中的每个位置的词都能直接关注其他所有词的位置，计算它们的相关性。
- 优点：RNN串行化处理，太长了就会忘记前面的；它可以并行化，长距离依赖建模强。
- 公式：  
$$
Attention(Q, K, V)=Softmax(\frac{QK^T}{\sqrt{d_k}})V
$$  
Q:当前词的查询向量（关注与其他词的关系）  
K：其他词的键向量（特征标识）  
V：其他词的值向量（真实值）  
公式的简单理解：当前词计算与其他词的相关性，归一化生成权重，最后乘上真实值，得到包含与其他词关系的输出。  
公式的细节理解：假设输入3个token，每个维度是4；生成wq,wk,wv矩阵，大小为4xdk，这个dk在单头通常=输入维度；与输入相乘得到QKV，3xdk；QK计算并经过softmax得到3x3矩阵；和V相乘得到3xdk矩阵，即最终的输出。  
<img src="https://i-blog.csdnimg.cn/blog_migrate/6d211bf03f22999ab2da2165b38d3c24.png" width="200" height="200" />
#### 2.多头注意力机制（Multi-Head Attention)
- 多个自注意力并行，学习不同的注意力模式。
- 此时dk根据想要的self-Attention数量决定，然后在维度上拼接,最后进行一次线性变换。  
<img src="https://i-blog.csdnimg.cn/blog_migrate/8f9ea8ba2c55dd2784675a19130a7da9.png" width="200" height="200" />

#### 3.位置编码（PE）  
- 目的：为了利用句子中单词的位置信息，它不像RNN是串行处理的，因此需要加入额外的位置编码来表示。
- 最终输出x=单词embedding+位置embedding，注意是add而非concat。
- 公式：  
$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
$$  
pos:词在序列中的位置  
i:维度索引（对应位置embedding向量的第i个分量）  
$d_{model}$：模型的嵌入维度  
对于一个位置编码的向量：偶数维度用sin，奇数维度用cos  
公式的理解：  
①为什么用周期函数？每一个token要唯一性表示，如果用1，2，3...句子长度不确定，遇到更长的直接爆炸；如果像01编码那样设置内部向量的值，不同位置的token不是连续的，对后续位置利用十分不友好；因此用了有界又连续的周期函数。  
②为什么周期函数内部这么计算？要让i小的时候震荡快，区分相邻位置，i大的时候震荡慢，区分远距离位置，因此用$\frac{1}{a^{\frac{2i}{d_{\text{model}}}}}$;至于用10000，是为了有更大的变化范围来表示不同的token位置信息，当token值很大的时候有利于区分不同位置。  
③为什么用sin和cos？使不同位置的向量可以通过线性转换得到。
#### 4.前馈神经网络（FFN）
- 一个两层的全连接层，第一层后有激活函数，第二层没有，这两层在多头注意力后。
- 注意力机制主要是线性变换，FFN引入非线性函数能够帮助拟合更复杂的特征。
#### 5.掩码（Mask）
#### 6.编码器-解码器（Encoder-Decoder）
# ViT
### 0.前期工作
#### 1.做了什么，为什么做？
- 目前CV领域CNN仍占主导，而transformer架构性能强大，CV领域并未实现纯transformer架构，因此希望构建一个纯transformer架构处理序列图像的模型。
#### 2.亮点
- Patch embedding
- transformer Encoder
- MLP Head（最终用于分类的层结构）
#### 3.搞懂
- embedding层什么作用，如何实现？  
目的：将图像转换成适合Transformer处理的向量序列。  
NLP句子如何处理：分词为基本单元token->查表变向量->加入CLS->加入位置编码  
CV图像如何处理：切分成n个patch->线性变换变向量->加入CLS->加入位置编码
- 为什么需要位置编码，如何实现？
- [CLS] token为什么需要设计它，如何实现？  
transformer输入是序列，输出也是
- transformer Encoder架构是什么，输出是什么
- 最终用于分类的层结构是什么？
# 数据增强方法
### 1.mixup
- 对两个样本-标签数据对按比例相加后形成新的样本-标签数据对。
- 公式：  
$$
\begin{aligned}
\tilde{x} &= \lambda x_i + (1 - \lambda) x_j \\
\tilde{y} &= \lambda y_i + (1 - \lambda) y_j \\
\lambda &\sim \text{Beta}(\alpha, \alpha), \quad \alpha > 0
\end{aligned}
$$
其中x是输入向量，y是标签的one-hot编码，$\lambda$符合参数为$\alpha$的${Beta}$分布，$\alpha$越小，越接近原来的某一张图像，越大混合越均匀。
- 在自己数据集上实验显示，它加强了模型的泛化能力，一些有过拟合现象的模型有了极大的改善，但准确率小幅度下降。
### 2.cutmix
- 把一张图片上的某个随机区域剪裁到另一张图片上生成新图片。标签处理与mixup一样，按新样本中两个原样本的比例确定新标签的混合比例。
### 3.manifold mixup
- 简单来说就是混合隐层的输出，mixup混合输入图像，只对网络后续层有影响，而它混合隐层输出，在反向传播的时候可以对隐层之前的层也有影响。
### 4.patchup
- 结合cutmix和manifold mixup，对中间隐层输出进行剪裁，有两种方式：  
①互换法（识别精度好）  
②插值法（对抗攻击的鲁棒性好）
### 5.puzzlemix
- 在cutmix基础上优化，由于合成图片时可能把不重要区域剪下或者把重要区域遮住，因此在剪裁之前先加入显著性分析。
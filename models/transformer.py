import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable

# 自注意力机制
def attention(query, key, value, mask=None, dropout=None):
    # query.shape=key.shape=valie.shape=[batch_size, h, seq_len, d_k]
    d_k = query.size(-1)
    # 计算注意力分数 scores.shape=[batch_size, h, seq_len_q, seq_len_k]
    # transpose(-2, -1)表示交换key的倒数第一和倒数第二个维度，即实现转置
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 用爱因斯坦表示法 scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / math.sqrt(d_k)
    if mask is not None:
        # mask.shape=[batch_size, h, seq_len, seq_len],这里的是padding mask，将填充的0值设为负无穷
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 注意力权重与V加权求和 [batch_size, h, seq_len_q, d_k]
    # !!!注意此处seq_len的值是q的，推理时decoder掩码多头注意力输入的q与k,v序列长度不一样！
    return torch.matmul(p_attn, value), p_attn
    # 用爱因斯坦表示法 return torch.einsum('bhqk,bhvd->bhqd', p_attn, value), p_attn

# 多头注意力机制
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # mask.shape changed [batch_size, 1(new), seq_len, seq_len]
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 构造QKV [batch_size, h, seq_len, d_k]
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        # 自注意力计算
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 多头拼接 [batch_size, seq_len, d_model(h*d_k)]
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 简洁写法 x = torch.einsum('bhld->blhd', x).reshape(nbatches, -1, self.h * self.d_k)
        # 线性投影
        return self.linears[-1](x)

# 前馈神经网络
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 线性层->relu->drop->线性层
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# 词嵌入
class Embeddings(nn.Module):
    # vocab 输入词表大小
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # pytorch内置模块，可以返回指定索引的词向量
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # 乘d_model的开方是为了平衡嵌入向量的方差
        return self.lut(x) * math.sqrt(self.d_model)

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

# 堆叠模块
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 编码器
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# 层归一化
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 可训练的缩放参数，shape[features]，与输入向量最后一维匹配
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# 残差+归一化包装
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # 1.自注意力层，mask屏蔽填充词
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 2.前馈网络层
        return self.sublayer[1](x, self.feed_forward)

# 解码器
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        # 编码器的输出
        m = memory
        # 1.掩码自注意力层,tgt_mask屏蔽未来词
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 2.自注意力层，Q是上一层输出，KV是编码器输出
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 3.前馈网络层
        return self.sublayer[2](x, self.feed_forward)

# 生成squence mask
def subsequent_mask(size):
    attn_shape = (1, size, size)
    # 生成上三角矩阵（值为1），k=1表示对角线以上的元素为1
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 将矩阵转换为布尔型：值为1的位置（未来位置）→ False（屏蔽），值为0的位置→ True（允许关注）
    return torch.from_numpy(subsequent_mask) == 0

# 将解码器的输出转为概率分布
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

# 编码器-解码器整体框架
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1)
        tgt_pad_mask = (tgt != 0).unsqueeze(1)
        tgt_sub_mask = subsequent_mask(tgt.size(1)).type_as(tgt_pad_mask)
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

# 构建模型
def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    # 避免组件共享参数
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        # 原序列：词嵌入+位置编码
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        # 目标序列： 词嵌入+位置编码
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        # 输出词汇概率
        Generator(d_model, tgt_vocab))
    
    for p in model.parameters():
        if p.dim() > 1:
            # Xavier初始化，让输入输出的方差一致
            nn.init.xavier_uniform(p)
    return model   
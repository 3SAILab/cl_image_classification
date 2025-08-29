import math, torch, copy

from torch import nn
import torch.nn.functional as F

from labml import tracker
import numpy as np

# 多头注意力模块的准备
class PrepareForMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model,  # 输入token维度
        heads,    # 头数
        d_k,      # 每个头部的维度
        bias):
        super().__init__()

        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k
    
    def forward(self, x):
        # 输入有[seq_len, batch_size, d_model] or [batch_size, d_model], 这里是保留除最后一维的x
        head_shape = x.shape[:-1]
        # 线性变换，实现最后一维的（因为前面的保留了）
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
        return x

# 多头注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        heads,
        d_model,
        dropout_prob=0.1,
        bias=True
    ):
        super().__init__()

        self.d_k = d_model // heads
        self.heads = heads

        # 初始化QKV
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)

        self.softmax = nn.Softmax(dim=1)

        self.output = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_prob)

        self.scale = 1 / math.sqrt(self.d_k)

        self.attn = None

    def get_scores(self, query, key):
        # i是query位置，b是batch, h是head, d是d_k, j是key位置
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask, query_shape, key_shape):
        # [seq_len_q, seq_len_k, batch_size]
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        # 增加一个维度以便与QK^T匹配
        mask = mask.unsqueeze(-1)
        return mask

    def forward(self, *,
                query,
                key,
                value,
                mask):
        # 传入的是wq,wk,wv [seq_len, batch_size, d_model]
        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
        
        # shape changed -> [seq_len, batch_size, heads, d_k]
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # shape changed -> [seq_len, seq_len, batch_size, heads]
        scores = self.get_scores(query, key)
        scores *= self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = self.softmax(scores)
        tracker.debug('attn', attn)
        attn = self.dropout(attn)

        # 此时[seq_len, batch, heads, d_k]
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)
        
        # 保存注意力权重
        self.attn = attn.detach()

        # 将heads与d_k合并,最终[seq_len, batch_size, d_model]
        x = x.reshape(seq_len, batch_size, -1)
        return self.output(x)

# 前馈神经网络
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
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
    def __init__(
        self,
        d_model,
        dropout_prob,
        max_len=5000
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout_prob)

        # 生成位置编码表并辨识它并非可训练参数
        self.register_buffer("positional_encodings", self.get_positional_encoding(d_model, max_len), False)

    def get_positional_encoding(d_model, max_len=5000):
        encodings = torch.zeros(max_len, d_model)
        # position.shape=[max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
        div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
        encodings[:, 0::2] = torch.sin(position * div_term)
        encodings[:, 1::2] = torch.cos(position * div_term)
        # encodings.shape=[max_len, 1, d_model]
        encodings = encodings.unsqueeze(1).requires_grad_(False)

        return encodings

    def forward(self, x):
        # x.shape=[seq_len, batch_size, d_model]
        pe = self.positional_encodings[:x.shape[0]].detach().requires_grad_(False)
        x = x + pe
        x = self.dropout(x)
        return x

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
        
    def forward(self, src, tgt, src_mask, tgt_mask):
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
    attn = MultiHeadAttention(h, d_model)
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
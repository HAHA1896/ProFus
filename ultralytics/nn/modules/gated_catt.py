import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.cddnet import LayerNorm
from einops import rearrange

##########################################################################
class Gated_CrossAtt(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Gated_CrossAtt, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Gated_Attention(dim, num_heads, bias)

    def forward(self, x_v, x_r):
        x = self.attn(self.norm1(x_v), self.norm1(x_r))

        return x


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class Gated_qdwconv(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(Gated_qdwconv, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        # self.project_out = nn.Conv2d(
        #     hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        # x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Cross-Attention (MDTCA)
class Gated_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Gated_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q_dwconv = Gated_qdwconv(dim=dim, ffn_expansion_factor=1, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(
            dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x_v, x_r):
        b, c, h, w = x_v.shape

        q = self.q_dwconv(x_r)
        kv = self.kv_dwconv(self.kv(x_v))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out




'''各种注意力'''

class AttentionBase(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=4,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q1 = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.q2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=qkv_bias)
        self.kv1 = nn.Conv2d(dim, dim*2, kernel_size=1, bias=qkv_bias)
        self.kv2 = nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=1, bias=qkv_bias)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, xv, xr):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = xv.shape

        q = self.q2(self.q1(xr))
        kv = self.kv2(self.kv1(xv))
        k, v = kv.chunk(2, dim=1)
        # 将q, k, v调整为MultiheadAttention需要的形状 (seq_len, batch, embed_dim)
        q = rearrange(q, 'b c h w -> (h w) b c')
        k = rearrange(k, 'b c h w -> (h w) b c')
        v = rearrange(v, 'b c h w -> (h w) b c')
        
        # 使用MultiheadAttention计算交叉注意力
        attn_output, attn_weights = self.attention(q, k, v)
        
        # 将输出调整回卷积层所需的形状
        attn_output = rearrange(attn_output, '(h w) b c -> b c h w', h=h, w=w)

        out = self.proj(attn_output)
        return out

class MD_CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=True):
        super(MD_CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv_conv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, xv, xr):
        b, c, h, w = x1.shape

        # 生成 Query、Key 和 Value
        q = self.q_conv(xr)
        kv = self.kv_conv(xv)
        k, v = kv.chunk(2, dim=1)

        # 进行深度可分离卷积
        qkv = torch.cat([q, k, v], dim=1)
        qkv = self.qkv_dwconv(qkv)
        q, k, v = qkv.chunk(3, dim=1)

        # 调整维度以符合 torch.nn.MultiheadAttention 的要求
        q = rearrange(q, 'b c h w -> (h w) b c')
        k = rearrange(k, 'b c h w -> (h w) b c')
        v = rearrange(v, 'b c h w -> (h w) b c')

        # 执行交叉注意力
        attn_output, attn_weights = self.attention(q, k, v)

        # 调整输出维度回到卷积的形式
        attn_output = rearrange(attn_output, '(h w) b c -> b c h w', h=h, w=w)
        
        # 应用投影层
        out = self.project_out(attn_output)
        return out

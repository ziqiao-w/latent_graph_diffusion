import torch
import torch.nn as nn
import math

from layer.common import FFN_SwiGLU


def sym_tensor(x):
    x = x.permute(0,3,1,2) # [bs, d, n, n]
    triu = torch.triu(x,diagonal=1).transpose(3,2) # [bs, d, n, n]
    mask = (triu.abs()>0).float()                  # [bs, d, n, n]
    x =  x * (1 - mask ) + mask * triu             # [bs, d, n, n]
    x = x.permute(0,2,3,1) # [bs, n, n, d]
    return x               # [bs, n, n, d]


class SeqNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x): # x [bs, n, d_model]
        x = x.permute(1, 2, 0).contiguous() # [n, d_model, bs]
        x = self.norm(x)
        x = x.permute(2, 0, 1).contiguous() # [bs, n, d_model]
        return x


class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_heads = d_model // n_heads
        self.scale = math.sqrt(self.d_heads)

        self.proj_q = nn.Linear(d_model,  d_model, bias=False)
        self.proj_k = nn.Linear(d_model,  d_model, bias=False)
        self.proj_v = nn.Linear(d_model,  d_model, bias=False)
        self.proj_e = nn.Linear(d_model,  d_model, bias=False)

        self.ni = nn.Linear(d_model, d_model, bias=False)
        self.nj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, h, e): # h [bs, n, d_model], e [bs, n, n, d_model]
        bs, n, _ = h.size()

        def proj_node(x, proj):
            y = proj(x) # [bs, n, d_model]
            y = y.view(bs, n, self.n_heads, self.d_heads) # [bs, n, n_heads, d_heads]
            y = y.permute(0, 2, 1, 3).contiguous() # [bs, n_heads, n, d_heads]
            return y
        
        query = proj_node(h, self.proj_q) # [bs, n_heads, n, d_heads]
        key = proj_node(h, self.proj_k) # [bs, n_heads, n, d_heads]
        value = proj_node(h, self.proj_v) # [bs, n_heads, n, d_heads]

        query = query.unsqueeze(3) # [bs, n_heads, n, 1, d_heads]
        key = key.unsqueeze(2) # [bs, n_heads, 1, n, d_heads]

        add_i = proj_node(h, self.ni).unsqueeze(3) # [bs, n_heads, n, 1, d_heads]
        add_j = proj_node(h, self.nj).unsqueeze(2) # [bs, n_heads, 1, n, d_heads]

        edge_val = self.proj_e(e) # [bs, n, n, d_model]
        edge_val = edge_val.view(bs, n, n, self.n_heads, self.d_heads) # [bs, n, n, n_heads, d_heads]
        edge_val = edge_val.permute(0, 3, 1, 2, 4).contiguous() # [bs, n_heads, n, n, d_heads]

        edge_val = edge_val + add_i + add_j # [bs, n_heads, n, n, d_heads]
        attn = (query * edge_val * key).sum(dim=-1) / self.scale # [bs, n_heads, n, n]
        attn = torch.softmax(attn, dim=-1) # [bs, n_heads, n, n]
        
        out = attn @ value # [bs, n_heads, n, d_heads]
        out = out.permute(0, 2, 1, 3).contiguous() # [bs, n, n_heads, d_heads]
        out = out.view(bs, n, self.n_heads * self.d_heads) # [bs, n, d_model]

        edge_val = edge_val.permute(0, 2, 3, 1, 4).contiguous() # [bs, n, n, n_heads, d_heads]
        edge_val = edge_val.view(bs, n, n, self.n_heads * self.d_heads) # [bs, n, n, d_model]

        return out, edge_val


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.mha = MHA(d_model, n_heads)
        self.drop_h = nn.Dropout(dropout)        
        self.drop_e = nn.Dropout(dropout)
        self.proj_oh = nn.Linear(d_model, d_model)
        self.proj_oe = nn.Linear(d_model, d_model)

    def forward(self, h, e): # h [bs, n, d_model], e [bs, n, n, d_model]
        h, e = self.mha(h, e) # [bs, n, d_model], [bs, n, n, d_model]

        h = self.proj_oh(h) # [bs, n, d_model]
        h = self.drop_h(h) # [bs, n, d_model]

        e = self.proj_oe(e) # [bs, n, n, d_model]
        e = self.drop_e(e) # [bs, n, n, d_model]

        return h, e


class DenseTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.norm_h1 = SeqNorm(d_model)
        self.norm_e1 = nn.LayerNorm(d_model)
        
        self.attention = Attention(d_model, n_heads, dropout)
        self.ffn_h = FFN_SwiGLU(d_model, 4 * d_model)
        self.ffn_e = FFN_SwiGLU(d_model, 4 * d_model)

        self.norm_h2 = SeqNorm(d_model)
        self.norm_e2 = nn.LayerNorm(d_model)

        self.drop_h = nn.Dropout(dropout)
        self.drop_e = nn.Dropout(dropout)

    def forward(self, h, e): # h [bs, n, d_model], e [bs, n, n, d_model]
        h = self.norm_h1(h)
        e = self.norm_e1(e)

        ha, ea = self.attention(h, e) # [bs, n, d_model], [bs, n, n, d_model]
        
        h = h + ha
        h = h + self.ffn_h(self.norm_h2(h))
        h = self.drop_h(h)

        e = e + ea
        e = e + self.ffn_e(self.norm_e2(e))
        e = self.drop_e(e)
        
        e = sym_tensor(e)

        return h, e

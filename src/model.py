from torch import nn
import torch
import numpy as np


class ScaledDotAttention(nn.Module()):
    def __init__(self, a_dropout, a_dim):
        super(ScaledDotAttention).__init__()
        self.dropout = nn.Dropout(a_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.sqrt = np.sqrt(a_dim)

    def forward(self, q, k, v, mask=None):
        atn = torch.bmm(q, k.transpose(1, 2)) / self.sqrt
        if mask is not None:
            atn = atn.masked_fill(mask, -np.inf)
        atn = self.softmax(atn)
        atn = self.dropout(atn)
        out = torch.bmm(atn, v)
        return out, atn


class MultiHeadAttention(nn.Module()):
    def __init__(self, num_heads, in_dim, k_dim, a_dropout=.1):
        super(MultiHeadAttention).__init__()
        self.q = nn.Linear(in_dim, num_heads * k_dim)
        self.k = nn.Linear(in_dim, num_heads * k_dim)
        self.v = nn.Linear(in_dim, num_heads * k_dim)
        self.attention = ScaledDotAttention(a_dropout, k_dim / num_heads)
        self.sqrt = np.sqrt(in_dim)
        self.dropout = nn.Dropout(a_dropout)
        self.num_heads = num_heads
        self.size = k_dim

    def forward(self, q, k, v, mask=None):
        batch, q_len = q.size()
        k_len = k.size(1)
        v_len = v.size(1)

        q = self.q(q).view(batch, q_len, self.num_heads, self.size)
        k = self.k(k).view(batch, k_len, self.num_heads, self.size)
        v = self.v(v).view(batch, v_len, self.num_heads, self.size)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.size)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.size)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, v_len, self.size)

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)

        out, atn = self.attention(q, k, v, mask)
        out = out.view(self.num_heads, batch, q_len, self.dim_value)
        out = out.permute(1, 2, 0, 3).contiguous().view(batch, q_len, -1)
        out = self.dropout(out)

        return out, atn


class PositionalEncoding(nn.Module()):
    def __init__(self, dim, max_len):
        super(PositionalEncoding).__init__()
        pe = torch.zeros(max_len, dim, requires_grad=False)
        pos = torch.arange(0, max_len).unsqueeze(1).type(torch.FloatTensor)
        term = torch.exp(torch.arange(0, dim, 2).type(torch.FloatTensor) * -(np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * term)
        pe[:, 1::2] = torch.cos(pos * term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, inp):
        return self.pe[:, :inp.size(1)]


class Net(nn.Module()):
    def __init__(self, dropout, dim_in, dim_ff):
        super(Net).__init__()
        self.fc1 = nn.Linear(dim_in, dim_ff)
        self.fc2 = nn.Linear(dim_ff, dim_in)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(dim_in)

    def forward(self, input):
        res = input
        out = self.fc1(input)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.layer_norm(out + res)
        return out


class EncodingLayer(nn.Module):
    def __init__(self, num_heads, dim_in, dim_ff, dropout=.01):
        super(EncodingLayer).__init__()
        self.attention = MultiHeadAttention(num_heads, dim_in, dropout)
        self.forward = Net(dropout, dim_in, dim_ff)

    def forward(self, input, mask = None):
        out, atn = self.self_attention(input, input, input, mask)
        out = self.forward(out)
        return out, atn

class Encoder():
    def __init__(self, num_layers, num_heads, dim_in, dim_ff, dropout, maxlen):
        pass






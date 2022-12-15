from torch import nn
import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor


class ScaledDotAttention(nn.Module):
    def __init__(self, a_dropout, a_dim):
        super().__init__()
        self.dropout = nn.Dropout(a_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.sqrt = np.sqrt(a_dim)

    def forward(self, q, k, v):
        atn = torch.bmm(q, k.transpose(1, 2)) / self.sqrt
        atn = self.softmax(atn)
        # print(atn)
        atn = self.dropout(atn)
        out = torch.bmm(atn, v)
        return out, atn


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, in_dim, k_dim, a_dropout=.1):
        super().__init__()
        self.in_dim = in_dim
        self.q_l = nn.Linear(in_dim, num_heads * k_dim)
        self.k_l = nn.Linear(in_dim, num_heads * k_dim)
        self.v_l = nn.Linear(in_dim, num_heads * k_dim)
        self.attention = ScaledDotAttention(a_dropout, in_dim / num_heads)
        self.sqrt = np.sqrt(in_dim)
        self.dropout = nn.Dropout(a_dropout)
        self.num_heads = num_heads
        self.size = k_dim
        self.linear = nn.Linear(k_dim*num_heads, in_dim)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, q, k, v):

        q_len = q.size(1)
        k_len = k.size(1)
        v_len = v.size(1)
        batch = v.size(0)
        res = q
        # print(q.shape)
        q = self.q_l(q)
        # print(q.shape)
        q = q.view(batch, -1, self.num_heads, self.size)
        k = self.k_l(k).view(batch, -1, self.num_heads, self.size)
        v = self.v_l(v).view(batch, -1, self.num_heads, self.size)

        q = q.permute(2, 0, 1, 3).contiguous().view(batch*self.num_heads, -1, self.size)
        k = k.permute(2, 0, 1, 3).contiguous().view(batch*self.num_heads, -1, self.size)
        v = v.permute(2, 0, 1, 3).contiguous().view(batch*self.num_heads, -1, self.size)
        # print(q.shape)
        out, atn = self.attention(q, k, v)
        out = out.view(self.num_heads, batch, -1, self.size)
        out = out.permute(1, 2, 0, 3).contiguous().view(batch, -1,self.num_heads*self.size)
        # print(out.shape)
        # print(out.shape)
        out = self.linear(out)
        out = self.dropout(out)
        # print(out.shape)
        # print(res.shape)
        out = self.norm(out+res)

        return out, atn


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super().__init__()
        pe = torch.zeros(max_len, dim, requires_grad=False)
        pos = torch.arange(0, max_len).unsqueeze(1).type(torch.FloatTensor)
        term = torch.exp(torch.arange(0, dim, 2).type(torch.FloatTensor) * -(np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * term)
        pe[:, 1::2] = torch.cos(pos * term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, inp, size):
        # print("pe")
        # print(self.pe.shape)
        # print("not pe")
        return self.pe[:, :size]


class FeedForward(nn.Module):
    def __init__(self, dropout, dim_in, dim_ff):
        super().__init__()
        # print(dim_in)
        # print(dim_ff)
        self.ff = nn.Sequential(
            nn.Linear(dim_in, dim_ff),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_in),
            nn.Dropout(dropout)
        )

    def forward(self, input):
        return self.ff(input)


class Net(nn.Module):
    def __init__(self, sub, dim, dropout):
        super().__init__()
        self.sub = sub
        self.dropout = nn.Dropout(dropout)
        self.layer = nn.LayerNorm(dim)

    def forward(self, *inputs):
        # print(input.shape)
        res = inputs[0]
        # print(res.shape)
        # if isinstance(self.sub, FeedForward):
        #     out = self.sub(input)
        # elif isinstance(self.sub, MultiHeadAttention):
        #     # print("for shapes")
        #     # print(input.shape)
        #     # print(memory.shape)
        #     out = self.sub(input, memory, memory)
        out = self.sub(*inputs)
        # print(res.shape)
        # print(out[0].shape)
        # print(out[0].shape)
        if isinstance(out, tuple):
            return self.layer(out[0] + res), out[1]
        return self.layer(out+res)


class EncodingLayer(nn.Module):
    def __init__(self, num_heads, dim_in, dim_ff, dropout=.01):
        super().__init__()
        self.attention = Net(MultiHeadAttention(num_heads, dim_in, dim_ff, dropout), dim_in, dropout)
        self.ff = Net(FeedForward(dropout, dim_in, dim_ff), dim_in, dropout)

    def forward(self, input):
        out, atn = self.attention(input, input, input)
        out = self.ff(out)
        return out, atn


class Encoder(nn.Module):
    # mask?
    def __init__(self, num_layers, num_heads, dim_in, dim_ff, dropout, maxlen):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.pos_encoding = PositionalEncoding(dim_in, maxlen)
        self.layers = nn.ModuleList([
            EncodingLayer(num_heads, dim_in, dim_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, input):
        encoder_self_attn_list = []
        # print(type(input))
        pos = input + self.pos_encoding(input, input.size(1))
        out = self.drop(pos)
        for layer in self.layers:
            encoder_output, self_attn = layer(out)
            encoder_self_attn_list += [self_attn]

        return encoder_output


class DecodingLayer(nn.Module):
    def __init__(self, num_heads, in_dim, dim_ff, dropout):
        super().__init__()
        self.atn1 = Net(MultiHeadAttention(num_heads, in_dim, dim_ff, dropout), in_dim, dropout)
        self.atn2 = Net(MultiHeadAttention(num_heads, in_dim, dim_ff, dropout), in_dim, dropout)
        self.ff = Net(FeedForward(dropout, in_dim, dim_ff), in_dim, dropout)

    def forward(self, input, memory):
        dec = self.atn1(input, input, input)
        # print(memory.shape)
        # print(dec[0].shape)
        input = self.atn2(dec[0], memory, memory)
        return self.ff(input[0])


class Decoder(nn.Module):
    def __init__(self, num_layers, num_classes, num_heads, dim_emb, dim_in, dim_ff, maxlen, dropout):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(num_classes, dim_emb)
        self.pos = PositionalEncoding(dim_in, maxlen)
        self.layers = nn.ModuleList([
            DecodingLayer(num_heads, dim_in, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Sequential(nn.LayerNorm(dim_in), nn.Linear(dim_in, num_classes, bias=False))
        self.maxlen = maxlen

    def forward(self, inputs, memory):
        outs = []
        batch = inputs.size(0)
        inps = inputs.new_zeros(batch, self.maxlen).long()

        for i in range(1,self.maxlen):
            print(i)
            out = self.encoder_forward(inps[:, :i], memory, i)
            step = self.fc(out).log_softmax(dim=-1)
            outs.append(step[:, -1, :])
            inps = outs[-1].topk(1)[1]
        print("done")
        # print(out.shape)
        outs = torch.stack(outs, dim=1)
        return outs


    def encoder_forward(self, inputs, memory, lens):
        # print(inputs.shape)
        out = self.embedding(inputs.long()) + self.pos(inputs, lens)
        # print(out.shape)/
        # out = self.dropout(out)
        for layer in self.layers:
            # print(out.shape)
            # print(memory.shape)
            out = layer(out,memory)
        return out


class Transformer(nn.Module):
    def __init__(self, num_layers, num_classes, num_heads, dim_emb, dim_in, dim_ff, maxlen, dropout):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor()
        self.encoder = Encoder(num_layers, num_heads, dim_in, dim_ff, dropout, maxlen)
        self.decoder = Decoder(num_layers, num_classes, num_heads, dim_emb, dim_in, dim_ff, maxlen, dropout)
    def forward(self, input, targets):
        # out = self.feature_extractor(input)
        print("encoding :>")
        # print(out.shape)
        print(input.shape)
        enc = self.encoder(input)
        print("decoding ...")
        out = self.decoder(targets, enc)
        preds = out.max(-1)[1]
        return preds

# if __name__ == "__main__":
#     torch.cuda.empty_cache()
#     device = torch.device("cuda")
#     src = torch.rand(32, 16, 64)
#     tgt = torch.rand(32, 8, 64)
#     model = Transformer(6, 6, 8, 64, 64, 64, 100, .01)
#     # model = model.to(device)
#     out = model(src, tgt)
#     print(out.shape)




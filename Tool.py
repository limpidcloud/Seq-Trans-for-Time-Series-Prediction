import torch
import torch.nn as nn


class Normalization(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        avg = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return self.scale*(x-avg)/(std + 1e-6) + self.bias


class ResidualConnection(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.norm = Normalization(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, model):
        hidden = self.dropout(model(self.norm(x)))
        return x + hidden


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim))

    def forward(self, x):
        return self.seq(x)


# 本函数用于基于历史序列生成数据集
def generate_seq(seq_len, value_seq, phase_seq):
    # value_seq的维度为batch*len*channel
    # phase_seq的维度为batch*len*4
    assert value_seq.shape[1] % seq_len == 0
    channel_dim = value_seq.shape[-1]
    value_seq = value_seq.transpose(-1, -2).contiguous()
    value_sub = value_seq.reshape(value_seq.shape[0], -1, seq_len)
    phase_sub = phase_seq[:,::seq_len, :].unsqueeze(-2).repeat(1,1, channel_dim, 1).reshape(phase_seq.shape[0], -1, phase_seq.shape[-1])
    return value_sub, phase_sub

# 定义频域former的attention机制
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as f


# 定义注意力机制
def attention(query, key, value, dropout):
    # 本部分和普通的attention并没有太多区别，然而维度意义不同
    # 维度为batch*head_num*seq_num*head_dim
    assert query.shape[-1] == key.shape[-1]
    head_dim = query.shape[-1]
    score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
    p_attn = f.softmax(score, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    v_attn = torch.matmul(p_attn, value)
    return v_attn, p_attn


# 定义多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, head_num, dropout=0.1):
        super().__init__()
        assert embed_dim % head_num == 0
        self.head_dim = embed_dim // head_num
        self.head_num = head_num
        self.dropout = nn.Dropout(dropout)
        self.linear_list = nn.ModuleList([copy.deepcopy(nn.Linear(embed_dim, embed_dim)) for _ in range(4)])

    def forward(self, query, key, value):
        # 首先对q和k和v按照head_num作分解
        # 输入维度为batch*seq_num*embed_dim
        batch_size = query.shape[0]
        query, key, value = [linear(x).reshape(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2) for linear, x in zip(self.linear_list[:3], (query, key, value))]
        x, _ = attention(query, key, value, self.dropout)
        x = x.transpose(1, 2).contiguous().reshape(batch_size, -1, self.head_dim*self.head_num)
        return self.linear_list[-1](x)

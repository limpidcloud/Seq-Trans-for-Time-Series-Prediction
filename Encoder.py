import copy

import torch
import torch.nn as nn

import Tool as t


class EncoderLayer(nn.Module):
    def __init__(self, attn, feed_forward, embed_dim, dropout=0.1):
        super().__init__()
        self.attn = attn
        self.feed = feed_forward
        self.embed_dim = embed_dim
        self.layer_list = nn.ModuleList([copy.deepcopy(t.ResidualConnection(embed_dim, dropout)) for _ in range(2)])

    def forward(self, x):
        # 此时x的维度为batch*(seq_num*scale)*embed_dim
        x = self.layer_list[0](x, lambda e: self.attn(e, e, e))
        return self.layer_list[1](x, self.feed)


class DeepEncoder(nn.Module):
    def __init__(self, encoder_layer, layer_num):
        super().__init__()
        self.layer_list = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(layer_num)])

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x


class WideEncoder(nn.Module):
    def __init__(self, encoder_layer, layer_num):
        super().__init__()
        self.layer_list = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(layer_num)])

    def forward(self, x_list):
        # 每个x的维度是batch*seq_num*embed_dim
        x_attn_list = []
        for x, layer in zip(x_list, self.layer_list):
            x_attn_list.append(layer(x))
        return torch.cat(tuple(x_attn_list), dim=1)


class Encoder(nn.Module):
    def __init__(self, encoder_layer, len_num, layer_num):
        super().__init__()
        self.wide_layer = WideEncoder(encoder_layer, len_num)
        self.deep_layer = DeepEncoder(encoder_layer, layer_num)
        self.norm = t.Normalization(encoder_layer.embed_dim)

    def forward(self, x_list):
        x = self.wide_layer(x_list)
        x = self.deep_layer(x)
        return self.norm(x)

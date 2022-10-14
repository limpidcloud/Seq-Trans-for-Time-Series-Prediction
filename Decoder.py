import copy

import torch.nn as nn

import Tool as t


class DecoderLayer(nn.Module):
    def __init__(self, self_attn, src_attn, feed_forward, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed = feed_forward
        self.layer_list = nn.ModuleList([copy.deepcopy(t.ResidualConnection(embed_dim, dropout)) for _ in range(3)])

    def forward(self, x, m):
        # m由encoder传过来，维度为batch*seq_num*head_num
        # x为decoder的输入，维度也应当为batch*seq_num*head_num
        x = self.layer_list[0](x, lambda e: self.self_attn(e, e, e))
        x = self.layer_list[1](x, lambda e: self.src_attn(e, m, m))
        return self.layer_list[2](x, self.feed)


class Decoder(nn.Module):
    def __init__(self, decoder_layer, layer_num):
        super().__init__()
        self.layer_list = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(layer_num)])
        self.norm = t.Normalization(decoder_layer.embed_dim)

    def forward(self, x, m):
        for layer in self.layer_list:
            x = layer(x, m)
        return self.norm(x)

import copy
import torch.nn as nn

import Attention as a
import Embed as em
import Encoder as e
import Decoder as d
import Generator as g
import Tool as t
from config import *


class SeqTrans(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        assert len(seq_len_list) == len(seq_num_list)
        attn = a.MultiHeadAttention(embed_dim, head_num, dropout)
        feed = t.FeedForward(embed_dim, dropout)
        encoder_layer = e.EncoderLayer(copy.deepcopy(attn), copy.deepcopy(feed), embed_dim, dropout)
        self.encoder = e.Encoder(encoder_layer, len(seq_len_list), layer_num+1)
        self.decoder = d.Decoder(d.DecoderLayer(copy.deepcopy(attn), copy.deepcopy(attn), copy.deepcopy(feed), embed_dim, dropout), layer_num)
        self.generator = g.Generator(conv_num, embed_dim, predict_len, out_dim)
        self.src_embed = nn.ModuleList([copy.deepcopy(
            em.MixEmbedding(seq_num_list[i], seq_len_list[i], aim_len, embed_dim, kernel_len, conv_num, phase_list)
        )
             for i in range(len(seq_num_list))])
        self.tgt_embed = em.MixEmbedding(in_dim, predict_len, aim_len, embed_dim, kernel_len, conv_num, phase_list)

    def process_en(self, value_en, phase_en):
        x_list = tuple()
        for i in range(len(seq_len_list)):
            v_en, p_en = t.generate_seq(seq_len_list[i], value_en, phase_en)  # 分割数据至指定维度
            x = self.src_embed[i](v_en, p_en)  # 生成当前sequence长度的embedding结果
            x_list = x_list + tuple([x])
        return x_list

    def process_de(self, value_de, phase_de):
        v_de, p_de = t.generate_seq(predict_len, value_de, phase_de)
        x = self.tgt_embed(v_de, p_de)
        return x

    def encode(self, value_en, phase_en):
        x_list = self.process_en(value_en, phase_en)
        return self.encoder(x_list)

    def decode(self, code, value_de, phase_de):
        return self.decoder(self.process_de(value_de, phase_de), code)

    def forward(self, value_en, phase_en, value_de, phase_de):
        code = self.encode(value_en, phase_en)
        code = self.decode(code, value_de, phase_de)
        return self.generator(code)

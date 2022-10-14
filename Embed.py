# 定义频域former的embed方案
import copy

import torch
import torch.nn as nn
from config import device

# 对每组序列作embedding
class SeqEmbedding(nn.Module):
    def __init__(self, seq_num, aim_len, embed_dim, kernel_len, conv_num=16):
        super().__init__()
        # 对序列作卷积
        self.conv = nn.Sequential(
            nn.Conv1d(seq_num, conv_num, kernel_len, stride=1, padding=(kernel_len - 1) // 2), nn.LeakyReLU(),
            nn.Conv1d(conv_num, conv_num, kernel_len, stride=1, padding=(kernel_len - 1) // 2), nn.LeakyReLU(),
            nn.Conv1d(conv_num, conv_num, kernel_len, stride=1, padding=(kernel_len - 1) // 2)
            )
        self.embed = nn.Linear(aim_len, embed_dim)  # 对序列作embedding

    def forward(self, mix):
        # 输入维度为batch*seq_num*seq_len
        # 首先在seq_len方向上作卷积提取特征得到维度batch*seq_num*aim_len
        x = self.conv(mix)
        # 得到新维度为batch*(seq_num*scale)*seq_len
        return self.embed(x)  # 最后输出的维度为batch*(seq_num*scale)*embed_dim


# 输入事件序列的相位，对相位作embedding
class PhaseEmbedding(nn.Module):
    def __init__(self, phase_list, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_list = nn.ModuleList([copy.deepcopy(nn.Embedding(num, embed_dim)) for num in phase_list])

    def forward(self, phase):
        phase = phase.long()
        # 输入维度为batch*seq_num*phase_len
        embed = torch.zeros([phase.shape[0], phase.shape[1], self.embed_dim]).to(device)
        for i in range(len(self.embed_list)):
            embed += self.embed_list[i](phase[:, :, i])
        return embed.reshape(phase.shape[0], -1, self.embed_dim)


class MixEmbedding(nn.Module):
    def __init__(self, seq_num, seq_len, aim_len, embed_dim, kernel_len, conv_num, phase_list):
        super().__init__()
        assert seq_len % aim_len <= seq_len // aim_len
        self.phase_embed = PhaseEmbedding(phase_list, aim_len)
        self.seq_embed = SeqEmbedding(seq_num, aim_len, embed_dim, kernel_len, conv_num)
        self.pool = nn.AvgPool1d(kernel_size=seq_len//aim_len, stride=seq_len//aim_len)

    def forward(self, value, phase):
        # 将输入的phase拉长至与aim_len一个长度, 将序列取平均压缩至一个长度
        # 由于如果限制死维度，并不能保证各种输入维度均能被正好划分，因此尽量让aim被seq整除
        tmp1 = self.pool(value)
        tmp2 = self.phase_embed(phase)
        mix = tmp1 + tmp2
        return self.seq_embed(mix)  # 输出的维度是batch*seq_num

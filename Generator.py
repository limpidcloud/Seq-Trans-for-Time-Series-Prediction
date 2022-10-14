import torch
import torch.nn as nn
from config import device


class Generator(nn.Module):
    def __init__(self, conv_num, embed_dim, seq_len, out_dim):
        super().__init__()
        self.seq_len = seq_len
        self.out_dim = out_dim
        self.generate_list = nn.ModuleList([nn.Linear(embed_dim * conv_num, seq_len) for _ in range(out_dim)])

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        result = torch.zeros(x.shape[0], self.out_dim, self.seq_len).to(device)
        for i in range(self.out_dim):
            result[:,i,:] = self.generate_list[i](x)
        return result.transpose(-1, -2)

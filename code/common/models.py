"""FM 和 DDPM 共用的网络。为了公平对比，两边都用这同一个 MLP，只是输出的含义不同：
FM 输出速度场 v(x, t)，DDPM 输出噪声 eps(x, t)。时间 t 统一缩放到 [0, 1]。"""

import math

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * 1000.0 * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class MLP(nn.Module):
    def __init__(self, dim: int = 2, width: int = 256, depth: int = 3, t_dim: int = 64):
        super().__init__()
        self.t_embed = TimeEmbedding(t_dim)
        layers, d = [], dim + t_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers.append(nn.Linear(d, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.net(torch.cat([x, self.t_embed(t)], dim=1))

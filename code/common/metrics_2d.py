"""2D 实验的样本质量指标。输入都是 [n, 2] 的 tensor 或 ndarray。"""

import numpy as np
import ot
import torch


def _np(x):
    return x.detach().cpu().numpy().astype(np.float64) if isinstance(x, torch.Tensor) else np.asarray(x, np.float64)


def wasserstein2(x, y) -> float:
    """经验分布之间的精确 W2 距离（POT 的 EMD 求解器），n 不宜超过约 5000。"""
    x, y = _np(x), _np(y)
    cost = ot.dist(x, y)  # 平方欧氏距离
    a, b = np.full(len(x), 1 / len(x)), np.full(len(y), 1 / len(y))
    return float(np.sqrt(ot.emd2(a, b, cost, numItermax=1_000_000)))


def mmd_rbf(x, y, bandwidths=(0.1, 0.2, 0.5, 1.0, 2.0)) -> float:
    """多带宽 RBF 核的 MMD^2 无偏估计。"""
    x, y = torch.as_tensor(_np(x)), torch.as_tensor(_np(y))
    xx, yy, xy = torch.cdist(x, x) ** 2, torch.cdist(y, y) ** 2, torch.cdist(x, y) ** 2
    n, m = len(x), len(y)
    total = 0.0
    for h in bandwidths:
        kxx, kyy, kxy = (torch.exp(-d / (2 * h**2)) for d in (xx, yy, xy))
        total += (kxx.sum() - n) / (n * (n - 1)) + (kyy.sum() - m) / (m * (m - 1)) - 2 * kxy.mean()
    return float(total)


def straightness(traj) -> float:
    """Rectified Flow 论文中的直线度：S = mean_k E||(x_N - x_0) - (x_{k+1} - x_k)/dt||^2。
    traj: [K+1, n, 2]，确定性采样器（ODE / DDIM）的完整轨迹，时间均匀分成 K 段。0 表示完全直线。"""
    traj = torch.as_tensor(_np(traj))
    k = traj.shape[0] - 1
    chord = traj[-1] - traj[0]
    vel = (traj[1:] - traj[:-1]) * k
    return float(((vel - chord[None]) ** 2).sum(-1).mean())

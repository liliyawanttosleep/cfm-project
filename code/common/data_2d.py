"""2D toy 数据集。所有数据集都标准化到大致均值 0、标准差 1，方便不同方法共用同一个噪声分布 N(0, I)。"""

import numpy as np
import torch
from sklearn.datasets import make_moons, make_swiss_roll

DATASETS = ["moons", "8gaussians", "swissroll", "checkerboard"]


def _raw(name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if name == "moons":
        x, _ = make_moons(n, noise=0.05, random_state=int(rng.integers(1 << 31)))
    elif name == "8gaussians":
        centers = np.array([(np.cos(a), np.sin(a)) for a in np.arange(8) * np.pi / 4])
        x = 2.0 * centers[rng.integers(8, size=n)] + 0.1 * rng.standard_normal((n, 2))
    elif name == "swissroll":
        x, _ = make_swiss_roll(n, noise=0.5, random_state=int(rng.integers(1 << 31)))
        x = x[:, [0, 2]]
    elif name == "checkerboard":
        x1 = rng.uniform(-2, 2, n)
        x2 = rng.uniform(0, 1, n) + rng.integers(0, 2, n) * 2 - 2 + np.floor(x1) % 2
        x = np.stack([x1, x2], axis=1)
    else:
        raise ValueError(f"unknown dataset {name}, choose from {DATASETS}")
    return x.astype(np.float32)


# 用一份固定的大样本算标准化常数，保证训练集和测试集用同一套常数
_STATS = {}


def _stats(name):
    if name not in _STATS:
        ref = _raw(name, 100_000, np.random.default_rng(12345))
        _STATS[name] = (ref.mean(0), ref.std(0))
    return _STATS[name]


def sample(name: str, n: int, rng: np.random.Generator) -> torch.Tensor:
    """从数据分布中采 n 个点，返回 float32 tensor [n, 2]。"""
    mean, std = _stats(name)
    return torch.from_numpy((_raw(name, n, rng) - mean) / std)

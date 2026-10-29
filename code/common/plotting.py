"""2D 实验的公共画图函数。"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_samples_by_nfe(real, samples: dict, title: str, path, traj=None):
    """第一张子图画真实数据，后面每张子图画一个 NFE 下的生成样本。
    samples: {nfe: [n, 2]}；traj: 可选 [K+1, n, 2]，在最后一张子图上叠加画轨迹。"""
    keys = list(samples)
    ncol = len(keys) + 1 + (traj is not None)
    fig, axes = plt.subplots(1, ncol, figsize=(2.6 * ncol, 2.8))
    lim = 3.2
    axes[0].scatter(real[:, 0], real[:, 1], s=1, c="k", alpha=0.5)
    axes[0].set_title("real data")
    for ax, k in zip(axes[1:], keys):
        s = np.asarray(samples[k])
        ax.scatter(s[:, 0], s[:, 1], s=1, c="tab:blue", alpha=0.5)
        ax.set_title(f"NFE={k}")
    if traj is not None:
        ax, traj = axes[-1], np.asarray(traj)
        idx = np.arange(min(300, traj.shape[1]))
        ax.plot(traj[:, idx, 0], traj[:, idx, 1], c="olive", alpha=0.25, lw=0.5)
        ax.scatter(traj[0, idx, 0], traj[0, idx, 1], s=2, c="k")
        ax.scatter(traj[-1, idx, 0], traj[-1, idx, 1], s=2, c="tab:blue")
        ax.set_title("trajectories")
    for ax in axes:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

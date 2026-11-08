"""第 27 页的数字：minibatch OT 的配对在 2D 和 MNIST 上分别改变了多少。只用 CPU。

    CUDA_VISIBLE_DEVICES="" python slides/make_ot_numbers.py

要完成的任务
    任务 1  配对代价：E‖x0 - x1‖² 在随机配对和 batch 内 OT 配对下的值，2D 和 MNIST 用同样的 batch 128
    任务 2  batch 扫描：MNIST 上 batch 16 到 2048，看 batch 变大能省多少
    任务 3  上限：配对只能改变内积项，E‖x0 - x1‖² = E‖x0‖² + E‖x1‖² - 2 E<x0, x1>。
            Gelbrich 下界 W2²(N(0, I), P) >= ‖μ‖² + tr(I + Σ - 2 Σ^(1/2)) 说明，任何配对最多省 2 tr Σ^(1/2)。
            用它算“实际省下的 / 最多能省的”，这比比值 ot / random 更能说明配对改变了多少
    任务 4  2D 的路径长度比，用 E1 的权重（来源 N(0, I)），和 eval_images.py 的 MNIST 口径相同

输出 results/e4/ot_numbers.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import ot
import torch
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from common.data_2d import DATASETS, sample  # noqa: E402
from common.models import MLP  # noqa: E402

device = "cpu"


def pair_cost(draw, batch, rounds, seed=0):
    """draw(batch, generator) returns x1 flattened; the noise has the same shape."""
    g = torch.Generator().manual_seed(seed)
    rnd, opt = [], []
    for _ in range(rounds):
        x1 = draw(batch, g)
        x0 = torch.randn(x1.shape, generator=g)
        M = torch.cdist(x0, x1).pow(2).numpy().astype(np.float64)
        rnd.append(float(np.trace(M) / batch))
        opt.append(float(ot.emd2(ot.unif(batch), ot.unif(batch), M, numItermax=2_000_000)))
    return {"batch": batch, "rounds": rounds, "random": float(np.mean(rnd)), "ot": float(np.mean(opt)),
            "ratio": float(np.mean(opt) / np.mean(rnd)), "saved": float(np.mean(rnd) - np.mean(opt))}


def most_saved(x):
    """2 tr Σ^(1/2): the most any coupling of N(0, I) with this distribution can save over independent pairs."""
    x = x.astype(np.float64)
    cov = np.cov(x, rowvar=False)
    eig = np.linalg.eigvalsh(cov).clip(min=0)
    return {"tr_cov": float(eig.sum()), "two_tr_sqrt_cov": float(2 * np.sqrt(eig).sum()), "mean_norm2": float((x.mean(0) ** 2).sum())}


@torch.no_grad()
def path_ratio_2d(method, data, n=2000, k=100):
    net = MLP(width=256)
    net.load_state_dict(torch.load(ROOT / f"results/e1/ckpt/{method}_{data}_s0.pt", map_location=device, weights_only=True))
    net.eval()
    x = torch.randn(n, 2, generator=torch.Generator().manual_seed(3))
    start, length = x.clone(), torch.zeros(n)
    for i in range(k):
        v = net(x, torch.full((n,), i / k))
        length += v.norm(dim=1) / k
        x = x + v / k
    # 总长度 / 总直线距离，与第 18 页同一口径。2D 里有些点几乎不动，先除再平均会被它们放大
    return float(length.sum() / (x - start).norm(dim=1).sum())


def main():
    out = {"2d": {}, "mnist": {}}
    for name in DATASETS:
        rng = np.random.default_rng(0)
        big = sample(name, 50_000, rng).numpy()
        pool = torch.tensor(big, dtype=torch.float32)
        rec = {"pairs_128": pair_cost(lambda b, g: pool[torch.randperm(len(pool), generator=g)[:b]], 128, 20), **most_saved(big)}
        rec["share_of_most_saved"] = rec["pairs_128"]["saved"] / rec["two_tr_sqrt_cov"]
        rec["path_ratio"] = {m: path_ratio_2d(m, name) for m in ["icfm", "otcfm"]}
        out["2d"][name] = rec
        print(name, json.dumps(rec), flush=True)

    x = datasets.MNIST(ROOT / "data", train=True, download=False).data.float().flatten(1) / 127.5 - 1
    out["mnist"] = most_saved(x.numpy())
    print("mnist", out["mnist"], flush=True)
    out["mnist"]["pairs"] = []
    for batch, rounds in [(16, 200), (128, 20), (512, 5), (2048, 2)]:
        rec = pair_cost(lambda b, g: x[torch.randperm(len(x), generator=g)[:b]], batch, rounds)
        rec["share_of_most_saved"] = rec["saved"] / out["mnist"]["two_tr_sqrt_cov"]
        out["mnist"]["pairs"].append(rec)
        print(rec, flush=True)
        (ROOT / "results/e4/ot_numbers.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

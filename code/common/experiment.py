"""E1 两条线（FM / DDPM）共用的实验骨架。要完成的任务：统一参数和随机种子，用同一份噪声和测试集按 NFE 评估，保存结果，保证比较公平。"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from common import metrics_2d
from common.data_2d import DATASETS, sample
from common.plotting import plot_samples_by_nfe

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "e1"


def base_parser(methods, default_nfes):
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=methods, default=methods[0])
    p.add_argument("--data", choices=DATASETS, default="moons")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=20_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-eval", type=int, default=2000)
    p.add_argument("--nfes", type=int, nargs="+", default=default_nfes)
    p.add_argument("--out", type=Path, default=RESULTS)
    return p


def setup(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return np.random.default_rng(args.seed), device


def log_progress(step, total, loss, t0):
    if step % max(1, total // 10) == 0 or step == total:
        print(f"  step {step:>6}/{total}  loss {loss:.4f}  ({time.time() - t0:.0f}s)", flush=True)


@torch.no_grad()
def evaluate(args, name, sampler, train_seconds, deterministic=True):
    """sampler(x0, nfe) -> 轨迹 [nfe+1, n, 2]，traj[0] 是噪声，traj[-1] 是生成样本。
    所有方法用同一个 seed 的噪声 x0 和同一份测试集，便于对比。"""
    rng = np.random.default_rng(10_000 + args.seed)
    real = sample(args.data, args.n_eval, rng)
    real_ref = sample(args.data, args.n_eval, rng)  # 两份独立真实样本之间的距离 = 有限样本下指标的“噪声底”
    g = torch.Generator().manual_seed(20_000 + args.seed)
    x0 = torch.randn(args.n_eval, 2, generator=g)

    rows, finals, last_traj = [], {}, None
    for nfe in args.nfes:
        t0 = time.time()
        traj = sampler(x0, nfe).cpu()
        sample_seconds = time.time() - t0
        gen = traj[-1]
        rows.append(
            {
                "nfe": nfe,
                "w2": metrics_2d.wasserstein2(gen, real),
                "mmd": metrics_2d.mmd_rbf(gen, real),
                "straightness": metrics_2d.straightness(traj) if deterministic else None,
                "sample_seconds": sample_seconds,
            }
        )
        print("  " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rows[-1].items()))
        finals[nfe], last_traj = gen.numpy(), traj

    result = {
        "method": args.method,
        "data": args.data,
        "seed": args.seed,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "train_seconds": train_seconds,
        "reference": {"w2": metrics_2d.wasserstein2(real_ref, real), "mmd": metrics_2d.mmd_rbf(real_ref, real)},
        "by_nfe": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.json").write_text(json.dumps(result, indent=2))
    plot_samples_by_nfe(
        real.numpy(), finals, f"{args.method} on {args.data} (seed {args.seed})", args.out / f"{name}.png",
        traj=last_traj if deterministic else None,
    )
    print(f"  saved {args.out / name}.json/.png")
    return result


def save_checkpoint(args, name, model):
    (args.out / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "ckpt" / f"{name}.pt")

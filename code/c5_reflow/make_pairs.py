"""C5 step 1: reflow pairs. Integrate a trained CFM model with dopri5 from fresh noise and keep (x0, x1_hat).

    python -u code/c5_reflow/make_pairs.py --method icfm --seed 0

Tasks
    Task 1  60,000 noise draws x0 ~ N(0, I) (generator seed 300, so the pairs never reuse the FID noise seeds 100-105),
            each integrated from t = 0 to t = 1 with dopri5 (rtol = atol = 1e-5, as in E4 and C2).
    Task 2  Save in chunks of 2,000 to results/c5_reflow/pairs/{method}_s{seed}/chunk_XXX.pt (x0 and x1_hat, float32).
            A rerun skips finished chunks, so a crash or driver reset loses at most one chunk.
    Task 3  Record the mean dopri5 NFE and GPU seconds (the cost of making pairs, C5 step 6), and the pair cost
            E||x0 - x1_hat||^2 next to random pairing and minibatch OT (C5 step 5, continues C3).
    Task 4  Throttle (code/common/throttle.py).

The pairs are about 376 MB per model and stay out of git (results/**/pairs/).
Output results/c5_reflow/pairs_{method}_s{seed}.json.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
import numpy as np
import torch
from torchdiffeq import odeint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from common.throttle import Throttle  # noqa: E402

spec = importlib.util.spec_from_file_location("eval_images", ROOT / "code/e4_cifar/eval_images.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


class Counted(torch.nn.Module):
    """Counts network calls; the throttle sleeps after each call so dopri5 batches stay at the chosen duty."""

    def __init__(self, net, th):
        super().__init__()
        self.net, self.th, self.nfe = net, th, 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.net(t.expand(x.shape[0]), x)
        self.th.step()
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["icfm", "otcfm"], required=True)
    p.add_argument("--seed", type=int, default=0, help="training seed of the source model")
    p.add_argument("--n", type=int, default=60000)
    p.add_argument("--chunk", type=int, default=2000)
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--duty", type=float, default=0.5)
    p.add_argument("--max-temp", type=int, default=72)
    p.add_argument("--resume-temp", type=int, default=65)
    p.add_argument("--min-free-gb", type=float, default=1.5)
    args = p.parse_args()
    th = Throttle(args.duty, args.max_temp, args.resume_temp, args.min_free_gb, threads=int(os.environ["OMP_NUM_THREADS"]))
    d = ROOT / f"results/c5_reflow/pairs/{args.method}_s{args.seed}"
    d.mkdir(parents=True, exist_ok=True)
    meta_f = ROOT / f"results/c5_reflow/pairs_{args.method}_s{args.seed}.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {"method": args.method, "seed": args.seed, "tol": args.tol,
                                                                     "noise_seed": 300, "chunks": {}}
    net = ev.load_mnist(args.method, args.seed)
    f = Counted(net, th)
    g = torch.Generator().manual_seed(300)
    n_chunks = args.n // args.chunk
    for c in range(n_chunks):
        x0 = torch.randn(args.chunk, 1, 28, 28, generator=g)  # drawn even for finished chunks, so every chunk keeps its noise
        out = d / f"chunk_{c:03d}.pt"
        if out.exists():
            continue
        t0, nfes, x1s = time.time(), [], []
        th.reset()
        with torch.no_grad():
            for b in x0.split(args.batch):
                f.nfe = 0
                y = odeint(f, b.to(ev.device), torch.tensor([0.0, 1.0], device=ev.device), rtol=args.tol, atol=args.tol, method="dopri5")[-1]
                x1s.append(y.cpu())
                nfes.append(f.nfe)
        x1 = torch.cat(x1s)
        tmp = out.with_suffix(".tmp")
        torch.save({"x0": x0, "x1": x1}, tmp)
        os.replace(tmp, out)
        meta["chunks"][str(c)] = {"mean_nfe": float(np.mean(nfes)), "seconds": time.time() - t0,
                                  "pair_cost": float((x1 - x0).flatten(1).pow(2).sum(1).mean())}
        meta_f.write_text(json.dumps(meta, indent=2))
        print(f"chunk {c + 1}/{n_chunks}", {k: round(v, 2) for k, v in meta["chunks"][str(c)].items()}, flush=True)

    ch = meta["chunks"].values()
    meta["summary"] = {"n": args.n, "mean_nfe": float(np.mean([v["mean_nfe"] for v in ch])),
                       "gpu_seconds": float(sum(v["seconds"] for v in ch)), "pair_cost": float(np.mean([v["pair_cost"] for v in ch]))}
    meta_f.write_text(json.dumps(meta, indent=2))
    print("done", meta["summary"], flush=True)


if __name__ == "__main__":
    main()

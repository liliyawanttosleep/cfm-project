"""Evaluate image models for Part V of the talk: FID vs NFE, fixed-noise grids, and the checks the slides quote.

    python code/e4_cifar/eval_images.py --data mnist --n 10000
    python code/e4_cifar/eval_images.py --data cifar --n 10000

Models
    mnist : results/e2/ckpt/{icfm,otcfm,ddpm}_s0.pt, trained by code/e2_mnist/train_mnist.py
    cifar : official EMA weights at 400k steps, results/e4/weights/{icfm,otcfm}/

Metrics
    FID uses clean-fid. CIFAR-10: mode legacy_tensorflow, reference stats of the train split, as in the library's
    compute_fid.py. MNIST: mode clean, reference from the 60,000 train images, so ImageNet Inception on digits.
    A real-vs-real reference: FID of n train images against all of them, which is the bias of FID at this n.

Also records
    one_step_to_mean : RMS pixel distance per pixel from one Euler step to the data mean, and to the nearest train image
    path             : length / straight line along 100 Euler steps
    pairs            : mean squared pair distance under random and OT pairs, at the training batch size

Outputs results/e4/{data}_eval.json and {data}_samples.npz (uint8 grids for the figures).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import ot
import torch
from cleanfid import fid
from torchdiffeq import odeint
from torchvision import datasets, transforms

from torchcfm.models.unet import UNetModel
from torchcfm.models.unet.unet import UNetModelWrapper

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "e4"
NFES = [1, 2, 5, 10, 20, 50, 100]
T = 1000
device = torch.device("cuda")


def load_cifar(method):
    net = UNetModelWrapper(dim=(3, 32, 32), num_res_blocks=2, num_channels=128, channel_mult=[1, 2, 2, 2], num_heads=4,
                           num_head_channels=64, attention_resolutions="16", dropout=0.1).to(device)
    ck = torch.load(OUT / "weights" / method / f"{method}_cifar10_weights_step_400000.pt", map_location=device, weights_only=True)
    state = ck["ema_model"]
    try:
        net.load_state_dict(state)
    except RuntimeError:
        net.load_state_dict({k[7:]: v for k, v in state.items()})
    return net.eval()


def load_mnist(method):
    net = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1).to(device)
    net.load_state_dict(torch.load(ROOT / f"results/e2/ckpt/{method}_s0.pt", map_location=device, weights_only=True))
    return net.eval()


abar = torch.cumprod(1 - torch.linspace(1e-4, 0.02, T, dtype=torch.float64), 0).float().to(device)


@torch.no_grad()
def euler(net, x, k):
    for i in range(k):
        x = x + net(torch.full((x.shape[0],), i / k, device=device), x) / k
    return x


@torch.no_grad()
def ddim(net, x, k):
    """Same grid as E1: k calls at equal DDPM steps from 999 to 0, then a last jump to pure data."""
    idx = torch.linspace(T - 1, 0, k).round().long().tolist()
    for j, i in enumerate(idx):
        ab = abar[i]
        e = net(torch.full((x.shape[0],), i / (T - 1), device=device), x)
        x1 = (x - (1 - ab).sqrt() * e) / ab.sqrt()
        ab2 = abar[idx[j + 1]] if j + 1 < len(idx) else torch.tensor(1.0, device=device)
        x = ab2.sqrt() * x1 + (1 - ab2).sqrt() * e
    return x


class Counted(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net, self.nfe = net, 0

    def forward(self, t, x):
        self.nfe += 1
        return self.net(t.expand(x.shape[0]) if t.ndim == 0 else t, x)


@torch.no_grad()
def dopri(net, x, tol):
    c = Counted(net)
    out = odeint(c, x, torch.tensor([0.0, 1.0], device=device), rtol=tol, atol=tol, method="dopri5")[-1]
    return out, c.nfe


def to_uint8(x):
    return (x * 127.5 + 128).clip(0, 255).to(torch.uint8)


def real_images(data):
    if data == "cifar":
        ds = datasets.CIFAR10(ROOT / "conditional-flow-matching-main/examples/images/cifar10/data", train=True, download=False)
        return torch.tensor(ds.data).permute(0, 3, 1, 2).float() / 127.5 - 1
    ds = datasets.MNIST(ROOT / "data", train=True, download=False)
    return ds.data[:, None].float() / 127.5 - 1


class FID:
    def __init__(self, data, real):
        self.data, self.mode = data, ("legacy_tensorflow" if data == "cifar" else "clean")
        self.model = fid.build_feature_extractor(self.mode, device, use_dataparallel=False)
        self.real = real
        if data == "cifar":
            # clean-fid's published stats; its own download helper wants to write inside site-packages
            s = np.load(OUT / "stats" / "cifar10_legacy_tensorflow_train_32.npz")
            self.mu, self.sigma = s["mu"], s["sigma"]
        else:
            f = self.features(lambda i, b: self.real[i:i + b].to(device), len(real))
            self.mu, self.sigma = f.mean(0), np.cov(f, rowvar=False)

    def features(self, fn, n, batch=500):
        """fn(start, batch) returns images in [-1, 1]."""
        pos = [0]

        def G(_z):
            x = fn(pos[0], batch)
            pos[0] += batch
            img = to_uint8(x)
            return img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img

        return fid.get_model_features(G, self.model, mode=self.mode, num_gen=n, batch_size=batch, device=device, verbose=False)

    def score(self, fn, n):
        f = self.features(fn, n)
        return float(fid.frechet_distance(f.mean(0), np.cov(f, rowvar=False), self.mu, self.sigma))


def pair_costs(real, batch, rng_seed=0, rounds=20):
    """Mean squared distance per image between noise and data, for random pairs and OT pairs in a batch."""
    g = torch.Generator().manual_seed(rng_seed)
    rnd, opt = [], []
    for _ in range(rounds):
        i = torch.randperm(len(real), generator=g)[:batch]
        x1 = real[i].flatten(1)
        x0 = torch.randn(x1.shape, generator=g)
        M = torch.cdist(x0, x1).pow(2).numpy().astype(np.float64)
        rnd.append(float(np.trace(M) / batch))
        opt.append(float(ot.emd2(ot.unif(batch), ot.unif(batch), M, numItermax=1_000_000)))
    return {"batch": batch, "random": float(np.mean(rnd)), "ot": float(np.mean(opt)), "ratio": float(np.mean(opt) / np.mean(rnd))}


@torch.no_grad()
def path_ratio(net, shape, n=256, k=100):
    x = torch.randn(n, *shape, device=device)
    start, length = x.clone(), torch.zeros(n, device=device)
    for i in range(k):
        v = net(torch.full((n,), i / k, device=device), x)
        length += v.flatten(1).norm(dim=1) / k
        x = x + v / k
    return float((length / (x - start).flatten(1).norm(dim=1)).mean())


@torch.no_grad()
def one_step(net, real, n=256):
    x = torch.randn(n, *real.shape[1:], device=device)
    y = euler(net, x, 1).cpu()
    mean = real.mean(0, keepdim=True)
    d = real.shape[1] * real.shape[2] * real.shape[3]
    to_mean = (y - mean).flatten(1).pow(2).sum(1).div(d).sqrt().mean().item()
    near = torch.cdist(y.flatten(1), real[:20000].flatten(1)).min(1).values.pow(2).div(d).sqrt().mean().item()
    spread = y.std(0).pow(2).mean().sqrt().item()
    return {"rms_to_data_mean": to_mean, "rms_to_nearest_train_image": near, "rms_spread_across_noise": spread}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["mnist", "cifar"], required=True)
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--only-dopri", action="store_true", help="only the adaptive solver, for a larger n")
    args = p.parse_args()
    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    real = real_images(args.data)
    methods = {"mnist": ["icfm", "otcfm", "ddpm"], "cifar": ["icfm", "otcfm"]}[args.data]
    load = load_mnist if args.data == "mnist" else load_cifar
    meter = FID(args.data, real)
    name = OUT / f"{args.data}_eval{'_n' + str(args.n) if args.only_dopri else ''}.json"
    result = {"data": args.data, "n": args.n, "tol": args.tol, "fid_mode": meter.mode, "by_method": {}}

    if not args.only_dopri:
        g = torch.Generator().manual_seed(1)
        perm = torch.randperm(len(real), generator=g)[:args.n]
        result["fid_real_vs_real"] = meter.score(lambda i, b: real[perm[i:i + b]].to(device), args.n)
        print("real vs real", result["fid_real_vs_real"], flush=True)
        result["pairs"] = pair_costs(real, 128)
        print("pairs", result["pairs"], flush=True)

    grids = {"real": to_uint8(real[:32]).numpy(), "mean": to_uint8(real.mean(0, keepdim=True)).numpy()}
    fixed = torch.randn(32, *real.shape[1:], generator=torch.Generator().manual_seed(2)).to(device)
    for m in methods:
        net = load(m)
        step = ddim if m == "ddpm" else euler
        rec = {"fid": {}, "seconds": {}}
        for k in ([] if args.only_dopri else NFES):
            t0 = time.time()
            torch.manual_seed(100)
            rec["fid"][str(k)] = meter.score(lambda i, b: step(net, torch.randn(b, *real.shape[1:], device=device), k), args.n)
            rec["seconds"][str(k)] = time.time() - t0
            grids[f"{m}_{k}"] = to_uint8(step(net, fixed, k)).cpu().numpy()
            print(m, k, round(rec["fid"][str(k)], 2), f"{rec['seconds'][str(k)]:.0f}s", flush=True)
        if m != "ddpm":
            nfes = []

            def gen(i, b):
                y, c = dopri(net, torch.randn(b, *real.shape[1:], device=device), args.tol)
                nfes.append(c)
                return y

            t0 = time.time()
            torch.manual_seed(100)
            rec["fid"]["dopri5"] = meter.score(gen, args.n)
            rec["dopri5_mean_nfe"] = float(np.mean(nfes))
            rec["seconds"]["dopri5"] = time.time() - t0
            grids[f"{m}_dopri5"] = to_uint8(dopri(net, fixed, args.tol)[0]).cpu().numpy()
            print(m, "dopri5", round(rec["fid"]["dopri5"], 2), "mean NFE", rec["dopri5_mean_nfe"], f"{rec['seconds']['dopri5']:.0f}s", flush=True)
            if not args.only_dopri:
                rec["one_step"] = one_step(net, real)
                rec["path_ratio"] = path_ratio(net, real.shape[1:])
                print(m, rec["one_step"], "path ratio", rec["path_ratio"], flush=True)
                # a trajectory strip for the figures
                x, frames = fixed[:8].clone(), []
                for i in range(100):
                    if i in [0, 25, 50, 75]:
                        frames.append(to_uint8(x).cpu().numpy())
                    x = x + net(torch.full((8,), i / 100, device=device), x) / 100
                frames.append(to_uint8(x).cpu().numpy())
                grids[f"{m}_traj"] = np.stack(frames)
        result["by_method"][m] = rec
        name.write_text(json.dumps(result, indent=2))
        del net
    if not args.only_dopri:
        np.savez_compressed(OUT / f"{args.data}_samples.npz", **grids)
    print("saved", name)


if __name__ == "__main__":
    main()

"""E2 · 图像实验的评估脚本：把训练好的模型变成 Part V（第 25–27 页）要用的全部数字和样本。

    python -u code/e4_cifar/eval_images.py --data mnist --n 10000

（文件放在 e4_cifar/ 是历史原因。现在只做 MNIST；CIFAR 分支保留，当前计划不用。）

要完成的任务
    任务 1  real-vs-real FID：两份真实图之间的 FID，也就是这个样本量下 FID 能到的下限（图 F11 的虚线）
    任务 2  配对代价：batch 128 内 OT 配对比随机配对短多少，检验假设 H3（第 27 页）
    任务 3  FID vs NFE：三个模型在 NFE 1, 2, 5, 10, 20, 50, 100 下各生成 n 张图算 FID（图 F11，第 26 页）
    任务 4  同一批噪声的样本网格：肉眼比较少步时的差别（图 F9，第 25 页）
    任务 5  dopri5 自适应求解器：两个流模型“步数足够多”时的 FID 和平均调用次数（第 26 页）
    任务 6  一步落点：I-CFM 走一步是否落到数据均值，检验第 16 页在 784 维下的预测（第 25、27 页）
    任务 7  路径长度比：路径长度 / 首尾直线距离，和 2D 的 1.35 / 1.001 对比（第 27 页）
    任务 8  生成轨迹帧：t = 0, .25, .5, .75, 1 的中间图（图 F9，第 25 页）
    任务 9  少步时的像素范围：有多少像素超出 [-1, 1]，解释 DDIM 在 1–2 步的白点（第 25 页）

模型
    mnist : results/e2/ckpt/{icfm,otcfm,ddpm}_s0.pt，由 code/e2_mnist/train_mnist.py 训练
    cifar : 官方 EMA 权重（400k 步），results/e4/weights/{icfm,otcfm}/

FID 的口径
    clean-fid。MNIST 用 clean 模式，参考集是 6 万张训练图，特征来自 ImageNet 上的 Inception，所以只看趋势，
    不能和文献里的 CIFAR 数字比。CIFAR 用 legacy_tensorflow 模式和发布的训练集统计量。

中断后再跑
    结果每算完一项就写入 results/e4/{data}_eval.json，再跑会跳过 json 里已有的 FID（每个 FID 前都重设种子，
    所以结果与顺序无关）。样本网格、一步落点、路径比和轨迹很快，每次重算。--fresh 从头开始。

输出 results/e4/{data}_eval.json 和 {data}_samples.npz（uint8 图，供 slides/make_image_figs.py 作图）。
"""
import argparse
import json
import os
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


# ---------- 载入模型 ----------
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


def load_mnist(method, seed=0):
    net = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1).to(device)
    net.load_state_dict(torch.load(ROOT / f"results/e2/ckpt/{method}_s{seed}.pt", map_location=device, weights_only=True))
    return net.eval()


abar = torch.cumprod(1 - torch.linspace(1e-4, 0.02, T, dtype=torch.float64), 0).float().to(device)


# ---------- 三种采样器：都把噪声变成图，NFE = 调用网络的次数 ----------
@torch.no_grad()
def euler(net, x, k):
    """流模型：t 从 0 到 1 走 k 个等长 Euler 步，x <- x + v(t, x) / k。"""
    for i in range(k):
        x = x + net(torch.full((x.shape[0],), i / k, device=device), x) / k
    return x


@torch.no_grad()
def ddim(net, x, k):
    """DDPM 模型的确定性采样，与 E1 同一网格：在 DDPM 的 999 到 0 步之间等距取 k 次，最后一跳到纯数据。"""
    idx = torch.linspace(T - 1, 0, k).round().long().tolist()
    for j, i in enumerate(idx):
        ab = abar[i]
        e = net(torch.full((x.shape[0],), i / (T - 1), device=device), x)
        x1 = (x - (1 - ab).sqrt() * e) / ab.sqrt()
        ab2 = abar[idx[j + 1]] if j + 1 < len(idx) else torch.tensor(1.0, device=device)
        x = ab2.sqrt() * x1 + (1 - ab2).sqrt() * e
    return x


class Counted(torch.nn.Module):
    """数 dopri5 实际调用了多少次网络。"""

    def __init__(self, net):
        super().__init__()
        self.net, self.nfe = net, 0

    def forward(self, t, x):
        self.nfe += 1
        return self.net(t.expand(x.shape[0]) if t.ndim == 0 else t, x)


@torch.no_grad()
def dopri(net, x, tol):
    """任务 5：自适应步长，误差不超过 tol，步数由求解器决定。"""
    c = Counted(net)
    out = odeint(c, x, torch.tensor([0.0, 1.0], device=device), rtol=tol, atol=tol, method="dopri5")[-1]
    return out, c.nfe


# ---------- 数据和 FID ----------
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

    def features(self, fn, n, batch=250):
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


# ---------- 任务 2、6、7 ----------
def pair_costs(real, batch, rng_seed=0, rounds=20):
    """任务 2：每张图的噪声到数据的平均平方距离，随机配对 vs batch 内 OT 配对。比值接近 1 说明配对几乎没变。"""
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
    """任务 7：100 步 Euler 的路径长度 / 首尾直线距离。等于 1 就是直线。"""
    x = torch.randn(n, *shape, device=device)
    start, length = x.clone(), torch.zeros(n, device=device)
    for i in range(k):
        v = net(torch.full((n,), i / k, device=device), x)
        length += v.flatten(1).norm(dim=1) / k
        x = x + v / k
    return float((length / (x - start).flatten(1).norm(dim=1)).mean())


@torch.no_grad()
def one_step(net, real, n=256):
    """任务 6：走一步后，离数据均值多远、离最近的训练图多远、不同噪声之间差多少（每像素 RMS，像素范围 [-1, 1]）。"""
    x = torch.randn(n, *real.shape[1:], device=device)
    y = euler(net, x, 1).cpu()
    mean = real.mean(0, keepdim=True)
    d = real.shape[1] * real.shape[2] * real.shape[3]
    to_mean = (y - mean).flatten(1).pow(2).sum(1).div(d).sqrt().mean().item()
    near = torch.cdist(y.flatten(1), real[:20000].flatten(1)).min(1).values.pow(2).div(d).sqrt().mean().item()
    spread = y.std(0).pow(2).mean().sqrt().item()
    return {"rms_to_data_mean": to_mean, "rms_to_nearest_train_image": near, "rms_spread_across_noise": spread}


@torch.no_grad()
def out_of_range(net, step, shape, n=256, ks=(1, 2, 5)):
    """任务 9：超出 [-1, 1] 的像素比例和最大绝对值。这些像素在存图和算 FID 前会被截断。"""
    out = {}
    for k in ks:
        torch.manual_seed(202)
        y = step(net, torch.randn(n, *shape, device=device), k)
        out[str(k)] = {"share_outside": (y.abs() > 1).float().mean().item(), "max_abs": y.abs().max().item()}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["mnist", "cifar"], required=True)
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--only-dopri", action="store_true", help="only the adaptive solver, for a larger n")
    p.add_argument("--fresh", action="store_true", help="ignore results already saved in the json")
    args = p.parse_args()
    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    real = real_images(args.data)
    methods = {"mnist": ["icfm", "otcfm", "ddpm"], "cifar": ["icfm", "otcfm"]}[args.data]
    load = load_mnist if args.data == "mnist" else load_cifar
    meter = FID(args.data, real)
    name = OUT / f"{args.data}_eval{'_n' + str(args.n) if args.only_dopri else ''}.json"
    result = {"data": args.data, "n": args.n, "tol": args.tol, "fid_mode": meter.mode, "by_method": {}}
    if name.exists() and not args.fresh:
        old = json.loads(name.read_text())
        if (old["n"], old["tol"], old["fid_mode"]) == (args.n, args.tol, meter.mode):
            result = old
            print("resuming from", name, flush=True)

    def save():
        tmp = name.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2))
        os.replace(tmp, name)

    if not args.only_dopri:
        # 任务 1：FID 的下限
        if "fid_real_vs_real" not in result:
            g = torch.Generator().manual_seed(1)
            perm = torch.randperm(len(real), generator=g)[:args.n]
            result["fid_real_vs_real"] = meter.score(lambda i, b: real[perm[i:i + b]].to(device), args.n)
            save()
        print("real vs real", result["fid_real_vs_real"], flush=True)
        # 任务 2：配对代价
        if "pairs" not in result:
            result["pairs"] = pair_costs(real, 128)
            save()
        print("pairs", result["pairs"], flush=True)

    # 任务 4：所有模型、所有 NFE 用同一批 32 个噪声，这样网格里同一格的差别只来自模型和步数
    grids = {"real": to_uint8(real[:32]).numpy(), "mean": to_uint8(real.mean(0, keepdim=True)).numpy()}
    fixed = torch.randn(32, *real.shape[1:], generator=torch.Generator().manual_seed(2)).to(device)
    for m in methods:
        net = load(m)
        step = ddim if m == "ddpm" else euler
        rec = result["by_method"].setdefault(m, {"fid": {}, "seconds": {}})
        # 任务 3：FID vs NFE
        for k in ([] if args.only_dopri else NFES):
            if str(k) not in rec["fid"]:
                t0 = time.time()
                torch.manual_seed(100)
                rec["fid"][str(k)] = meter.score(lambda i, b: step(net, torch.randn(b, *real.shape[1:], device=device), k), args.n)
                rec["seconds"][str(k)] = time.time() - t0
                save()
            grids[f"{m}_{k}"] = to_uint8(step(net, fixed, k)).cpu().numpy()
            print(m, k, round(rec["fid"][str(k)], 2), f"{rec['seconds'][str(k)]:.0f}s", flush=True)
        if not args.only_dopri:
            rec["out_of_range"] = out_of_range(net, step, real.shape[1:])
            save()
            print(m, "out of range", rec["out_of_range"], flush=True)
        if m != "ddpm":
            # 任务 5：自适应求解器
            if "dopri5" not in rec["fid"]:
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
                save()
            grids[f"{m}_dopri5"] = to_uint8(dopri(net, fixed, args.tol)[0]).cpu().numpy()
            print(m, "dopri5", round(rec["fid"]["dopri5"], 2), "mean NFE", rec["dopri5_mean_nfe"], f"{rec['seconds']['dopri5']:.0f}s", flush=True)
            if not args.only_dopri:
                # 任务 6、7
                torch.manual_seed(200)
                rec["one_step"] = one_step(net, real)
                torch.manual_seed(201)
                rec["path_ratio"] = path_ratio(net, real.shape[1:])
                save()
                print(m, rec["one_step"], "path ratio", rec["path_ratio"], flush=True)
                # 任务 8：轨迹帧
                x, frames = fixed[:8].clone(), []
                with torch.no_grad():  # the graph of 100 steps holds about 11 GB of GPU memory and starves everything after it
                    for i in range(100):
                        if i in [0, 25, 50, 75]:
                            frames.append(to_uint8(x).cpu().numpy())
                        x = x + net(torch.full((8,), i / 100, device=device), x) / 100
                    frames.append(to_uint8(x).cpu().numpy())
                grids[f"{m}_traj"] = np.stack(frames)
        del net
        if not args.only_dopri:
            np.savez_compressed(OUT / f"{args.data}_samples.npz", **grids)
    print("saved", name, flush=True)


if __name__ == "__main__":
    main()

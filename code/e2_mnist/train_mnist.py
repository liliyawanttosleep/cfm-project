"""E2 · MNIST 训练脚本，演示级，参照 examples/images/mnist_example.ipynb。

要完成的任务
    任务 1  训练第 23–27 页用的三个模型，网络和预算完全相同，只有配对或回归目标不同
    任务 2  每 5 个 epoch 存一次续跑点，GPU 驱动重置后可以接着训练
    任务 3  输出 results/e2/ckpt/{method}_s{seed}.pt 给 code/e4_cifar/eval_images.py 评估，
            输出 results/e2/{method}_s{seed}_train.json 给第 27 页的 loss 曲线

理论目标：同样的五行训练循环从 2 维搬到 784 维，只换数据和网络。

Same U-Net (32 channels, 1 residual block), Adam with its default lr, batch 128, 50 epochs, no EMA.
    icfm  : independent pairs, target x1 - x0
    otcfm : minibatch OT pairs, target x1 - x0
    ddpm  : linear beta schedule, T = 1000, target eps. Sampled with DDIM in code/e4_cifar/eval_images.py.

    python code/e2_mnist/train_mnist.py --method icfm --seed 0

Final report 的变体（docs/FINAL_REPORT_PLAN.md；默认参数和上面的模型完全相同，随机数顺序也不变）
    C3  --ot-batch 512 / 2048（只用于 otcfm）：一次取 ot-batch 张图和同样多的噪声求 OT 计划，再按计划抽出配对，
        切成 batch 128 的小块依次训练。梯度步的 batch 不变，每个 epoch 的步数基本不变（末尾不够一组的丢掉）。
    C4  --t-dist logitnormal：t = sigmoid(z)，z ~ N(0, 1)（SD3 的默认设置）
        --t-dist kappa：t 来自 (1 − w)·U[0, 1] + w·κ(t)/∫κ，w = --t-mix（默认 0.5），κ(t) 取自 C2 的
        results/c2_solvers/{method}_s0.json（可用 --kappa-from 指定）
    C5  --reflow：在 code/c5_reflow/make_pairs.py 生成的配对 (x0, x1_hat) 上训练 2-rectified flow，目标 x1_hat − x0。
        默认从同一方法、同一种子的原模型权重开始微调（新的 Adam）；--from-scratch 从随机初始化开始。
    变体的文件名带后缀，例如 otcfm_s0_otb2048.pt、icfm_s0_tlogitnormal.pt、icfm_s0_reflow.pt，不会覆盖原模型。
    训练默认限速（--duty 0.5，code/common/throttle.py），限速不改变随机数。

    python code/e2_mnist/train_mnist.py --method otcfm --seed 0 --ot-batch 2048
    python code/e2_mnist/train_mnist.py --method icfm --seed 0 --t-dist kappa
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
from torchcfm.models.unet import UNetModel
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "e2"
T = 1000
sys.path.insert(0, str(ROOT / "code"))
from common.throttle import Throttle  # noqa: E402


def make_t_sampler(args, device):
    """C4：返回 None（均匀，交给 torchcfm 自己抽，随机数顺序和原模型相同），或一个 n -> t 的函数。"""
    if args.t_dist == "uniform":
        return None
    if args.t_dist == "logitnormal":
        return lambda n: torch.sigmoid(torch.randn(n, device=device))
    src = Path(args.kappa_from) if args.kappa_from else ROOT / f"results/c2_solvers/{args.method}_s0.json"
    k = json.loads(src.read_text())["kappa"]
    t, kappa = np.array(k["t"]), np.array(k["kappa"])
    area = np.sum(0.5 * (kappa[1:] + kappa[:-1]) * np.diff(t))
    dens = (1 - args.t_mix) + args.t_mix * kappa / area  # 混合分布的密度，在 [0, 1] 上积分为 1
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(t))])
    cdf /= cdf[-1]
    cdf_t = torch.tensor(cdf, dtype=torch.float32, device=device)
    grid_t = torch.tensor(t, dtype=torch.float32, device=device)

    def sample(n):  # 逆 CDF，分段线性插值
        u = torch.rand(n, device=device)
        j = torch.searchsorted(cdf_t, u).clamp(1, len(cdf_t) - 1)
        c0, c1, t0, t1 = cdf_t[j - 1], cdf_t[j], grid_t[j - 1], grid_t[j]
        return t0 + (u - c0) / (c1 - c0).clamp_min(1e-12) * (t1 - t0)

    return sample


def run_name(args):
    name = f"{args.method}_s{args.seed}"
    if args.ot_batch != args.batch:
        name += f"_otb{args.ot_batch}"
    if args.t_dist != "uniform":
        name += f"_t{args.t_dist}" + (f"{args.t_mix:g}" if args.t_dist == "kappa" and args.t_mix != 0.5 else "")
    if args.reflow:
        name += "_reflow" + ("_scratch" if args.from_scratch else "")
    return name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["icfm", "otcfm", "ddpm"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--ot-batch", type=int, default=None, help="C3：求 OT 计划用的样本数（batch 的整数倍），默认等于 batch")
    p.add_argument("--t-dist", choices=["uniform", "logitnormal", "kappa"], default="uniform", help="C4：训练时 t 的分布")
    p.add_argument("--t-mix", type=float, default=0.5, help="--t-dist kappa 时 κ 部分的权重")
    p.add_argument("--kappa-from", default=None, help="κ(t) 的来源，默认 results/c2_solvers/{method}_s0.json")
    p.add_argument("--reflow", action="store_true", help="C5：在 results/c5_reflow/pairs/{method}_s{seed}/ 的配对上训练")
    p.add_argument("--from-scratch", action="store_true", help="C5：不从原模型权重开始")
    p.add_argument("--duty", type=float, default=0.5, help="显卡忙碌时间的比例，1 = 不限速")
    p.add_argument("--min-free-gb", type=float, default=1.5)
    args = p.parse_args()
    if args.reflow and (args.method == "ddpm" or args.ot_batch not in (None, args.batch)):
        p.error("--reflow only applies to icfm / otcfm without --ot-batch")
    args.ot_batch = args.ot_batch or args.batch
    if args.ot_batch != args.batch and (args.method != "otcfm" or args.ot_batch % args.batch):
        p.error("--ot-batch only applies to otcfm and must be a multiple of --batch")
    if args.t_dist != "uniform" and args.method == "ddpm":
        p.error("--t-dist only applies to the flow models")
    th = Throttle(args.duty, min_free_gb=args.min_free_gb)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = datasets.MNIST(ROOT / "data", train=True, download=False,
                          transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch, shuffle=True, drop_last=True)
    model = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1).to(device)
    if args.reflow:
        chunks = sorted((ROOT / f"results/c5_reflow/pairs/{args.method}_s{args.seed}").glob("chunk_*.pt"))
        pairs = [torch.load(c, weights_only=True) for c in chunks]
        P0, P1 = torch.cat([q["x0"] for q in pairs]), torch.cat([q["x1"] for q in pairs])
        loader = torch.utils.data.DataLoader(TensorDataset(P0, P1), batch_size=args.batch, shuffle=True, drop_last=True)
        print(f"reflow pairs: {len(P0):,} from {len(chunks)} chunks", flush=True)
        if not args.from_scratch:
            model.load_state_dict(torch.load(OUT / "ckpt" / f"{args.method}_s{args.seed}.pt", map_location=device, weights_only=True))
    opt = torch.optim.Adam(model.parameters())
    n_params = sum(q.numel() for q in model.parameters())
    print(f"{args.method}: {n_params:,} parameters, {len(loader)} steps per epoch", flush=True)

    if args.method == "ddpm":
        abar = torch.cumprod(1 - torch.linspace(1e-4, 0.02, T, dtype=torch.float64), 0).float().to(device)
    else:
        fm = ConditionalFlowMatcher(sigma=0.0) if args.method == "icfm" else ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
        plain = ConditionalFlowMatcher(sigma=0.0)  # C3：配对已经在大 batch 上做好，小块里不再求 OT
        t_sampler = make_t_sampler(args, device)
    group = args.ot_batch // args.batch

    (OUT / "ckpt").mkdir(parents=True, exist_ok=True)
    name = run_name(args)
    last = OUT / "ckpt" / f"{name}_last.pt"
    log, t0, start, ot_seconds = [], time.time(), 0, 0.0
    if last.exists():  # a GPU driver reset killed an earlier run: carry on from the last saved epoch
        ck = torch.load(last, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        log, start, ot_seconds = ck["log"], ck["epoch"], ck.get("ot_seconds", 0.0)
        print(f"resumed after epoch {start}", flush=True)

    def train_step(t, xt, target):
        loss = ((model(t, xt) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        th.step()
        return loss.item()

    def batches():
        """group = 1 时就是原来的 loader；group > 1 时（C3）把 group 个 batch 拼成一个大 batch。"""
        buf = []
        for x1, x1_hat in loader:
            if args.reflow:  # C5：loader 给出 (x0, x1_hat)，整对传下去
                yield x1, x1_hat
                continue
            buf.append(x1)
            if len(buf) == group:
                yield torch.cat(buf)
                buf = []

    for epoch in range(start, args.epochs):
        total, steps = 0.0, 0
        th.reset()
        for x1 in batches():
            if args.reflow:
                x0, x1 = x1[0].to(device), x1[1].to(device)
            else:
                x1 = x1.to(device)
                x0 = torch.randn_like(x1)
            if args.method == "ddpm":
                i = torch.randint(0, T, (x1.shape[0],), device=device)
                ab = abar[i][:, None, None, None]
                xt = ab.sqrt() * x1 + (1 - ab).sqrt() * x0
                total += train_step(i.float() / (T - 1), xt, x0)
                steps += 1
            elif args.reflow:  # C5：配对固定，路径是 x0 到 x1_hat 的直线
                t = None if t_sampler is None else t_sampler(x1.shape[0])
                t, xt, ut = plain.sample_location_and_conditional_flow(x0, x1, t=t)
                total += train_step(t, xt, ut)
                steps += 1
            elif group == 1:
                t = None if t_sampler is None else t_sampler(x1.shape[0])
                t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1, t=t)
                total += train_step(t, xt, ut)
                steps += 1
            else:
                s0 = time.time()
                x0, x1 = fm.ot_sampler.sample_plan(x0, x1)  # 和 torchcfm 相同：按 OT 计划有放回地抽配对
                ot_seconds += time.time() - s0
                for c0, c1 in zip(x0.split(args.batch), x1.split(args.batch)):
                    t = None if t_sampler is None else t_sampler(c1.shape[0])
                    t, xt, ut = plain.sample_location_and_conditional_flow(c0, c1, t=t)
                    total += train_step(t, xt, ut)
                    steps += 1
        log.append(total / steps)
        print(f"epoch {epoch + 1}/{args.epochs} loss {log[-1]:.4f} steps {steps} elapsed {time.time() - t0:.0f}s"
              + (f" (OT {ot_seconds:.0f}s)" if group > 1 else ""), flush=True)
        if (epoch + 1) % 5 == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "log": log, "epoch": epoch + 1,
                        "ot_seconds": ot_seconds}, last)

    torch.save(model.state_dict(), OUT / "ckpt" / f"{name}.pt")
    (OUT / f"{name}_train.json").write_text(json.dumps({"config": vars(args), "params": n_params, "seconds": time.time() - t0,
                                                         "ot_seconds": ot_seconds, "throttled": args.duty < 1, "loss": log}, indent=2))


if __name__ == "__main__":
    main()

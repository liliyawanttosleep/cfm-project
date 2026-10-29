"""E1 · Diffusion 线（成员 B）。最小的 DDPM 实现（Ho et al. 2020）+ DDIM 采样（Song et al. 2021）。

前向过程：q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I)，线性 beta schedule，T = 1000。
训练目标：预测噪声 eps，loss = E||eps - eps_theta(x_t, t)||^2（等价于 denoising score matching，
    score = -eps / sqrt(1 - abar_t)）。
采样：
    ddpm : 祖先采样，随机，固定 NFE = T（--nfes 被忽略）
    ddim : eta = 0 的确定性采样，可以只用很少的步数；相当于 VP-SDE 的 probability flow ODE 的离散化

两种 method 训练过程完全相同，只是采样器不同。

用法（在 cfm_project 根目录下）：
    python code/e1_2d/train_ddpm.py --method ddim --data moons --seed 0
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from common.data_2d import sample
from common.experiment import base_parser, evaluate, log_progress, save_checkpoint, setup
from common.models import MLP


class Schedule:
    def __init__(self, T, schedule, device):
        self.T = T
        if schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float64)
        else:  # cosine，Nichol & Dhariwal 2021
            s = 0.008
            f = torch.cos((torch.arange(T + 1, dtype=torch.float64) / T + s) / (1 + s) * torch.pi / 2) ** 2
            betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
        alphas = 1 - betas
        self.betas = betas.float().to(device)
        self.alphas = alphas.float().to(device)
        self.abar = torch.cumprod(alphas, 0).float().to(device)

    def t_input(self, t_idx):
        """把整数时间步 0..T-1 映射到 [0, 1]，与 FM 用同一个时间编码。"""
        return t_idx.float() / (self.T - 1)


@torch.no_grad()
def ddpm_sample(model, sch, x):
    traj = [x]
    for i in reversed(range(sch.T)):
        t = torch.full((x.shape[0],), i, device=x.device)
        eps = model(x, sch.t_input(t))
        a, ab, b = sch.alphas[i], sch.abar[i], sch.betas[i]
        mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
        if i > 0:
            var = b * (1 - sch.abar[i - 1]) / (1 - ab)  # 后验方差 beta_tilde
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = mean
        traj.append(x)
    return torch.stack(traj)


@torch.no_grad()
def ddim_sample(model, sch, x, nfe):
    steps = torch.linspace(sch.T - 1, 0, nfe, device=x.device).round().long().tolist()
    traj = [x]
    for k, i in enumerate(steps):
        t = torch.full((x.shape[0],), i, device=x.device)
        eps = model(x, sch.t_input(t))
        ab = sch.abar[i]
        x0_pred = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        ab_prev = sch.abar[steps[k + 1]] if k + 1 < len(steps) else torch.tensor(1.0, device=x.device)
        x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps
        traj.append(x)
    return torch.stack(traj)


def main():
    p = base_parser(["ddim", "ddpm"], default_nfes=[1, 2, 5, 10, 20, 50, 100])
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    args = p.parse_args()
    if args.method == "ddpm":
        args.nfes = [args.T]
    rng, device = setup(args)
    name = f"{args.method}_{args.data}_s{args.seed}" + ("" if args.schedule == "linear" else f"_{args.schedule}")
    print(f"[DDPM] {name} on {device}")

    sch = Schedule(args.T, args.schedule, device)
    model = MLP(width=args.width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    t0 = time.time()
    for step in range(1, args.iters + 1):
        x_data = sample(args.data, args.batch, rng).to(device)
        t = torch.randint(0, sch.T, (args.batch,), device=device)
        eps = torch.randn_like(x_data)
        ab = sch.abar[t][:, None]
        xt = ab.sqrt() * x_data + (1 - ab).sqrt() * eps
        loss = ((model(xt, sch.t_input(t)) - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        log_progress(step, args.iters, loss.item(), t0)
    train_seconds = time.time() - t0

    model.eval()
    save_checkpoint(args, name, model)
    if args.method == "ddpm":
        sampler = lambda x0, nfe: ddpm_sample(model, sch, x0.to(device))
    else:
        sampler = lambda x0, nfe: ddim_sample(model, sch, x0.to(device), nfe)
    evaluate(args, name, sampler, train_seconds, deterministic=args.method == "ddim")


if __name__ == "__main__":
    main()

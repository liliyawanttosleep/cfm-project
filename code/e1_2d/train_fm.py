"""E1 · Flow Matching 线（成员 A）。

要完成的任务
    任务 1  在 2D 数据上训练速度场 v_theta(x, t)，目标是条件速度 x1 - x0（第 19 页）
    任务 2  用 Euler 法从噪声走到数据，在 NFE 1–100 下采样
    任务 3  算 W2、MMD、直线度，存 json / png / 权重，供 summarize.py 汇总成第 19 页的图

理论目标：检验“配对决定路径，路径决定步数”。OT-CFM 只比 I-CFM 多一步 batch 内 OT 配对，
预期 OT-CFM 一步就接近数据，而 I-CFM 一步落到数据均值。

从 N(0, I) 到 2D 数据分布，训练速度场 v_theta(x, t)，用 Euler 法解 ODE 采样。
    icfm      : I-CFM，x0 与 x1 独立配对，直线路径；sigma=0 时等价于 Rectified Flow（1-RF）
    otcfm     : OT-CFM，在每个 minibatch 内用精确 OT 配对 x0 与 x1（Tong et al. 2024）
    fm_lipman : Lipman et al. 2023 原始 FM，以 x1 为条件的高斯路径

用法（在 cfm_project 根目录下）：
    python code/e1_2d/train_fm.py --method otcfm --data moons --seed 0
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
)

from common.data_2d import sample
from common.experiment import base_parser, evaluate, log_progress, save_checkpoint, setup
from common.models import MLP

MATCHERS = {
    "icfm": ConditionalFlowMatcher,
    "otcfm": ExactOptimalTransportConditionalFlowMatcher,
    "fm_lipman": TargetConditionalFlowMatcher,
}


@torch.no_grad()
def euler_sample(model, x0, nfe):
    """dx/dt = v(x, t)，t 从 0（噪声）走到 1（数据），等步长 Euler，返回整条轨迹。"""
    x, dt, traj = x0, 1.0 / nfe, [x0]
    for k in range(nfe):
        t = torch.full((x.shape[0],), k * dt, device=x.device)
        x = x + dt * model(x, t)
        traj.append(x)
    return torch.stack(traj)


def main():
    p = base_parser(list(MATCHERS), default_nfes=[1, 2, 5, 10, 20, 50, 100])
    p.add_argument("--sigma", type=float, default=0.0, help="条件路径的噪声宽度，0 = 确定性直线路径")
    args = p.parse_args()
    rng, device = setup(args)
    name = f"{args.method}_{args.data}_s{args.seed}"
    print(f"[FM] {name} on {device}")

    model = MLP(width=args.width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    matcher = MATCHERS[args.method](sigma=args.sigma)

    t0 = time.time()
    for step in range(1, args.iters + 1):
        x1 = sample(args.data, args.batch, rng).to(device)
        x0 = torch.randn_like(x1)
        # t ~ U(0,1)，xt 在 x0 与 x1 的条件路径上，ut 是条件速度（直线路径下 ut = x1 - x0）
        t, xt, ut = matcher.sample_location_and_conditional_flow(x0, x1)
        loss = ((model(xt, t) - ut) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        log_progress(step, args.iters, loss.item(), t0)
    train_seconds = time.time() - t0

    model.eval()
    save_checkpoint(args, name, model)
    evaluate(args, name, lambda x0, nfe: euler_sample(model, x0.to(device), nfe), train_seconds)


if __name__ == "__main__":
    main()

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
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
from torchcfm.models.unet import UNetModel
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "e2"
T = 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["icfm", "otcfm", "ddpm"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=128)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = datasets.MNIST(ROOT / "data", train=True, download=False,
                          transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch, shuffle=True, drop_last=True)
    model = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1).to(device)
    opt = torch.optim.Adam(model.parameters())
    n_params = sum(q.numel() for q in model.parameters())
    print(f"{args.method}: {n_params:,} parameters, {len(loader)} steps per epoch", flush=True)

    if args.method == "ddpm":
        abar = torch.cumprod(1 - torch.linspace(1e-4, 0.02, T, dtype=torch.float64), 0).float().to(device)
    else:
        fm = ConditionalFlowMatcher(sigma=0.0) if args.method == "icfm" else ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    (OUT / "ckpt").mkdir(parents=True, exist_ok=True)
    name = f"{args.method}_s{args.seed}"
    last = OUT / "ckpt" / f"{name}_last.pt"
    log, t0, start = [], time.time(), 0
    if last.exists():  # a GPU driver reset killed an earlier run: carry on from the last saved epoch
        ck = torch.load(last, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        log, start = ck["log"], ck["epoch"]
        print(f"resumed after epoch {start}", flush=True)
    for epoch in range(start, args.epochs):
        total = 0.0
        for x1, _ in loader:
            x1 = x1.to(device)
            x0 = torch.randn_like(x1)
            if args.method == "ddpm":
                i = torch.randint(0, T, (x1.shape[0],), device=device)
                ab = abar[i][:, None, None, None]
                xt = ab.sqrt() * x1 + (1 - ab).sqrt() * x0
                loss = ((model(i.float() / (T - 1), xt) - x0) ** 2).mean()
            else:
                t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
                loss = ((model(t, xt) - ut) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        log.append(total / len(loader))
        print(f"epoch {epoch + 1}/{args.epochs} loss {log[-1]:.4f} elapsed {time.time() - t0:.0f}s", flush=True)
        if (epoch + 1) % 5 == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "log": log, "epoch": epoch + 1}, last)

    torch.save(model.state_dict(), OUT / "ckpt" / f"{name}.pt")
    (OUT / f"{name}_train.json").write_text(json.dumps({"config": vars(args), "params": n_params, "seconds": time.time() - t0, "loss": log}, indent=2))


if __name__ == "__main__":
    main()

"""Numbers behind Part IV (link to diffusion). Prints everything the slides quote.

Run from the project root:
    python slides/make_link_numbers.py

A. Running example at (t, x) = (0.5, 0.1): the four parameterizations, including a direct score check.
B. Running example with the exact field: W2 of Euler on the straight path vs. DDIM on the VP path, no network.
   Also checks that DDIM equals Euler on the straight path in rescaled coordinates.
C. E1 checkpoints (seed 0): why DDIM fails at few steps. Separates the time grid from the noise parameterization.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from common.data_2d import sample  # noqa: E402
from common.metrics_2d import wasserstein2  # noqa: E402
from common.models import MLP  # noqa: E402

A = 2.0
T = 1000
betas = np.linspace(1e-4, 0.02, T)
abar = np.cumprod(1 - betas)
DATASETS = ["moons", "8gaussians", "checkerboard", "swissroll"]


def grid(K):
    """The E1 DDIM grid: K network calls at these DDPM steps, then a last jump to pure data."""
    idx = np.round(np.linspace(T - 1, 0, K)).astype(int) if K > 1 else np.array([T - 1])
    a = np.append(np.sqrt(abar[idx]), 1.0)
    b = np.append(np.sqrt(1 - abar[idx]), 0.0)
    return idx, a, b


def part_a():
    t, x = 0.5, 0.1
    x1 = A * np.tanh(A * t * x / (1 - t) ** 2)
    eps = (x - t * x1) / (1 - t)
    u = x1 - eps
    s = -eps / (1 - t)
    # direct score of the mixture N(+1, .25), N(-1, .25)
    w = np.exp(-(x - 1) ** 2 / (2 * 0.25))
    v = np.exp(-(x + 1) ** 2 / (2 * 0.25))
    wp, wm = w / (w + v), v / (w + v)
    s_direct = wp * (1 - x) / 0.25 + wm * (-1 - x) / 0.25
    print("A. running example at t = 0.5, x = 0.1")
    print(f"   weights {wp:.3f} {wm:.3f}")
    print(f"   x1_hat {x1:.4f}, eps_hat {eps:.4f}, u {u:.4f}, (x-eps)/t {(x - eps) / t:.4f}")
    print(f"   score from eps {s:.4f}, score direct {s_direct:.4f}, u from score {(x + (1 - t) * s) / t:.4f}")


def x1_hat_toy(x, a, b):
    return A * np.tanh(A * a * x / b ** 2)


def w2_toy(x, q):
    y = np.where(q < 0.5, -A, A)
    return np.sqrt(np.mean((np.sort(x) - y) ** 2))


def part_b():
    q = (np.arange(200000) + 0.5) / 200000
    z = norm.ppf(q)
    print("B. running example, exact field, no network")
    for K in [1, 2, 5, 10, 20, 50]:
        # Euler on the straight path, equal steps in t
        x = z.copy()
        for k in range(K):
            t = k / K
            x = x + (x1_hat_toy(x, t, 1 - t) - x) / (1 - t) / K
        w_lin = w2_toy(x, q)
        # DDIM on the VP path
        _, a, b = grid(K)
        xv = z.copy()
        for k in range(K):
            x1 = x1_hat_toy(xv, a[k], b[k])
            e = (xv - a[k] * x1) / b[k]
            xv = a[k + 1] * x1 + b[k + 1] * e
        # the same thing as Euler on the straight path, in y = x / (a + b), at tau = a / (a + b)
        tau = a / (a + b)
        y = z / (a[0] + b[0])
        for k in range(K):
            x1 = x1_hat_toy(y, tau[k], 1 - tau[k])
            y = y + (tau[k + 1] - tau[k]) * (x1 - y) / (1 - tau[k])
        print(f"   K={K:3d}  straight Euler W2 {w_lin:.3f}  DDIM W2 {w2_toy(xv, q):.3f}  "
              f"Euler on DDIM grid W2 {w2_toy(y, q):.3f}  |DDIM - rescaled Euler| {np.abs(xv - y).max():.2e}")
    for K in [1, 2, 5, 10]:
        _, a, b = grid(K)
        print(f"   K={K}: tau grid", np.round(a / (a + b), 3).tolist(), " a[0] =", round(a[0], 4), " b/a =", round(b[0] / a[0], 1))
    idx = np.array([999, 750, 500, 250, 100, 0])
    print("   DDPM step, t, a, b:", [(int(i), round(1 - i / 1000, 2), round(float(np.sqrt(abar[i])), 3), round(float(np.sqrt(1 - abar[i])), 3)) for i in idx])
    print(f"   a^2 at the last step {abar[-1]:.2e}, a {np.sqrt(abar[-1]):.4f}")


@torch.no_grad()
def part_c():
    print("C. E1 checkpoints, seed 0, same noise and test set as E1")
    for data in DATASETS:
        rng = np.random.default_rng(10_000)
        real = sample(data, 2000, rng)
        g = torch.Generator().manual_seed(20_000)
        z = torch.randn(2000, 2, generator=g)
        eps_net = MLP(width=256)
        eps_net.load_state_dict(torch.load(ROOT / f"results/e1/ckpt/ddim_{data}_s0.pt", map_location="cpu"))
        eps_net.eval()
        vel_net = MLP(width=256)
        vel_net.load_state_dict(torch.load(ROOT / f"results/e1/ckpt/icfm_{data}_s0.pt", map_location="cpu"))
        vel_net.eval()
        print(f"  {data}")
        for K in [1, 2, 5, 10, 20]:
            idx, a, b = grid(K)
            # DDIM with the noise network
            x = z.clone()
            for k in range(K):
                t = torch.full((2000,), float(idx[k]) / (T - 1))
                e = eps_net(x, t)
                x1 = (x - b[k] * e) / a[k]
                x = a[k + 1] * x1 + b[k + 1] * e
            w_ddim = wasserstein2(x.float(), real)
            # the velocity network, equal steps
            y = z.clone()
            for k in range(K):
                y = y + vel_net(y, torch.full((2000,), k / K)) / K
            w_eq = wasserstein2(y, real)
            # the velocity network, on the DDIM grid, in rescaled coordinates
            tau = a / (a + b)
            y = z.clone() / float(a[0] + b[0])
            for k in range(K):
                y = y + float(tau[k + 1] - tau[k]) * vel_net(y.float(), torch.full((2000,), float(tau[k])))
            w_grid = wasserstein2(y.float(), real)
            print(f"    K={K:3d}  DDIM {w_ddim:6.2f}   velocity net, equal steps {w_eq:5.2f}   velocity net, DDIM grid {w_grid:5.2f}")
        # how far is the noise network from the truth at the first call?
        ref = sample(data, 20000, np.random.default_rng(1))
        _, a, b = grid(1)
        t = torch.full((2000,), 1.0)
        e = eps_net(z, t)
        # exact posterior mean of x1 given x at this t, from a large data sample
        d2 = ((z[:, None, :] - float(a[0]) * ref[None, :, :]) ** 2).sum(-1) / (2 * float(b[0]) ** 2)
        w = torch.softmax(-d2.double(), dim=1).float()
        x1_true = (w[:, :, None] * ref[None, :, :]).sum(1)
        e_true = (z - float(a[0]) * x1_true) / float(b[0])
        rms = (e - e_true).pow(2).sum(-1).mean().sqrt().item()
        x1_net = (z - float(b[0]) * e) / float(a[0])
        print(f"    first call: a {a[0]:.4f}, b/a {b[0] / a[0]:.0f}, RMS noise error {rms:.4f}, "
              f"predicted endpoint RMS (b/a * error) {rms * b[0] / a[0]:.2f}, "
              f"RMS of net endpoint around true posterior mean {(x1_net - x1_true).pow(2).sum(-1).mean().sqrt().item():.2f}")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()

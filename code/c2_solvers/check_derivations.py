"""推导笔记 docs/notes/derivations.md 的数值核对（只用 CPU，约 1 分钟）。

任务：
1. D8：TV 的噪声下限，公式与多项分布模拟对照
2. D4：MNIST（像素缩放到 [-1, 1]）上独立配对的代价和 Gelbrich 上限 2 tr Σ^½
3. D5：1D 例子上均匀网格、√κ 网格、shift 网格的 Euler 终点误差（笔记 D5 的表）
4. D9：VP 路径与直线路径换坐标后相同、DDIM 一步 = Euler 一步、ε 损失与 CFM 损失只差时间权重

运行：python code/c2_solvers/check_derivations.py
"""
import numpy as np, json, os
os.environ.setdefault("OMP_NUM_THREADS", "4")
root = r"C:\Users\Wentao\Desktop\cfm_project"

# ---- D8: TV noise floor
n, m = 10000, 30000
p = np.full(10, 0.1)
floor = 0.5 * np.sum(np.sqrt(2 / np.pi * p * (1 - p) * (1 / n + 1 / m)))
ref = np.array(json.load(open(rf"{root}\results\e3\kmeans.json"))["ref_digit_share"])
floor2 = 0.5 * np.sum(np.sqrt(2 / np.pi * ref * (1 - ref) * (1 / n + 1 / m)))
rng = np.random.default_rng(0)
sims = []
for _ in range(2000):
    a = rng.multinomial(n, p) / n; b = rng.multinomial(m, p) / m
    sims.append(0.5 * np.abs(a - b).sum())
print(f"D8 TV floor formula {floor:.4f} (digit shares {floor2:.4f}); simulated {np.mean(sims):.4f}")

# ---- D4: Gelbrich bound on MNIST in [-1, 1]
from torchvision.datasets import MNIST
x = MNIST(rf"{root}\data", train=True, download=False).data.numpy().reshape(60000, -1).astype(np.float64) / 127.5 - 1
mu = x.mean(0); S = np.cov(x, rowvar=False)
ev = np.clip(np.linalg.eigvalsh(S), 0, None)
d = x.shape[1]
indep = d + mu @ mu + ev.sum()
print(f"D4 independent cost {indep:.1f}, bound 2 tr S^1/2 = {2*np.sqrt(ev).sum():.1f}, max saving {2*np.sqrt(ev).sum()/indep:.1%}")


A, SD = 2.0, 0.5
def make_u(sd):
    def u(x, t):
        var = t**2 * sd**2 + (1 - t)**2
        lw = []; mu1 = []
        for c in (A, -A):
            lw.append(-(x - t*c)**2 / (2*var))
            mu1.append(c + t*sd**2/var * (x - t*c))
        lw = np.array(lw); w = np.exp(lw - lw.max(0)); w /= w.sum(0)
        xh = (w * np.array(mu1)).sum(0)
        return (xh - x) / (1 - t)
    return u
T1 = 1 - 1e-4
def rk4(u, x, grid):
    for a, b in zip(grid[:-1], grid[1:]):
        h = b - a; k1 = u(x, a); k2 = u(x + h/2*k1, a + h/2); k3 = u(x + h/2*k2, a + h/2); k4 = u(x + h*k3, b)
        x = x + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return x
def euler(u, x, grid):
    for a, b in zip(grid[:-1], grid[1:]): x = x + (b - a)*u(x, a)
    return x
for sd in (0.0, SD):
    u = make_u(sd); x0 = rng.standard_normal(4000)
    xref = rk4(u, x0, np.linspace(0, T1, 20001))
    tg = np.linspace(0, T1, 4001); xs = x0[:1000].copy(); vs = []
    for i, t in enumerate(tg):
        vs.append(u(xs, t))
        if i < len(tg)-1: xs = rk4(u, xs, tg[i:i+2])
    vs = np.array(vs); w = np.diff(tg)
    kappa = np.abs(np.diff(vs, axis=0)/w[:, None]).mean(1); kappa = np.maximum(kappa, 1e-3*kappa.max())
    G = (np.sum(np.sqrt(kappa)*w))**2/(np.sum(kappa*w)*T1)
    def kgrid(N):
        F = np.concatenate([[0], np.cumsum(np.sqrt(kappa)*w)]); F /= F[-1]
        return np.interp(np.linspace(0, 1, N+1), F, tg)
    def shift(N, s):
        t = np.linspace(0, 1, N+1); return T1 * s*t/(1 + (s-1)*t)
    print(f"target sd={sd}: G={G:.3f}, kappa peak t={tg[np.argmax(kappa)]:.2f}, kappa at t=0.9 / peak = {kappa[int(0.9*4000)]/kappa.max():.3f}")
    for N in (2, 4, 8, 16):
        row = {"uniform": np.linspace(0, T1, N+1), "sqrt-kappa": kgrid(N)}
        for s in (0.25, 0.5, 2, 4): row[f"shift{s}"] = shift(N, s)
        e = {k: np.sqrt(np.mean((euler(u, x0, g) - xref)**2)) for k, g in row.items()}
        print(f"  N={N:2d} " + "  ".join(f"{k} {v:.3f}" for k, v in e.items()))

# ---- D9：CFM 与 diffusion 的换算（两高斯目标，VP 路径 a = sin(πt/2), b = cos(πt/2)）
def post_x1(x, a, b, sd=SD):
    """x_t = a x1 + b x0 时 E[x1 | x_t = x]，直接按 VP 路径的高斯混合计算，不经过直线路径。"""
    var = a**2 * sd**2 + b**2
    lw = np.array([-(x - a*c)**2 / (2*var) for c in (A, -A)])
    mu = np.array([c + a*sd**2/var*(x - a*c) for c in (A, -A)])
    w = np.exp(lw - lw.max(0)); w /= w.sum(0)
    return (w*mu).sum(0)
av = lambda t: np.sin(np.pi*t/2); bv = lambda t: np.cos(np.pi*t/2)
da = lambda t: np.pi/2*np.cos(np.pi*t/2); db = lambda t: -np.pi/2*np.sin(np.pi*t/2)
xs = rng.standard_normal(2000)
# (1) 后验均值在换坐标后相同：E[x1 | x^VP_t = x] = E[x1 | y_τ = x/s]
t = 0.37; a, b = av(t), bv(t); s = a + b; tau = a/s
print(f"D9 (1) posterior mean, max |VP - straight(rescaled)| = {np.max(np.abs(post_x1(xs*s, a, b) - post_x1(xs, tau, 1-tau))):.2e}")
# (2) VP 概率流 ODE 的解 / s 等于直线 ODE 在 τ(t) 处的解
def u_vp(x, t):
    a, b = av(t), bv(t); x1 = post_x1(x, a, b); x0 = (x - a*x1)/b
    return da(t)*x1 + db(t)*x0
t_end = 0.9
x_vp = rk4(u_vp, xs.copy(), np.linspace(0, t_end, 20001))
u_st = make_u(SD); tau_end = av(t_end)/(av(t_end) + bv(t_end))
y_st = rk4(u_st, xs.copy(), np.linspace(0, tau_end, 20001))
print(f"D9 (2) ODE: max |x_VP(t)/s(t) - y_straight(tau)| at t = {t_end} = {np.max(np.abs(x_vp/(av(t_end)+bv(t_end)) - y_st)):.2e}")
# (3) DDIM 一步 = 直线路径换坐标后的一步 Euler
t, t2 = 0.3, 0.7; x = xs*(av(t) + bv(t))
x1h = post_x1(x, av(t), bv(t)); x0h = (x - av(t)*x1h)/bv(t)
x_ddim = av(t2)*x1h + bv(t2)*x0h
s1, s2 = av(t) + bv(t), av(t2) + bv(t2); tau1, tau2 = av(t)/s1, av(t2)/s2
y_euler = x/s1 + (tau2 - tau1)*u_st(x/s1, tau1)
print(f"D9 (3) DDIM step vs Euler in (y, tau): max diff = {np.max(np.abs(x_ddim/s2 - y_euler)):.2e}")
# (4) 损失恒等式 ||v - (x1 - x0)||^2 = ||eps - x0||^2 / t^2（直线路径，v = (x_t - eps)/t）
x0, x1, eps, t = rng.standard_normal((4, 1000))
t = np.abs(t) % 1 + 0.01; xt = t*x1 + (1 - t)*x0
lhs = ((xt - eps)/t - (x1 - x0))**2; rhs = (eps - x0)**2/t**2
print(f"D9 (4) loss identity, max |lhs - rhs| = {np.max(np.abs(lhs - rhs)):.2e}")

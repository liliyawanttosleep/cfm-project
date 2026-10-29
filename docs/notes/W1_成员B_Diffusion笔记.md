# W1 笔记 · 成员 B · Diffusion

> 目标：1 页左右，之后直接改写成 report 的 Methods（Diffusion 部分）。

## 1. 前向过程
- q(x_t | x_{t−1})、q(x_t | x_0) = N(√ᾱ_t x_0, (1−ᾱ_t) I) 的推导

## 2. 反向过程与 ELBO
- 后验 q(x_{t−1} | x_t, x_0) 的均值与方差
- 为什么 ELBO 可以简化成 ε-预测的 MSE（Ho et al. 2020 式 14）

## 3. 与 score matching 的联系
- s_θ(x_t, t) = −ε_θ(x_t, t) / √(1−ᾱ_t)
- Denoising score matching（Vincent 2011）

## 4. SDE 视角与 DDIM
- VP-SDE：dx = −½β(t) x dt + √β(t) dW
- Probability flow ODE：dx = [−½β(t) x − ½β(t) ∇log p_t(x)] dt
- DDIM（η=0）可以看作这个 ODE 的离散化 → 与 FM 的 ODE 采样可以直接对比

## 5. 复现记录
- `code/e1_2d/train_ddpm.py` 读懂了哪些部分，跑出的结果（图放 `results/w1/`）

## 6. 疑问（周会讨论）
-

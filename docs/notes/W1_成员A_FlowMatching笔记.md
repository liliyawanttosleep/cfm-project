# W1 笔记 · 成员 A · Flow Matching

> 目标：1 页左右，之后直接改写成 report 的 Methods（FM 部分）。

## 1. 问题设定
- 源分布 p0 = N(0, I)，目标分布 p1 = 数据分布；用 ODE dx/dt = v_θ(x, t) 把 p0 推到 p1。

## 2. Flow Matching 目标
- 边缘概率路径 p_t(x) 与边缘速度场 u_t(x)：
- FM loss：E_{t, x~p_t} ||v_θ(x,t) − u_t(x)||²，问题在于 u_t 无法直接计算。

## 3. Conditional Flow Matching
- 条件路径 p_t(x | z)、条件速度 u_t(x | z)；直线路径时 x_t = (1−t)x0 + t x1，u_t = x1 − x0
- **定理**：∇_θ L_CFM = ∇_θ L_FM（写出证明要点）

## 4. 不同耦合 q(z)
| 方法 | q(z) | 特点 |
|---|---|---|
| I-CFM / 1-RF | q(x0) q(x1) 独立 | |
| OT-CFM | minibatch OT 计划 π(x0, x1) | |
| FM (Lipman) | 以 x1 为条件 | |

## 5. 复现记录
- 跑了哪些 notebook，看到什么现象（截图放 `results/w1/`）

## 6. 疑问（周会讨论）
-

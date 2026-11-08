# 11/10 课堂报告 Slide 重做规划

> 状态：规划稿，等 Wentao 和 Kaiming 确认（待决定事项见第 10 节）。确认后，第 4 节的逐页大纲取代 `TEAM_PLAN.md` §4 P6 的旧大纲。
> 语言：规划用中文。**slide 上的文字、讲稿、图中文字一律用英文**，所以第 4 节的页标题直接写英文。
> 新 deck：源文件 `slides/deck_v2/project/`，线上 https://claude.ai/artifact/7Lpjq76Dw2ywYrmNCLgUDR ，PDF 用 `python slides/build_pdf.py --deck deck_v2 --notes` 生成（`build/FM_vs_Diffusion_Talk.pdf`）。旧 deck 不动。
> 依据：
> - 教程：[A Visual Dive into Conditional Flow Matching](https://dl.heeere.com/conditional-flow-matching/blog/conditional-flow-matching/)（ICLR Blogposts 2025，下称 Blog）
> - 库：[atong01/conditional-flow-matching](https://github.com/atong01/conditional-flow-matching)（torchcfm，MIT，本地在 `conditional-flow-matching-main/`）
> - 主线 notebook：`examples/2D_tutorials/Flow_matching_tutorial.ipynb`（下称 NB，我们本地带 W2 评估的版本）

---

## 1. 这次报告讲什么

**主题**：Conditional Flow Matching 怎样把"生成数据"变成"回归一个速度场"，以及同一套方法怎样从 2D 用到图像。Diffusion 作为"同一个回归、换一条路径"来对比。

**和旧 deck 的区别**

| | 旧 deck（26 页） | 新 deck |
|---|---|---|
| 组织方式 | 按方法并列，再讲实验设计 | 按一条推导链走，像一堂 tutorial |
| 重点 | 比较 + 实验设计 | 讲懂公式：每个公式的含义，以及它怎样支持生成 |
| Diffusion | 占一个大部分（3 页） | 放在 CFM 之后，用转换公式连起来（2 页） |
| 例子 | 只有 2D | 手算 1D → 2D → MNIST → CIFAR-10 |
| 实验设计 | 4–5 页 | 压缩成 1 页（课程仍要求讲项目计划） |

**听完后听众应该能做到**
1. 写出 `L_CFM` 并解释回归目标为什么是 `x₁ − x₀`。
2. 说清楚为什么回归一个带噪声的目标，得到的场真的能把噪声送到数据（后验均值 + 连续性方程）。
3. 解释配对方式为什么决定路径形状，路径形状为什么决定采样步数。
4. 把 velocity、noise、score 三种参数化互相转换，知道 DDPM/DDIM 在这个框架里的位置。
5. 知道换成图像时什么变了（网络结构），什么没变（损失函数和那 5 行训练代码）。

**听众**：统计 / CS 博士生，有 CNN 基础。
- 可以默认会：回归、条件期望、贝叶斯公式、ResNet、U-Net 的大致结构。
- 不能默认会：连续性方程、最优传输、Neural ODE、SDE。这些要从头讲，但不证明细节。
- 因此公式可以多，但**每个符号都要解释**；证明只留一两步关键步骤，完整证明放 backup。

**最终例子（开场就放，结尾再回到它）**
- 2D：8 Gaussians → two moons，I-CFM 的路径弯曲，OT-CFM 接近直线。
- 图像：同样的代码换成 U-Net，在 CIFAR-10 上从噪声生成图像，并展示不同步数的结果。

---

## 2. 贯穿全场的两个设计

### 2.1 主线：解释这 5 行代码

第 4 页就放出这段代码，并提三个问题（slide 上写 1、2、3，不用 Q 编号，因为 Q1–Q4 已是项目的研究问题），后面每一部分回答一个。第 14 页回来给每行标上对应的公式。代码摘自 `mnist_example.ipynb`，2D 和图像只差数据和网络。

```python
x1 = data_batch;  x0 = torch.randn_like(x1)                 # z = (x0, x1)
t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1) # x_t = (1-t)x0 + t x1,  u = x1 - x0
vt = model(t, xt)                                           # v_θ(x_t, t)
loss = torch.mean((vt - ut) ** 2)                           # L_CFM
loss.backward(); optimizer.step()
```

三个问题：**问题 1** 为什么回归 `x₁ − x₀`，它甚至不是一个确定的函数？（Part II）**问题 2** 为什么这样训练出来的场能生成数据？（Part I + II）**问题 3** 为什么换一种配对，步数就能大幅减少？（Part III）

### 2.2 每个公式页的固定模板

这是对"讲清楚公式含义"这个要求的落实。每个核心公式页都回答这四件事，版式也固定：

| 栏目 | 内容 |
|---|---|
| **Formula** | 公式本身，只放一个主公式 |
| **In words** | 一句话说它是什么意思，逐个解释符号 |
| **Why it helps generation** | 它在"从噪声到数据"这条链上解决了什么问题 |
| **In code** | 对应的一行代码，或 NB 的 cell 编号 |

---

## 3. 记号约定（全场统一）

三份资料的记号不一样，这是听众最容易混的地方，所以第 9 页和第 20 页都要明确讲。

| | 我们的 slide | Blog | NB / torchcfm | DDPM 原文 |
|---|---|---|---|---|
| 噪声端 | `x₀ ~ p₀`，t = 0 | 同左 | `x0`，分布记 α | `x_T`，t = T |
| 数据端 | `x₁ ~ p₁ = p_data`，t = 1 | 同左（Diffusion 小节除外） | `x1`，分布记 β | `x₀`，t = 0 |
| 边缘速度场 | `uₜ(x)` | `u(x,t)` | `u_t(x)` | — |
| 条件速度场 | `uₜ(x│z)` | `u^cond(x,t,z)` | `u_t(x│z)` | — |
| 网络 | `vθ(x,t)` | `u_θ` | `v_θ(t,x)`，`model(t, xt)` | `ε_θ(x_t,t)` |
| 配对 | `z = (x₀,x₁) ~ q` 或 `π` | `z`，`π` | `z ~ q` | — |

**讲法**：DDPM 的时间方向与 FM 相反。我们在 Diffusion 部分只说一次，然后全部换算成 "t = 0 噪声，t = 1 数据"。

---

## 4. 逐页大纲（主体 29 页，约 35 分钟）

"讲者"是默认分配（W = Wentao，K = Kaiming），调整方案见第 10 节。"图"的编号对应第 7 节。

### 总览

| 部分 | 回答什么 | 页码 | 时长 | 讲者 |
|---|---|---|---|---|
| Opening | 终点是什么，要解释什么 | 1–4 | 4′ | W |
| Part I · Generation as transport | 速度场怎样决定分布 | 5–8 | 4.5′ | K |
| Part II · Conditional Flow Matching | 问题 1、2：为什么回归 `x₁ − x₀` 就够了 | 9–14 | 9′ | W |
| Part III · 2D examples | 问题 3：配对、路径、步数 | 15–19 | 6′ | W |
| Part IV · Link to diffusion | 同一个回归，另一条路径 | 20–21（实际 20–22，之后各页 +1） | 3.5′（实际约 3.8′） | K |
| Part V · From 2D to images | 高维下什么变、什么不变 | 22–26 | 6′ | K / W |
| Close | 项目计划、总结 | 27–29 | 2′ | K / W |

### Opening（4′）✅ 第一批已完成（2026-11-05），等审阅

**1. Flow Matching vs. Diffusion**（0.3′，W）封面，副标题 "Turning generation into regression, from 2D to images"。

**2. What we set out to do**（1.2′，W）项目主要目标（用户 11/05 要求从这里开始，所以新增这一页）
- 一句话目标：讲清楚 flow matching 如何把生成变成回归，再测量它和 diffusion 在质量与采样成本上的权衡。
- 三个目标：Explain / Compare / Evaluate。尺度：d = 2 → 784 → 3,072。

**3. Where the method takes us**（1′，W）
- 放两段动图：8 Gaussians → moons；CIFAR-10 从噪声到图像。图 F1。
- 只讲一句话：这两个例子用的是同一个损失和同一个训练循环，只有数据和网络不同。
- CIFAR 动图来自 torchcfm 库，图注已标明；F10 做好后可以换成我们自己的结果。

**4. Five lines that train a generative model**（1.5′，W）
- 放第 2.1 节的代码和三个问题。
- 讲法：不用似然、不解 ODE、不用判别器，就是一个 MSE 回归。接下来解释它为什么成立。

### Part I · Generation as transport（4.5′）✅ 第二批已完成（2026-11-05），等审阅

**5. The goal: push noise forward to data**（0.7′，K）
- 公式：学一个映射 T，使 `T#p₀ ≈ p_data`。
- 讲法：Normalizing flow 需要可逆和可算的 log-det，结构受限；CNF 去掉了限制，但按似然训练时每步都要解 ODE。这就是 flow matching 要解决的问题。Blog "Normalizing Flows / Continuous Normalizing Flows"。

**6. A ResNet with infinitely many layers is an ODE**（1.3′，K）
- 公式：`x_{k+1} = x_k + (1/K) u(x_k, t_k)` → `dx/dt = uₜ(x)`。
- 含义：残差块是一步 Euler，这是给 CNN 背景听众的入口。
- 对生成的作用：**采样就是从 `x₀` 积分到 t = 1；NFE 就是调用网络的次数。**
- 图：path / velocity / flow 三者关系（F2）。

**7. Velocity moves density: the continuity equation**（1.5′，K）
- 公式：`∂ₜpₜ + ∇·(pₜuₜ) = 0`。NB cell 2。
- 含义：概率质量守恒，某处密度的变化等于流入减流出，和流体一样。
- 对生成的作用：**只要 `uₜ` 和 `pₜ` 满足这个方程，且 `p₁ = p_data`，积分这个场得到的就是数据分布。**这是"生成正确"的判据，第 11 页要用。
- 图：`prob_paths.png` 或 `paths.gif`（已有）。

**8. The ideal loss, and why we cannot compute it**（1′，K）
- 公式：`L_FM(θ) = E_{t, x~pₜ} ‖vθ(x,t) − uₜ(x)‖²`。NB cell 3。
- 两个困难：`uₜ` 和 `pₜ` 未知；连接 `p₀` 和 `p_data` 的路径有无穷多条，不选一条就没有监督信号。
- 过渡：先在最简单的情形下把路径定下来。

### Part II · Conditional Flow Matching（9′，核心）✅ 第三批已完成（2026-11-05），等审阅

> 实现时加了两个设计，让听众跟得上推导：
> - 页面顶部有一条进度条：One pair → All pairs → Posterior mean → Closed form → Same gradient → Code，标题也写成 Step 1–6。
> - 全程跟踪同一个点 (t, x) = (0.5, 0.1)：正好两条线经过它（`x₀ = −1.8` 和 `x₀ = 2.2`），速度 +3.8 和 −4.2，后验权重 0.69 和 0.31，平均 1.32，损失下限 13.69。第 11、12、13 页用的都是这组数字（`slides/make_cfm_figs.py` 里算出，已用蒙特卡洛核对）。


**9. Start from one pair: a straight line**（1.2′，W）
- 公式：`xₜ = (1−t)x₀ + t x₁`，`uₜ(x│z) = dxₜ/dt = x₁ − x₀`，`z = (x₀, x₁)`。
- 含义：给定一对点，路径和速度都有闭式解，这就是回归目标的来源。NB cell 4–7。
- 在这页说明时间方向约定。图 F3 左。
- 脚注：一般的高斯形式是 `uₜ(x│z) = (σ'ₜ/σₜ)(x − μₜ) + μ'ₜ`；σ 取常数时第一项为 0，所以 torchcfm 直接返回 `x1 - x0`。

**10. Mix over all pairs to get the path**（1.3′，W）
- 公式：`pₜ(x) = ∫ pₜ(x│z) q(z) dz`。
- 含义：单条直线不是生成模型，把所有配对的直线叠加起来才得到一条分布路径。
- 对生成的作用：检查两端。t = 0 时混合后是 `p₀`，t = 1 时是 `p_data`，所以这条路径的两端是对的。
- 图：几条线 → 很多条线 → 密度热图（F3 右），对应 Blog 的 "empirical average converging"。

**11. The marginal velocity is a posterior mean**（2′，W，全场最重要的一页）
- 公式：`uₜ(x) = E_q[ uₜ(x│z) pₜ(x│z) / pₜ(x) ] = E[ x₁ − x₀ │ xₜ = x ]`。NB cell 6，Blog Theorem 1。
- 含义：站在 (t, x) 这一点，所有可能经过这里的配对各有一个方向，**用贝叶斯后验加权平均**，就是边缘速度。
- 对生成的作用：这个 `uₜ` 和 `pₜ` 满足连续性方程。讲一步证明即可：对 `pₜ(x│z)` 的连续性方程关于 `q(z)` 积分，线性性直接给出结果。由第 7 页，积分它就能到达数据分布。
- 图 F4：(t, x) 平面上经过某点的几条直线和它们的平均箭头（Blog 交互图的静态版）。

**12. Worked example: the velocity in closed form**（1.5′，W）
- 设定：`x₀ ~ N(0,1)`，`x₁ = ±a` 各占一半。
- 推导：`x₁ − x₀ = (x₁ − xₜ)/(1−t)`，所以 `uₜ(x) = (E[x₁│xₜ = x] − x)/(1−t)`；再由贝叶斯公式，`E[x₁│xₜ = x] = a·tanh(a t x/(1−t)²)`。
- 结论：`uₜ(x) = ( a·tanh(a t x/(1−t)²) − x ) / (1−t)`，**不需要神经网络，直接积分这个公式就生成了数据。**
- 图 F5：`pₜ` 热图 + 速度箭头 + 积分轨迹。
- 已用数值验证（2026-11-05）：20 万个点积分到 t = 1 后全部落在 ±2，各占约 50%，每个点都留在初始符号的一侧；公式与蒙特卡洛估计的 `E[x₁−x₀│xₜ≈x]` 一致。
- 顺带一点：条件直线会交叉，但边缘流的轨迹不交叉。

**13. The tractable loss has the same gradient**（1.7′，W）
- 公式：`L_CFM(θ) = E_{t, z, x~pₜ(·│z)} ‖vθ(x,t) − uₜ(x│z)‖²`。NB cell 8。
- 关键一步（统计听众最容易接受的讲法）：把 `Y = x₁ − x₀` 看作带噪声的响应，

  `E‖vθ − Y‖² = E‖vθ − E[Y│xₜ,t]‖² + E[Var(Y│xₜ,t)]`

  第一项就是 `L_FM`，第二项与 θ 无关。所以 `∇θL_CFM = ∇θL_FM`，**最小二乘恢复的是回归函数**。这回答了问题 1。
- 推论：损失**不会趋于 0**，最小值就是条件方差。
- 真实例子：NB 中 I-CFM 的训练损失在 7.9–9.2，OT-CFM 在 0.14–0.18（8 Gaussians → moons，来自 NB cell 11、17 的输出）。这个差距反映的是条件方差，不能用损失值比较生成质量。

**14. From formulas to code**（1.3′，W）
- 回到第 4 页的代码，每行标上公式；旁边放采样循环：`x ← x + (1/N) vθ(x, k/N)`。
- 对应 NB cell 5、7、11，`ConditionalFlowMatcher`。
- 小结：问题 1、2 已回答。

### Part III · 2D examples（6′）✅ 第四批已完成（2026-11-05），等审阅

> 这部分直接对应 NB 的实验：8 Gaussians → moons。注意源分布不是标准高斯，这正好说明 CFM 的 `p₀` 可以是任意可采样分布。

> 实际做法（第四批）：继续沿用 Part II 的运行示例 `N(0,1) → ±2`，并加了第 16 页的 `x₀ = 1`（初速度 −1，先离开 +2）和第 17 页的交换恒等式：交换两条在 t 时刻相交的直线的终点，总平方长度正好减少 `2t(1−t)‖v−v′‖²`；在 (0.5, 0.1) 上是 32.08 → 0.08，减少 32。第 18 页的路径长度比（1.35 / 1.001）和直线度（9.7 / 0.02）由 `make_2d_figs.py` 从 `moon_*.pt` 算出（2,000 条路径，Euler 200 步）。

**15. Demo: 8 Gaussians to two moons with I-CFM**（1′，W）
- 图 F6 左：轨迹图。讲法：能生成，但路径明显弯曲。

**16. Why the paths bend, and what one step does**（1.2′，W）
- 独立配对下直线互相交叉，交叉处网络只能学平均方向，所以轨迹是弯的。图 F7 左。
- 一个可以当场验证的推论：在 t = 0，`u₀(x) = E[x₁] − x`，所以**走一步 Euler 正好落到数据的均值 `E[x₁]`**。这解释了为什么 I-CFM 一步时所有点缩到中心。
- 数据支持：把四个数据集标准化后，塌缩到均值的 W2 理论值是 √(tr Cov) ≈ 1.41；我们测到的 I-CFM 一步 W2 为 1.28–1.37（`results/e1/summary.csv`，seed 0）。

**17. Change the coupling: optimal transport pairs**（1.5′，W）
- 公式：`π* = argmin_π ∫ ‖x₀ − x₁‖² dπ(x₀,x₁)`；minibatch 版本在每个 batch 上解。NB cell 15–17。
- 代码只多一行：`x0, x1 = ot_sampler.sample_plan(x0, x1)`，理论和损失都不变，只把 `q(z)` 换成 `π`。
- 图 F7 右：同一批点，随机配对 vs OT 配对。
- 必须澄清：Lipman 的 "OT path" 是**单对之间**的路径；OT-CFM 的 OT 是**配对方式**。这是两件事。

**18. OT-CFM: same data, straighter paths**（1′，W）
- 图 F6：I-CFM / OT-CFM 轨迹并排（SB-CFM 放 backup）。
- 表：W2、损失值、训练耗时。W2 数字来自 NB：I-CFM 0.46 ± 0.06，OT-CFM 0.34 ± 0.06，SB-CFM 0.37 ± 0.05（1024 样本，100 次）。**注意**：第 19 页的数字不能和这张表直接比，见第 9 节。

**19. Fewer steps: quality vs NFE on four datasets**（1.3′，W）
- 图：`w2_all.png`，`datasets_grid.png`（已有），可只留一张加小表。
- 结论：OT-CFM 从第 1 步就基本平稳，I-CFM 在 20–50 步才接近（8 Gaussians 上停在 0.31），DDIM 少步时最差。
- 一句话带过 Rectified Flow：用模型自己生成的配对重新训练，是另一种拉直方法（详见 backup）。
- 标明：Seed 0，源分布是标准高斯。

### Part IV · Link to diffusion（3.5′）✅ 第五批已完成（2026-11-05），等审阅

> **实际做法与下面的原始大纲不同：**
> - 拆成 3 页（第 20–22 页，`same-regression`、`one-quantity`、`samplers`），讲稿共约 3.8′。**后面各页的编号都要 +1**：Part V 变成 23–27，Close 变成 28–30。本文档后面的页码仍是原编号。
> - 第 21 页把跟踪点算完整：`x̂₁ = 0.76`，`ε̂ = −0.56`，`u = 1.32`，`s = 1.12`（用混合分布直接算 score 也是 1.12）。
> - 第 22 页纠正了下面第 21 条的说法：**"步数差异来自路径形状"不成立。** 独立配对、高斯源时，VP 的轨迹除以 `a + b`、以 `τ = a/(a+b)` 为时钟后，就是直线路径的轨迹，DDIM 等于该时间网格上的 Euler（玩具例子上数值差 < 1e-14）。实测原因（`results/e1/ckpt`，seed 0，4 个数据集）：
>   - NFE 1–2：第一次调用 `b/a = 157`，噪声误差 .05–.09 放大成 8–14，实测 W2 7.2–12.6；精确场上只有 1.98。
>   - NFE 5：主要是时间网格。同一个速度网络等步长 .29–.33，换成 DDIM 网格 .50–.54，DDIM 本身 .56–.83。这与数据有关：玩具例子上 DDIM 网格反而更好（.001 vs .046）。
>   - NFE 20：DDIM .19–.52，同一网格上的速度网络 .15–.30。剩余差距来自参数化还是训练，**没有分开**。
> - 数字来自 `slides/make_link_numbers.py`，图来自 `slides/make_link_figs.py`。B8 的内容已并入第 22 页。

**20. Diffusion: the same regression on another path**（1.7′，K）
- 公式：两者都是 `xₜ = aₜx₁ + bₜx₀`。

  | | `aₜ` | `bₜ` | 回归目标 |
  |---|---|---|---|
  | Linear (I-CFM / RF) | `t` | `1−t` | `x₁ − x₀` |
  | VP (DDPM) | `√ᾱ` | `√(1−ᾱ)` | `ε = x₀` |

- 讲法：DDPM 的时间方向相反，这页说一次，之后统一换算。图：`prob_paths.png`。

**21. Velocity, noise and score are interchangeable**（1.8′，K）
- 公式（线性路径，高斯 `x₀`）：

  `uₜ(x) = (x − ε̂)/t = (x + (1−t) sₜ(x))/t`，其中 `ε̂ = E[x₀│xₜ=x]`，`sₜ = ∇log pₜ = −ε̂/(1−t)`。

- 含义：预测速度、预测噪声、预测 score，只是同一个条件期望的不同写法。第 12 页的例子上已数值验证。
- 结论：DDIM 就是解 VP 路径的 ODE，DDPM 是同一边缘分布上的 SDE。Blog "Link Between Diffusion and Flow-Matching" 证明了 VP 路径上 FM ODE = 反向 probability-flow ODE。
- ~~区别只在路径，所以 Part III 的步数差异来自路径形状，而不是"两类不同的模型"。~~ 已被实测否定，见本节开头。
- 再放 `sampling.gif` 对比 OT-CFM / I-CFM / DDIM。

### Part V · From 2D to MNIST（6′）✅ 已完成（2026-11-06），等审阅

> **实测结论（`results/e4/mnist_eval.json`，`results/e4/ot_numbers.json`，seed 0，每种方法一个模型）：**
> - FID（NFE 1 / 5 / 20 / 100 / dopri5）：OT-CFM 223.1 / 26.7 / 17.0 / 13.9 / 11.5，I-CFM 332.5 / 45.3 / 20.7 / 14.5 / 12.1，DDIM 292.7 / 99.3 / 28.4 / 12.2 / —。real-vs-real 0.54。
> - I-CFM 一步落到数据均值（RMS 0.04，真实数字离均值 0.52），预测成立。DDIM 在 1–2 步是白点（74% 像素超出范围），NFE 1 时 FID 反而排在 I-CFM 的模糊图之前，这是 FID 的缺陷，页面已说明。DDIM 在 100 步时最低（12.2），原因**没有分开**。
> - H3 部分成立：batch 128 只省下配对上限的 17%（2D 为 85–94%），但 2–5 步时 FID 仍低约 40%；100 步时只差 4%。只有一个种子。
> - 第 26 页按 MNIST 做，F10、F12 取消，第 27 页用表代替 F12。B7、B8 不做（B7 属于 CIFAR，B8 已并入第 22 页）。

> **2026-11-06 改动：不再做 CIFAR-10 案例，图像部分只用我们自己训练的 MNIST。** 编号已按 Part IV 之后的实际页码（+1）。第 2、3 页也要同步：尺度只剩 2D 和 MNIST；第 3 页右侧换成我们自己的 MNIST 动图。

**23. What changes from 2D to images, and what does not**（1.2′，K）

| | 2D（NB） | MNIST（`code/e2_mnist/train_mnist.py`） |
|---|---|---|
| 维度 | 2 | 784 |
| 网络 | MLP（64 宽） | U-Net，32 通道，1.1M 参数 |
| 训练细节 | Adam | Adam，batch 128，50 epochs，无 EMA |
| **损失和循环** | **完全相同** | **完全相同** |

**24. What the network sees**（1′，K）
- 图 F8：MNIST 数字的 `x₀`，`x_{0.25}`，`x_{0.5}`，`x_{0.75}`，`x₁`，以及目标 `x₁ − x₀`。
- `vθ(t, xₜ)` 输入和输出同尺寸，U-Net 是自然的选择（听众熟悉）；t 通过 time embedding 注入每个残差块。输出是每个像素的速度。

**25. MNIST: flow from noise to digits**（1.3′，K）
- 图 F9：生成轨迹条（t 从 0 到 1）；同一批噪声在 NFE = 1, 2, 5, 10, 100 的样本网格，I-CFM、OT-CFM、DDIM。旁边放真实数字和数据均值。
- 讲法：重点看少步时的差距，以及第 16 页的"均值"现象在图像上是否出现：I-CFM 一步应接近模糊的平均图。

**26. MNIST: quality vs steps**（1.3′，W）
- 图 F11：我们自己测的 FID vs NFE，1 万张，三种方法同一张图，带 real-vs-real 参考线。
- 注明：Inception 特征是 ImageNet 上训练的，用在数字上只看趋势；seed 0，每种方法一个网络。

**27. Does the OT advantage survive in 784 dimensions?**（1.2′，W）
- 对比 2D 和 MNIST：少步时 OT-CFM 相对 I-CFM 的优势，用实测数字说。
- 配对代价：batch 128 下 OT 与随机配对的平均平方距离比值；路径长度 / 直线距离。训练 loss 曲线作为辅助。
- 解释：batch 相对维度很小时，minibatch OT 的配对接近随机配对。这是我们的核心假设 H3。结论必须跟实测走，不预设结果。

### Close（2′）

**28. Project plan**（1′，K）一页放 E1–E4 和课程方法（TSTR、real-vs-fake、PCA/t-SNE、聚类 + χ²），只讲它们回答什么问题。标明已完成和待完成的部分。

**29. Takeaways**（0.7′，W）
1. CFM 把生成变成回归：最小二乘恢复后验均值，后验均值就是正确的速度场。
2. 配对决定路径，路径决定步数。
3. Velocity、noise、score 是同一个量的不同写法；MNIST 上 OT 优势还剩多少，用第 27 页的实测回答。

**30. Questions**（0.2′）放主要参考文献和代码来源（torchcfm）。

### Backup（不计时，答疑用）

| 页 | 内容 | 来源 |
|---|---|---|
| B1 | 边缘场满足连续性方程的完整证明 | Blog Theorem 1 |
| B2 | `L_CFM = L_FM + C` 完整展开 | Blog / NB cell 8 |
| B3 | Lipman 高斯路径（`μₜ = t x₁`，`σₜ = 1 − (1−σ_min)t`），`u = (x₁ − x)/(1−t)` | Blog Example 2 |
| B4 | Rectified Flow 和 reflow | Blog |
| B5 | Schrödinger Bridge CFM 及 NB 的结果 | NB cell 21–22 |
| B6 | 指标定义：W2、MMD、FID，以及 FID 用于 MNIST 的注意事项 | 旧 deck `metrics` |
| B7 | CIFAR-10 文献基准全表 | 旧 deck `expected` |
| B8 | DDIM 在 1–2 步失败的原因 | 旧 deck `sde-ddim` |

### 如果彩排超时

按这个顺序砍：页 5 并入页 6（省 0.5′）→ 页 26 并入页 25（省 0.5′）→ 页 23 并入页 22（省 0.5′）→ 页 15 并入页 18（省 0.5′）。**不要砍第 11、12、13 页。**

---

## 5. 与 tutorial 和库的对照

| Slide | Blog 小节 | NB / 库 |
|---|---|---|
| 5 | Normalizing Flows | — |
| 6–7 | Continuous Normalizing Flows | NB cell 2 |
| 8 | Intuition of CFM | NB cell 3 |
| 9–10 | Modelling Choices, Example 1 | NB cell 4–5，`compute_mu_t`，`sample_xt` |
| 11 | From Conditional to Unconditional Velocity, Theorem 1 | NB cell 6–7，`compute_conditional_flow` |
| 13 | 同上，Loss | NB cell 8 |
| 14 | Summary: Flow Matching In Practice | NB cell 9–11，`ConditionalFlowMatcher` |
| 15–16 | — | NB cell 11–13 |
| 17–18 | Optimal Transport Flow Matching | NB cell 14–20，`ExactOptimalTransportConditionalFlowMatcher`，`OTPlanSampler`；`The_unreasonable_performance_of_minibatch_OT.ipynb` |
| 19 | Fast Sampling with Straight Flows | 我们的 `code/e1_2d/` |
| 20–21 | Diffusion Models；Link Between Diffusion and FM | `VariancePreservingConditionalFlowMatcher` |
| 22–24 | 无（Blog 不涉及图像） | `examples/images/mnist_example.ipynb` |
| 25 | 无 | `examples/images/cifar10/` |

Blog 没有图像内容，也不展开 minibatch OT 的统计性质，这两块是我们在 tutorial 之外要补的。

---

## 6. 例子设计

### 6.1 2D

| 用途 | 数据 | 模型来源 | 说明 |
|---|---|---|---|
| 手算（页 12） | 1D：N(0,1) → ±a | 无模型 | 精确公式 |
| 主线（页 15–18） | 8 Gaussians → moons | 本地 `moon_classical.pt`，`moon_optimal.pt`，`moon_sch.pt`，**不用重训** | 与 NB 完全对应 |
| 对比（页 19） | N(0,I) → 4 个数据集 | `results/e1/`，16 组已完成 | 包含 DDIM、DDPM |

### 6.2 图像

**这是对旧计划的调整**：原计划图像实验全放在 Pre 之后。现在要讲图像例子，所以 Pre 前只做**演示级**的结果，完整的 E2 / E4（多种子、消融）仍放在 Pre 之后。

| | MNIST | CIFAR-10 |
|---|---|---|
| 模型 | 按 `mnist_example.ipynb` 训练 I-CFM、OT-CFM | 官方权重 `icfm`、`otcfm`（400k 步） |
| 现成资源 | `examples/images/models/mnist/ot_cfm_model.pt`（来历待查，notebook 输出的样本偏噪，建议重训） | 本地 `model/cifar_otcfm.pt`，见第 9 节 |
| 耗时（本机 RTX 5070 Ti 实测） | 约 40 秒/epoch，50 epochs 约 35 分钟/模型 | 单步前向约 1,500 张/秒；1 万张 × NFE{1,2,5,10,20,50,100} 约 21 分钟/模型 |
| 产出 | F8、F9、MNIST FID（可选） | F10、F11 |

耗时是在 GPU 约 44% 占用时测的短测试，只算大致估计。CIFAR 的估计未含 FID 计算和磁盘写入。

---

## 7. 图表和实验清单

| 编号 | 内容 | 用于页 | 状态 | 做法 | 负责 |
|---|---|---|---|---|---|
| F1 | 开场动图 | 3 | **已做** | 2D 用自己生成的 `opening_2d.gif`（`slides/make_opening_figs.py`，库里的 GIF 有 24 MB，太大）；CIFAR 用库的 `169_generated_samples_otcfm.gif`，注明来源 | W |
| F2 | ODE 路径和 Euler；连续性方程；多条路径 | 6, 7, 8 | **已做** | `slides/make_transport_figs.py`，全部来自 N(0,1) → ±2 的闭式解，无需训练：`ode_euler.png`，`continuity_1d.png`，`many_paths.png`（第 7 页用它代替 `prob_paths.png`，因为那张图过早出现了 diffusion） | K |
| — | 概率路径图和动图 | 20 | **已有** | `prob_paths.png`，`paths.gif` | — |
| F3 | 一对直线 → 多对 → 密度 | 9, 10 | **已做** | `slides/make_cfm_figs.py` | W |
| F4 | 某点的后验平均方向 | 11 | **已做** | `slides/make_cfm_figs.py` | W |
| F5 | 闭式解的 `pₜ`、箭头、轨迹 | 12 | **已做** | `slides/make_cfm_figs.py` | W |
| F6 | I-CFM / OT-CFM 轨迹 | 15, 18 | **已做** | `slides/make_2d_figs.py`，加载本地 `moon_*.pt`：`icfm_demo.gif`（PDF 用 `.png`），`traj_ot_vs_i.png`。SB-CFM 留给 backup B5 | W |
| F7 | 交叉示意；随机 vs OT 配对 | 16, 17 | **已做** | `slides/make_2d_figs.py`：`bend_1d.png`，`pairs_swap.png`（16 个点，`OTPlanSampler`）；第 19 页另有 `w2_four.png`，`moons_5steps.png` | W |
| — | W2 vs NFE，四数据集 | 19 | **已有** | `w2_all.png`，`datasets_grid.png`，`w2_vs_nfe.png` | — |
| — | 采样动图 | 21 | **已有** | `sampling.gif` | — |
| F8 | 图像的 `xₜ` 和目标 `x₁−x₀` | 23 | 新做 | 几行代码，无需训练 | K |
| F9 | MNIST 轨迹和 NFE 网格 | 24 | 新做 | 训练后采样（`code/e2_mnist/`，目前为空） | K |
| F10 | CIFAR 不同 NFE 的样本 | 25 | 新做 | 官方权重 + Euler，固定噪声 | W |
| F11 | CIFAR FID vs NFE | 25 | 新做 | 建议 1 万张做快速版，标明样本量（不能和 5 万张的文献值直接比） | W |
| F12 | OT 增益 vs 维度 | 26 | 可选 | 需要 F11 和 MNIST FID | W |

---

## 8. 时间线（距 11/12 还有 7 天）

| 日期 | 事项 | 负责 |
|---|---|---|
| 11/05（四） | 确认本规划和第 10 节的决定 | W, K |
| 11/06（五） | 下载 CIFAR 官方权重；跑 F10、F11；MNIST 训练（F9） | W, K |
| 11/06–07 | 写 F2–F8 的作图脚本 | W, K |
| 11/07–08 | 按第 4 节重做 slide 和英文讲稿，**11/08 出初稿**（与 `TEAM_PLAN.md` 一致） | W, K |
| 11/09 | 对照第 9 节检查公式和数字；补 backup | W, K |
| 11/10 | **彩排**并计时，按第 4 节的顺序调整 | W, K |
| 11/11 | 定稿，导出 PDF | W |
| 11/12（四） | 报告 | W, K |

**保底**：如果图像结果没赶上，页 24 用 `mnist_example.ipynb` 现有输出，页 25 用库的 `169_generated_samples_otcfm.png` 和文献 FID，并明确标注来源。

---

## 9. 易错点（做 slide 时逐条核对）

**资料本身的问题**
1. **Blog 的符号错误**：Blog 正文写 Lipman 路径的速度为 `(x − x₁)/(1−t)`，同页总结表写 `(x₁ − x)/(1−t)`。由 `xₜ = t x₁ + (1−t)ε` 可推出**后者正确**。我们用后者。
2. **NB 的 σ**：cell 11、17 里定义了 `sigma = 0.1`，实际调用的是 `sample_conditional_pt(..., sigma=0.01)`；cell 17 的 `FM` 也没有被用到。slide 上若写 σ，写实际值 0.01。
3. **NB 的 W2 数字来自我们本地版本**，不是库原版的内容。引用时说明是我们自己跑的。

**数字不能混用**
4. 页 18（NB）和页 19（E1）**不可比**：前者源分布是 8 Gaussians（scale 5），用 dopri5，1024 样本，100 次重复；后者源分布是标准高斯，固定 NFE，2000 样本，seed 0，数据也标准化过。
5. 损失值不能当质量指标（页 13）。
6. 1 万张的 FID 和文献的 5 万张 FID 不能直接比，要标明。文献值之间的网络和实现也不同。
7. **本地 CIFAR 权重**：`cifar_otcfm.pt` 测得 FID 约 11.8–12.2（`fid_log.json`），远高于官方的约 3.5，配置没有记录。原因查清之前，**slide 上只用官方权重的结果**，这一条 `TEAM_PLAN.md` 也已写明。

**容易讲错的概念**
8. Lipman 的 "OT path"（单对之间）≠ OT-CFM 的 OT 配对（如何配对）。
9. minibatch OT **不是**全局 OT，只是一个近似。OT 配对也不保证学到的流严格为直线。torchcfm 的 `sample_plan` 默认有放回地按 π 重采样，一个 batch 内可能有重复的点。
10. `σ > 0` 时线性路径的 t = 0 边缘分布严格来说不是 `p₀`（Blog 提到）；σ = 0 才严格。
11. 独立配对下**条件**直线会交叉，**边缘**流的轨迹不会交叉。不要说"I-CFM 的轨迹交叉"。
12. 页 21 的 velocity ↔ noise 转换是代数恒等式，对任何源和配对都成立；只有 score 那一步需要 `x₀│x₁` 是高斯，所以它对页 15 的 8 Gaussians 和 OT 配对不成立。
13. Diffusion 和 FM 的等价性（VP 路径）依赖 `exp(−½∫β) ≈ 0` 这一近似，不是严格相等。
14. 页 16 的"一步到均值"针对最优场。训练出来的网络是近似的，所以 W2 是 1.28–1.37 而不是正好 1.41。

**格式**
15. `build_pdf.py` 的 `BLOBS` 要登记新图；GIF 要有静态替代图。字号 ≥ 24px，每页内容高度不超过 824px。

---

## 10. 需要决定的事

| # | 问题 | 建议（默认） | 备选 |
|---|---|---|---|
| 1 | 报告定位 | Tutorial 为主，比较结果和项目计划保留，实验设计压缩成 1 页 | 如果课程评分看重实验设计，把页 27 扩成 3 页，并把页 5 并入页 6、页 23 并入页 22 来腾时间 |
| 2 | 讲者分配 | 默认 W 约 22′，K 约 13′ | 想更均衡：把 Part III 的页 17–19 给 K，变成 W 约 19′ / K 约 16′ |
| 3 | 图像部分做到什么程度 | MNIST 自己训练；CIFAR 用官方权重，1 万张快速 FID | 时间紧：用第 8 节的保底方案 |
| 4 | MNIST 是否加 DDIM 对比 | 可选，取决于 Kaiming 的时间 | 需要额外训练一个 ε 模型，约 35 分钟，加上写 DDIM 采样 |
| 5 | Rectified Flow | 主体一句话，细节放 backup | 若老师特别关注，加一页 reflow 结果 |

**旧 deck 的处理**：线上版本和 PDF 暂时保留，直到新 deck 完成。可复用的内容：页 `pseudo`（放 backup）、`metrics`、`expected`、`sde-ddim`、`course-eval`（压缩进页 27），以及现有配色和版式。

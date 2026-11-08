# TEAM PLAN · Flow Matching vs. Diffusion（Data Mining Final Project）

> **本文件是项目的唯一总计划（single source of truth）。** 队友和队友的 Claude 读这一份就能开工；`docs/` 下其他文件是补充细节。
> 最后更新：2026-11-07（新增下面的“当前进度”；完整实验改以 [docs/FINAL_REPORT_PLAN.md](docs/FINAL_REPORT_PLAN.md) 为准）

---

## 当前进度（2026-11-07）

- **课堂报告**：改到 **11/10（周二）**，30 分钟，两人讲。37 页（正文 31 + backup 6）已定稿：`slides/FM_vs_Diffusion_Talk.pptx`（带讲者备注）、`slides/FM_vs_Diffusion_Talk.pdf`、讲稿 `slides/FM_vs_Diffusion_Speech_Script.docx`。规划见 `docs/SLIDE_PLAN.md`。
- **已完成的实验**（数字见 `docs/FINAL_REPORT_PLAN.md` §1）：
  - E1 2D：4 个数据集 × I-CFM / OT-CFM / DDIM / DDPM，种子 0。
  - E2 MNIST：I-CFM / OT-CFM / DDPM（DDIM 采样）× 2 个训练种子，FID vs NFE。
  - E3 课程方法：k-means + χ²（课程只要求一种方法），两个种子；TSTR、real-vs-fake、PCA 只做了预实验。
  - E4：CIFAR-10 官方权重已下载，未正式评估。
- **报告之后（11/11–12/02）**：做 `docs/FINAL_REPORT_PLAN.md` 中的 R1–R7。R1（第三个种子）和 R2（E3 统计分析）是必做；R3–R5 是体现复杂度的主要实验；R6、R7 选做。下面 §5 是 10/29 的原始设想，保留作参考。

---

## 0. 给读者的说明（人和 Claude 都请先读）

**项目一句话**：从统计视角比较两类“免模拟（simulation-free）”的生成模型——**Flow Matching**（I-CFM / OT-CFM / Rectified Flow）和 **Diffusion**（DDPM / DDIM）。先在 2D 数据上看清机制，再在高维图像（MNIST、CIFAR-10）上比较“生成质量 vs 采样步数（NFE）”，最后用课程方法（线性分类、树/SVM、聚类、降维）评估生成数据。

**两个硬截止**
| 事项 | 时间 | 内容 |
|---|---|---|
| **Pre（项目汇报，30 分钟）** | **2026-11-10（周二）**（原定 11/12） | **以“方法 + 实验设计”为主**：文献脉络、核心公式、算法、实验怎么搭、评估方法、高维下的预期结果（引用文献数字）；只用已完成的 2D 示例做演示。完整实验放到 Pre 之后再做 |
| **Final report** | **2026-12-16 23:59** | 约 5 页，技术论文格式，**必须写明每个人的贡献** |

**如果你是 Claude（队友的 AI 助手）**：先读完第 0、1、2 节和第 10 节的工作规则，然后找到你的用户是 **A** 还是 **B**（第 1 节），从第 3 节 Step 0 开始，按阶段顺序执行属于该成员的任务。每完成一项，就在本文件中把 `[ ]` 改为 `[x]` 并写上日期。

---

## 1. 分工

| | 成员 A：**Wentao Yang** | 成员 B：**Kaiming Zhang** |
|---|---|---|
| 算法线 | **Flow Matching**：I-CFM、OT-CFM、Lipman FM、Rectified Flow | **Diffusion**：DDPM、DDIM、Score SDE 视角 |
| 评估线 | 降维（PCA / t-SNE / UMAP）+ 聚类（k-means / GMM / χ² 检验） | 分类器评估（TSTR：LogReg / LDA / RF / SVM）+ real-vs-fake 判别器 + 统计基线（GMM / KDE） |
| 高维 | CIFAR-10（已有 OT-CFM 权重） | MNIST 上的 DDPM / DDIM（与 A 用同一个 UNet） |
| Pre | 背景、FM 公式、2D 结果、CIFAR | Diffusion 公式、统一视角、实验设计、评估方法 |
| Report | Intro、Methods（FM）、E1/E2 | Related Work、Methods（Diffusion）、E3 |

共同负责：公共代码 `code/common/`（一人写、另一人 review）、预期结果表、结论、个人贡献一节。

---

## 2. 阶段总览

```
阶段 P：Pre 准备（10/29–11/12）   → 阅读 + 公式 + 算法 + 实验设计 + 文献基准 + 2D 演示 → 11/12 Pre
阶段 E：完整实验（11/13–12/02）    → E1 2D 完整网格 → E2 MNIST 主实验 → E3 课程方法评估 → E4 CIFAR / 一步生成
阶段 R：Report（12/03–12/16）      → 初稿 → 交叉审阅 → 12/16 提交
```

| 周 | 日期 | 阶段 | 里程碑 |
|---|---|---|---|
| W1 | 10/29–11/04 | P | 环境、精读核心论文、跑通 2D 起步代码、写公式与算法笔记 |
| W2 | 11/05–11/11 | P | 实验设计定稿、文献基准表、**slides 初稿 11/08 → 彩排 11/10 → 定稿 11/11** |
| **Pre** | **11/12（周四）** | P | **30 分钟汇报：方法 + 实验设计** |
| W3 | 11/13–11/18 | E | 根据 Pre 反馈调整设计；E1 完整网格；MNIST 训练管线 |
| W4 | 11/19–11/25 | E | E2 MNIST 主实验（多种子 + NFE 扫描） |
| W5 | 11/26–12/02 | E | E3 课程方法评估 + E4 |
| W6 | 12/03–12/09 | R | Report 初稿 + 交叉审阅 |
| W7 | 12/10–12/16 | R | 定稿，**12/16 23:59 前提交** |

---

## 3. Step 0：获取项目与配置环境（两人都做，第 1 天）

### 3.1 项目目录

```
cfm_project/
├── TEAM_PLAN.md                     ← 本文件（总计划）
├── README.md                        项目入口
├── docs/                            补充细节：01 计划、02 周清单、03 参考资料、notes/ 笔记
├── papers/                          论文 PDF（不进 git；用 scripts/download_papers.sh 下载）
├── code/                            我们自己的代码（详见 code/README.md）
│   ├── common/                      数据、网络、指标、画图、实验骨架（两人共用）
│   └── e1_2d/                       train_fm.py（A）、train_ddpm.py（B）、summarize.py
├── conditional-flow-matching-main/  torchcfm 原仓库（只读参考，不要改）
├── data/                            数据集（不进 git，脚本自动下载）
├── results/                         实验输出（图、表、json；权重不进 git）
├── report/  slides/                 LaTeX 与 PPT
└── scripts/                         下载、批量运行等辅助脚本
```

### 3.2 获取代码
- [x] 由 Wentao 建 GitHub 私有仓库（10/29）：**https://github.com/liliyawanttosleep/cfm-project**。协作流程见 `CONTRIBUTING.md`
- [ ] 把队友加为 collaborator（仓库 Settings → Collaborators，或者把队友的 GitHub 用户名告诉 Claude）
- [ ] 队友 `git clone https://github.com/liliyawanttosleep/cfm-project.git cfm_project`；然后在仓库里执行 `git clone https://github.com/atong01/conditional-flow-matching conditional-flow-matching-main`
- [ ] 下载论文：`bash scripts/download_papers.sh`（下载到 `papers/`）

### 3.3 Python 环境（按自己的显卡选 CUDA 版本）

```bash
conda create -n flow python=3.12 -y
conda activate flow
# 到 https://pytorch.org/get-started/locally/ 选择与显卡匹配的命令；RTX 50 系显卡需要 CUDA 12.8 及以上
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e conditional-flow-matching-main
pip install pot torchdyn torchdiffeq clean-fid umap-learn scikit-learn pandas matplotlib
python -c "import torch, torchcfm, ot; print(torch.__version__, torch.cuda.is_available())"   # 必须输出 True
```

> Wentao 的电脑上已有可用环境：`D:\Anaconda\envs\flow`（torch 2.9 + CUDA 13，RTX 5070 Ti 12GB）。系统默认的 `python` 是 CPU 版，不要用它跑实验。

### 3.4 冒烟测试（两人都要跑通）

```bash
python code/e1_2d/train_fm.py   --method otcfm --data moons --iters 2000 --out results/_smoke
python code/e1_2d/train_ddpm.py --method ddim  --data moons --iters 2000 --out results/_smoke
```
**通过标准**：两个脚本都能跑完，并在 `results/_smoke/` 生成 `.json` 和 `.png`；png 里 NFE ≥ 10 的子图能看出两个月牙形。

---

## 4. 阶段 P：Pre 准备（10/29–11/12，优先级最高）

**11/12 的 Pre 以“方法 + 实验设计”为主，不要求实验结果。** 要讲清楚 5 件事：
① 文献脉络 ② 核心公式 ③ 算法 ④ 实验怎么搭（数据、模型、公平对比、指标、课程方法评估）⑤ 高维下的预期结果（引用文献数字，加上我们的假设）。
唯一的“结果”是已经完成的 **2D 演示**，用来直观展示方法的差别。MNIST、CIFAR 等完整实验放到 Pre 之后的阶段 E。

| 截止 | 任务 |
|---|---|
| 11/04（周三） | P1 读完必读论文 + 笔记；两人都跑通 2D 代码 |
| 11/06（周五） | P2 方法讲解材料、P3 实验设计定稿 |
| 11/08（周日） | **slides 初稿**（各自负责的页） |
| 11/10（周二） | **彩排**，掐时间，互相提问 |
| 11/11（周三） | 定稿 |
| **11/12（周四）** | **Pre** |

### P1. 精读文献（截止 11/04）
论文都在 `papers/`，编号见第 9 节。每篇读完在 `docs/notes/` 的笔记中写 3–5 行：解决什么问题、核心公式、主要结果。

| 优先级 | A 读 | B 读 | 两人都读 |
|---|---|---|---|
| 必读 | 20 Lipman FM（§2–4）、21 Rectified Flow（§2–3）、23 OT-CFM（§3–4） | 10 DDPM（§2–4）、11 DDIM（§4–5）、13 Score SDE（§2–4） | 00 Luo 综述（§1–3）、01 FM Guide（§1–4） |
| 写 Pre 用 | 25 Neural ODE（只看 §1–3）、22 Stochastic Interpolants（只看 abstract 和 §1） | 12 NCSN（只看 §2）、14 Improved DDPM（cosine schedule 一节） | 40 FID |
| 了解即可 | 32 MeanFlow | 15 CFG、16 EDM | 30 SD3（动机页讲意义用） |

- [ ] A：完成必读 + 笔记 `docs/notes/W1_成员A_FlowMatching笔记.md`
- [ ] B：完成必读 + 笔记 `docs/notes/W1_成员B_Diffusion笔记.md`
- [ ] 共同：读综述 00 和 01

### P2. 方法讲解材料（截止 11/06）——Pre 的核心
- [ ] A：Flow Matching 部分。CFM 公式推导（重点讲“FM loss 无法计算 → CFM loss 梯度相同”的定理，用“条件期望 = 回归最优解”这个统计直觉来讲）；三种路径与耦合（I-CFM / Lipman FM / OT-CFM）；Rectified Flow 的直线度与 reflow
- [ ] B：Diffusion 部分。前向加噪 → ε-预测 loss → 祖先采样；ε 与 score 的关系；VP-SDE 与 probability flow ODE；DDIM 是这个 ODE 的离散化
- [ ] B：**统一视角对比表**（一页）：

  | | DDPM | DDIM | I-CFM / RF | OT-CFM |
  |---|---|---|---|---|
  | 网络学什么 | 噪声 ε | 噪声 ε（同一个模型） | 速度 v | 速度 v |
  | 训练目标 | ε 的 MSE | 同左 | (x1−x0) 的 MSE | 同左，配对用 OT |
  | 采样 | SDE（随机） | ODE（确定） | ODE | ODE |
  | 路径 | 弯曲 | 弯曲 | 较弯 | 近似直线 |
  | 典型 NFE | 1000 | 20–100 | 20–100 | 少步也可用 |
- [ ] B：两种算法训练和采样的伪代码，左右并排（直接对照 `code/e1_2d/train_fm.py` 与 `train_ddpm.py` 写）
- [ ] 公式全部取自第 8 节，统一记号：t=0 为噪声、t=1 为数据（FM 的约定）；讲 DDPM 时说明它的 t 方向相反

### P3. 实验设计定稿（截止 11/06）——Pre 的第二个核心
在 `docs/experiment_design.md` 中写定下面这些内容（两人各写一半，互相 review），之后 slides 直接从这里取材：
- [ ] B：**研究问题 Q1–Q4**（见 `docs/01_项目计划.md`）与实验 E1–E4 的对应表
- [ ] B：**实验总表**：每个实验的数据集、方法、变化的变量（NFE、耦合、噪声调度）、固定的变量
- [ ] B：**公平对比协议 + 超参数表**：同一网络（2D 用 MLP 3×256；图像用 torchcfm UNet）、同样的训练步数 / batch / lr / EMA、同一份采样噪声、同一份测试集、3 个种子
- [ ] B：**指标定义**：W2、MMD、FID（讲清楚 FID 就是两个拟合高斯之间的 W2²）、直线度；FID 的样本数和实现（clean-fid）
- [ ] A：**课程方法评估流程图**：生成样本 → 分类器打标签 → ① TSTR（LogReg / LDA / RF / SVM）② real-vs-fake 判别器 ③ PCA / t-SNE / UMAP ④ k-means / GMM + χ² 检验，说明每一步回答哪个研究问题
- [ ] A：**预期结果**：第 7 节文献基准表 + 假设 H1–H4，每条假设写清楚“用哪个实验、哪个指标来检验”

### P4. 2D 演示（截止 11/06，起步代码已写好，moons 上已验证）
已有结果（`results/e1/`，moons，seed 0，W2 越小越好；两份独立真实样本之间的 W2 = 0.043，是有限样本下的下限）：

| NFE | 1 | 5 | 20 | 100 |
|---|---|---|---|---|
| OT-CFM | **0.12** | **0.089** | 0.092 | 0.092 |
| I-CFM | 1.32 | 0.30 | 0.12 | 0.10 |
| DDIM | 9.2 | 0.56 | 0.19 | 0.17 |
| DDPM（1000 步） | — | — | — | 0.148 |

- [ ] A：读懂并跑通 `code/e1_2d/train_fm.py`，补跑 `8gaussians`（seed 0）；准备两张 Pre 用图：轨迹对比（OT-CFM 直线 vs I-CFM 弯曲）、样本随 NFE 的变化
- [ ] B：读懂并跑通 `code/e1_2d/train_ddpm.py`，补跑 `8gaussians`；**准备一页讲 DDIM 在 1–2 步时为什么失败**（t=T−1 时 ᾱ≈4e-5，x̂0 = (x−√(1−ᾱ)ε)/√ᾱ，误差被放大约 150 倍）
- [ ] 共同：`python code/e1_2d/summarize.py` → `results/e1/w2_vs_nfe_*.png` 作为演示主图

### P5（可选，有空再做）：CIFAR-10 预览
如果 11/08 前所有任务都已完成，A 可以用官方权重测几个 NFE 下的 FID，放到“预期结果”页作为“初步验证”。没时间就跳过，这部分会在 E4 完整做。
```bash
cd conditional-flow-matching-main/examples/images/cifar10
# compute_fid.py 读取 {input_dir}/{model}/{model}_cifar10_weights_step_{step}.pt
mkdir -p ../../../../results/e4/weights/otcfm
curl -L -o ../../../../results/e4/weights/otcfm/otcfm_cifar10_weights_step_400000.pt \
  https://github.com/atong01/conditional-flow-matching/releases/download/1.0.4/otcfm_cifar10_weights_step_400000.pt
W=../../../../results/e4/weights
for n in 2 10 100; do   # 10k 样本，FID 会比 50k 时偏高，需注明
  python compute_fid.py --input_dir $W --model otcfm --integration_method euler --integration_steps $n --num_gen 10000
done
```
本地还有一份旧权重 `model/cifar_otcfm.pt`（以前在 50k 样本下测得 FID ≈ 11.8–12.2，但设置没有记录），E4 前要查清楚它是怎么来的；查不清就只用官方权重。

### P6. Pre 幻灯片（约 22 页，讲 28 分钟 + 2 分钟问答）
**初稿已完成（10/29）**：https://claude.ai/artifact/GRZZAyn8ir3RXQE1Hc8P5e （在线 Slides，可导出 PPTX/PDF；需要 Wentao 在 Share 菜单中共享后队友才能打开）。每页的备注里有中文讲稿，并标明了由谁讲。幻灯片用图在 `slides/figs/`，用 `python slides/make_slide_figs.py slides/figs` 可以重新生成。待补：成员名字（`[Member A]`、`[Member B]`），以及 15 页图像实验的训练步数 `[__]`。

| # | 部分 | 页面 | 内容与图 | 负责 | 时长 |
|---|---|---|---|---|---|
| 1 | | 标题 | 题目、成员 | A | – |
| 2 | 引言 | 动机 | 生成模型的意义；SD3、Flux 基于 Rectified Flow；统计视角：从样本学分布，再采样 | A | 2′ |
| 3 | | 提纲 | 背景 → 方法 → 实验设计 → 演示与预期 | A | 0.5′ |
| 4 | **背景** | 文献脉络 | 时间轴：NCSN 2019 → DDPM 2020 → DDIM / Score SDE 2021 → FM / RF / SI 2023 → OT-CFM / SD3 2024 → MeanFlow 2025 | B | 1.5′ |
| 5 | | 统一框架 | p0 = N(0, I) → p1；ODE dx/dt = v(x, t)；连续性方程；**NFE 的概念**（生成一个样本要调用几次网络） | A | 1.5′ |
| 6 | **方法：Diffusion** | 前向与训练 | q(x_t \| x_0)、ε-预测 loss；加噪示意图 | B | 1.5′ |
| 7 | | 采样与 score | 祖先采样公式；ε ↔ score | B | 1.5′ |
| 8 | | SDE 与 DDIM | VP-SDE、PF-ODE；DDIM = ODE 离散化，可以少步采样 | B | 1.5′ |
| 9 | **方法：FM** | CFM | FM loss 无法计算 → CFM；梯度相同的定理（条件期望 = 回归最优解） | A | 2′ |
| 10 | | 路径与耦合 | I-CFM / Lipman / OT-CFM；直线度；reflow | A | 2′ |
| 11 | **方法：对比** | 伪代码 | FM 与 DDPM 左右并排 | B | 1′ |
| 12 | | 统一对比表 | P2 中的表 | B | 1′ |
| 13 | **实验设计** | 研究问题 | Q1–Q4 | A | 1′ |
| 14 | | 实验总表 | E1–E4：数据、方法、变量 | B | 1.5′ |
| 15 | | 公平对比协议 | 同一网络、同样预算、同一份噪声、3 个种子；超参数表 | B | 1′ |
| 16 | | 指标 | W2、MMD、FID（= 高斯 W2）、直线度 | B | 1.5′ |
| 17 | | 课程方法评估 | 流程图：TSTR / 判别器 / 降维 / 聚类 + χ² | A | 1.5′ |
| 18 | **演示与预期** | 2D 演示 | 轨迹对比图 + 样本随 NFE 的变化 | A | 1.5′ |
| 19 | | 2D 定量结果 | W2 vs NFE 曲线；DDIM 少步失败的原因 | A | 1′ |
| 20 | | 高维预期（文献） | 第 7.1 节 CIFAR-10 基准表（标出 NFE） | A | 1.5′ |
| 21 | | 我们的假设 | H1–H4 及其检验方式 | B | 1′ |
| 22 | | 计划与分工 | 后续时间线（E1–E4、report） | B | 0.5′ |
| 23 | | Q&A | | 共同 | 2′ |

A 讲约 14.5 分钟，B 讲约 13.5 分钟。

- [ ] 各自负责的页，初稿 11/08 完成
- [ ] 11/10 彩排：控制在 28 分钟内；互相提问，准备可能被问到的问题（例：“FM 和 diffusion 本质上有什么区别？”“为什么 OT 在高维时效果变弱？”“FID 用在 MNIST 上合适吗？”）
- [ ] 11/11 定稿；Pre 后在 `docs/meetings.md` 中记录老师和同学的反馈，W3 开始时据此调整实验设计



---

## 5. 阶段 E：完整实验（W3–W5）

> **11/07 起以 [docs/FINAL_REPORT_PLAN.md](docs/FINAL_REPORT_PLAN.md) 为准。** 本节是 10/29 的原始设想：其中 E2 的两个种子、E3 的 k-means 已经完成；TSTR、判别器、降维改为附录中的预实验；EMA、cosine 调度、classifier-free guidance、MNIST-FID 暂不做。

所有实验遵守**公平对比约定**：同一网络结构、同样的训练步数 / batch / lr / EMA、同一份采样噪声、同一份测试集、**3 个种子，报告均值 ± 标准差**。

### E1. 2D 完整网格（W3 前半）
- 数据：moons、8gaussians、swissroll、checkerboard；方法：icfm、otcfm、fm_lipman、ddim、ddpm；种子 0/1/2
- 运行：`code/README.md` 中有批量循环命令（每个 run 约 1.5–5 分钟，全部约 3–4 小时，可以放到后台跑）
- [ ] A 跑 FM 的三种方法，B 跑 DDPM/DDIM 和统计基线
- [ ] B：统计基线 `code/e1_2d/baselines.py`：GMM（按 BIC 选成分数）、KDE（交叉验证选带宽）；计算 W2、MMD、测试集 NLL
- [ ] 交付：`results/e1/summary.csv` + 各数据集的 `w2_vs_nfe_*.png`
- **验收**：OT-CFM 在 NFE ≤ 2 时的 W2 明显低于 I-CFM 和 DDIM；所有方法在 NFE = 100 时都接近参考值的 2–4 倍以内

### E2. MNIST 主实验（W3 后半 – W4）
- [ ] 第一步先做可行性验证：B 写 `code/e2_mnist/train_ddpm.py`（复用 2D 版的 `Schedule`、`ddpm_sample`、`ddim_sample`，网络换成 torchcfm 的 UNet，配置取自 `examples/images/mnist_example.ipynb`）；A 写 `code/e2_mnist/train_fm.py`，用**完全相同**的 UNet。各训练约 20k 步，保存 8×8 样本网格
- [ ] 共同：`code/common/data.py`（MNIST / Fashion-MNIST，归一化到 [-1, 1]）
- [ ] B：`code/common/fid.py`（clean-fid）+ “MNIST-FID”（用自己训练的 LeNet 的倒数第二层特征）—— A review
- [ ] 共同：训练 MNIST 分类器（准确率约 99%，存到 `results/e2/classifier.pt`），用来打伪标签，E3 也会用
- [ ] A：I-CFM、OT-CFM × 3 种子；NFE ∈ {1, 2, 5, 10, 20, 50, 100} 和 dopri5
- [ ] B：DDPM（1000 步）、DDIM × 3 种子，同样的 NFE 网格；消融：linear vs cosine 噪声调度
- [ ] 加分（A）：条件生成 + classifier-free guidance，w ∈ {0, 1, 2, 4}（参考 `examples/images/conditional_mnist.ipynb`）
- [ ] 共同：Fashion-MNIST 上重复最好的设定
- [ ] 交付：**FID vs NFE 曲线**（FM 和 Diffusion 在同一张图，带误差带）+ 训练 / 采样时间表

### E3. 课程方法评估（W5）——满足课程硬性要求
每个模型生成 60k 张图，用 E2 的分类器打标签。脚本要能接受任意模型的样本目录作为输入。

| 课程要求 | 负责 | 方法 | 输出 |
|---|---|---|---|
| 线性分类 | B | Logistic Regression（L2）、LDA | TSTR 与 TRTR 准确率 |
| 树 / SVM | B | Random Forest、RBF-SVM | 同上；real-vs-fake 判别器的准确率和 AUC（越接近 0.5 越好） |
| 降维 | A | PCA（解释方差曲线对比）、t-SNE、UMAP | 真实样本与生成样本画在同一张图 |
| 聚类 | A | k-means（k=10）、GMM | ARI / NMI；各簇占比的 χ² 均匀性检验（检查 mode collapse） |

- [ ] B：`code/e3_eval/tstr.py`，另画“准确率 vs 生成样本数”曲线（1k / 5k / 10k / 60k）
- [ ] A：`code/e3_eval/dimred.py`、`code/e3_eval/cluster.py`
- [ ] 交付：`results/e3/` 中的表和图

### E4. 扩展（W5，二选一，由 A 负责）
- [ ] CIFAR-10：50k 样本下完整的 NFE vs FID 曲线，与第 7 节文献值对比
- [ ] 或者 MNIST 上的 2-Rectified Flow（reflow）一步生成，比较 1 步时的 FID

---

## 6. 阶段 R：Report 撰写（W6–W7）

格式：NeurIPS/ICML LaTeX 模板，在 Overleaf 上协作，约 5 页正文加参考文献。

| 章节 | 负责 | 篇幅 | 要点 |
|---|---|---|---|
| Abstract + Introduction | A | 0.5 页 | 意义、统计视角、贡献列表 |
| Related Work | B | 0.5 页 | 第 9 节的文献脉络 |
| Methods：FM / OT-CFM / RF | A | 0.75 页 | CFM 定理、耦合方式 |
| Methods：DDPM / DDIM / SDE / 统一视角 | B | 0.75 页 | ε ↔ score、PF-ODE |
| Experiments：设定 + E1 + E2 | A 主笔，B 补 Diffusion 部分 | 1.25 页 | 表 1：2D 指标；图 1：FID vs NFE |
| Experiments：E3（+E4） | B 主笔，A 补降维与聚类部分 | 0.75 页 | 表 2：TSTR；图 2：UMAP |
| Conclusion + 局限 | 共同 | 0.25 页 | |
| **Individual Contributions** | 共同 | 0.25 页 | **课程硬性要求**，按第 1 节分工如实写 |

- [ ] 初稿（12/07 前）→ 交叉审阅（A 审 B 的部分，B 审 A 的部分）→ 修改
- [ ] 检查清单：每张图表都在正文中被引用；图表有编号、坐标轴有标签和单位；均值 ± 标准差；参考文献完整；页数约 5 页
- [ ] **12/15 定稿，12/16 23:59 前提交 PDF**
- [ ] 代码整理：`README.md` 中写明每张图对应的复现命令

---

## 7. 预期结果（Pre 的“高维预期”页 + Report 的对照组）

### 7.1 文献基准：CIFAR-10 无条件生成 FID（越低越好；均已对照 `papers/` 原文核实）

| 方法 | 采样器 / NFE | FID | 出处 |
|---|---|---|---|
| DDPM | 祖先采样，1000 步 | 3.17 | 10 DDPM，Table 1 |
| DDIM (η=0) | 10 / 20 / 50 / 100 / 1000 步 | 13.36 / 6.84 / 4.67 / 4.16 / 4.04 | 11 DDIM，Table 1 |
| DDPM（同样的步数，η=1） | 10 / 100 步 | 367.43 / 9.99 | 11 DDIM，Table 1 |
| FM w/ OT（Lipman） | dopri5，NFE 142 | 6.35 | 20 FM，Table 1 |
| OT-FM（复现） | Euler 100 / 1000 步 | 4.44 / 3.74 | 23 OT-CFM，Table 5 |
| I-CFM | dopri5，NFE ≈ 146 | 3.66 | 23 OT-CFM，Table 5 |
| OT-CFM | dopri5，NFE ≈ 134 | 3.58 | 23 OT-CFM，Table 5 |
| 1-Rectified Flow | RK45，NFE 127 | 2.58 | 21 RF，Table 1 |
| 1-Rectified Flow | **1 步 Euler** | 378 | 21 RF，Table 1（轨迹弯，1 步失败） |
| 2-Rectified Flow（reflow 一次） | **1 步 Euler** | 12.21 | 21 RF，Table 1（拉直后 1 步可用） |
| 2-Rectified Flow + Distill | **1 步** | 4.85 | 21 RF，Table 1 |
| torchcfm 仓库 README | OT-CFM 400k 步 | 约 3.5 | `examples/images/cifar10/README.md` |

> 注意：不同论文的网络、训练步数、FID 实现方式不同，**数字不能严格横向比较**；我们只看趋势。我们自己的 FID 用 clean-fid，计算设定要写进 report。

### 7.2 我们的假设（Pre 中作为“预期结果”，实验完成后逐条验证）
- **H1（少步优势）**：同一网络、同样的预算下，NFE ≤ 10 时，FID 从低到高依次为 OT-CFM < I-CFM < DDIM ≪ DDPM（随机采样）。依据：OT 耦合让轨迹更直，而直线只需 1 步 Euler 就能走完。2D 结果已经支持这一点。
- **H2（多步收敛）**：NFE ≥ 100 时各方法差距缩小（CIFAR 上都在 3.5–4.5 之间）。原因是它们本质上都在学同一个边缘分布的输运。
- **H3（高维下 OT 优势变弱）**：随着维度升高，minibatch OT 配对越来越接近随机配对（batch 相对于维度太小），所以 OT-CFM 相对 I-CFM 的优势在 CIFAR 上比在 2D 上小：OT-CFM 论文中，dopri5 下 OT-CFM 和 I-CFM 的 FID 分别为 3.58 和 3.66，几乎相同；而在我们的 2D 实验中，1 步时两者的 W2 相差约 10 倍。这是一个有统计味道的讨论点（维数灾难下的经验 OT）。在高维中要做到少步生成，需要 reflow 或蒸馏（RF 论文：1 步 FID 从 378 降到 12.21，再降到 4.85）。
- **H4（数据效用）**：MNIST 上，TSTR 准确率比 TRTR 低几个百分点。FID 更低的模型，TSTR 更高、判别器 AUC 更接近 0.5；聚类各簇占比接近均匀（χ² 检验不显著），说明没有 mode collapse。

---

## 8. 核心公式速查

**问题设定**：给定数据分布 $p_1$ 的样本，学一个向量场 $v_\theta(x,t)$，使 ODE $\frac{dx}{dt}=v_\theta(x,t)$ 把 $p_0=\mathcal N(0,I)$ 推到 $p_1$。密度演化满足连续性方程 $\partial_t p_t + \nabla\cdot(p_t v_t)=0$。

**Flow Matching**
- FM 目标（无法直接计算）：$\mathcal L_{FM}=\mathbb E_{t,\,x\sim p_t}\|v_\theta(x,t)-u_t(x)\|^2$
- 条件路径（直线）：$x_t=(1-t)x_0+t\,x_1+\sigma\epsilon$，条件速度 $u_t(x\mid x_0,x_1)=x_1-x_0$
- CFM 目标：$\mathcal L_{CFM}=\mathbb E_{t,\,(x_0,x_1)\sim q,\,x_t}\|v_\theta(x_t,t)-(x_1-x_0)\|^2$
- **定理**：$\nabla_\theta\mathcal L_{CFM}=\nabla_\theta\mathcal L_{FM}$，因为 $u_t(x)=\mathbb E[u_t(x\mid z)\mid x_t=x]$，最优解是条件期望（回归意义下的最优预测）
- 耦合方式：I-CFM 取 $q=p_0\otimes p_1$；OT-CFM 取 $q=\pi^*$（minibatch 上 $W_2$ 的最优传输计划）；Lipman FM 取 $\mu_t=t x_1$，$\sigma_t=1-(1-\sigma_{\min})t$
- 采样：Euler 法 $x_{k+1}=x_k+\frac1N v_\theta(x_k,k/N)$，NFE = N

**Diffusion（DDPM）**
- 前向：$q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$，$\bar\alpha_t=\prod_{s\le t}(1-\beta_s)$
- 训练：$\mathcal L_{simple}=\mathbb E_{t,x_0,\epsilon}\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\,t)\|^2$
- 与 score 的关系：$\nabla_x\log p_t(x)\approx-\epsilon_\theta(x,t)/\sqrt{1-\bar\alpha_t}$（denoising score matching）
- 祖先采样：$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta\big)+\tilde\sigma_t z$，$\tilde\sigma_t^2=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$
- DDIM（η=0）：$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta}{\sqrt{\bar\alpha_t}}$，$x_{s}=\sqrt{\bar\alpha_s}\hat x_0+\sqrt{1-\bar\alpha_s}\,\epsilon_\theta$
- VP-SDE：$dx=-\tfrac12\beta(t)x\,dt+\sqrt{\beta(t)}\,dW$；probability flow ODE：$dx=\big[-\tfrac12\beta(t)x-\tfrac12\beta(t)\nabla_x\log p_t(x)\big]dt$

**评估指标**
- $W_2(\mu,\nu)^2=\min_{\pi\in\Pi(\mu,\nu)}\mathbb E_\pi\|x-y\|^2$（2D 上用 POT 的精确 EMD 计算）
- $\mathrm{MMD}^2=\mathbb E k(x,x')+\mathbb E k(y,y')-2\mathbb E k(x,y)$（多带宽 RBF 核，无偏估计）
- $\mathrm{FID}=\|\mu_r-\mu_g\|^2+\mathrm{Tr}\big(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\big)$，**即 Inception 特征上拟合的两个高斯之间的 $W_2^2$**
- 直线度（Rectified Flow）：$S=\int_0^1\mathbb E\|(x_1-x_0)-v(x_t,t)\|^2dt$，0 表示完全直线

---

## 9. 阅读材料与开源库

论文（`papers/` 中的编号；下载链接与完整列表见 `docs/03_参考资料.md`）：

| 编号 | 论文 | 编号 | 论文 |
|---|---|---|---|
| 00 | Luo 2022，Diffusion 统一视角综述 | 20 | Lipman 2023，Flow Matching |
| 01 | Lipman 2024，FM Guide and Code | 21 | Liu 2023，Rectified Flow |
| 10 | Ho 2020，DDPM | 22 | Albergo 2023，Stochastic Interpolants |
| 11 | Song 2021，DDIM | 23 | Tong 2024，OT-CFM |
| 12 | Song 2019，NCSN | 24 | Tong 2024，[SF]²M |
| 13 | Song 2021，Score SDE | 25 | Chen 2018，Neural ODE |
| 14 | Nichol 2021，Improved DDPM | 30 | Esser 2024，SD3 |
| 15 | Ho 2022，CFG | 31 | Song 2023，Consistency Models |
| 16 | Karras 2022，EDM | 32 | Geng 2025，MeanFlow |
| 40 | Heusel 2017，FID | 41 | Kingma 2014，VAE |

博客与课程：Lilian Weng *What are Diffusion Models?*；Cambridge MLG *An Introduction to Flow Matching*；MIT 6.S184 讲义 https://diffusion.csail.mit.edu/

开源库：**torchcfm**（主力，https://github.com/atong01/conditional-flow-matching）、facebookresearch/flow_matching、HF diffusers（DDPM/DDIM 参考实现）、POT、torchdyn/torchdiffeq、clean-fid、scikit-learn/umap-learn。

---

## 10. 工作规则（人和 Claude 都要遵守）

1. **环境**：只在 `flow` conda 环境里跑实验，并确认 `torch.cuda.is_available()` 为 True。
2. **不改上游代码**：`conditional-flow-matching-main/` 只读；需要改动时复制到 `code/` 再改。
3. **公共代码**：改 `code/common/` 前先在群里说一声；接口（函数名、参数）一旦确定，不能悄悄改。
4. **输出位置**：实验输出放 `results/<实验名>/`；每个实验目录放一个 `README.md`，记录**运行命令、日期、git commit、关键结论**。
5. **长任务**：训练超过几分钟就放到后台运行，并写日志文件；不要在同一块 GPU 上同时跑两个大任务。
6. **结果诚实**：只报告真实跑出来的数字；失败、异常和没做完的部分如实写明。不要编造结果，也不要把文献数字当成我们的结果。
7. **进度同步**：完成一项就在本文件中把 `[ ]` 改成 `[x]` 并写上日期；每周末两人对一次进度（15 分钟），结论写进 `docs/meetings.md`。
8. **Git**：每人在自己的分支上开发（`a/<功能>`、`b/<功能>`），合并到 main 前请对方看一眼；`papers/`、`data/`、`*.pt` 不提交。
9. **遇到问题**：先查 `code/README.md` 和本文件；还不行就记到 `docs/meetings.md` 的“待讨论”中。

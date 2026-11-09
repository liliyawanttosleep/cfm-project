# 交接文档（2026-11-07）

> 新对话先读本文件，再按需读：`TEAM_PLAN.md`（开头的“项目全流程”）、`docs/FINAL_REPORT_PLAN.md`（实验 C1–C7）、`report/README.md`（report 格式）、`docs/notes/derivations.md`（公式推导 D1–D9）。
> 取代 10/29 第一次会话的交接文档（旧内容见 git 历史）。

## 0. 工作规则

- 成员：Wentao Yang（A，flow matching 线）、Kaiming Zhang（B，diffusion / 评估线）。
- 和用户用**中文**交流；slide、讲稿、report、图中文字一律**英文**；代码注释可以用中文。
- 只在 `C:\Users\Wentao\Desktop\cfm_project` 工作。实验用 `D:\Anaconda\envs\flow\python.exe`（torch CUDA，RTX 5070 Ti 12 GB）。系统 python 是 CPU 版，不要用它跑实验。
- Git：**只有用户要求时才 commit / push**，走“分支 → PR”流程；仓库 https://github.com/liliyawanttosleep/cfm-project （私有）。
- 不改 TDR 等系统设置。
- GPU 任务一律限速运行（`code/common/throttle.py`，默认 `--duty 0.5`，显卡超过 72°C 暂停、65°C 恢复，可用内存低于阈值时暂停；本机内存占用高，阈值用 1.5 GB）。同一时间只跑一个 GPU 任务。
- 报告不排练，不要再讨论报告时长。

## 1. 研究目标和问题

**课题**：Conditional Flow Matching（CFM）把生成变成回归，采样就是解 ODE，代价是网络调用次数（NFE）。**Final report 的主线：怎样让 CFM 用更少的 NFE 生成好样本，以及怎样可靠地检测生成质量。** Diffusion（DDPM 网络 + DDIM 采样）只作参照。

| # | 问题 |
|---|---|
| RQ1 | 质量检测：怎样可靠地判断一个 CFM 模型更好？FID 的误差有多大？FID 漏掉了什么（类别覆盖）？ |
| RQ2 | 推理端：不重新训练，只换求解器和时间网格，固定 NFE 下能提升多少？ |
| RQ3 | 训练端：OT batch 大小、训练时间分布、reflow 各能改善多少少步质量？训练代价多大？ |
| RQ4 | 诊断：路径直线度能否预测需要多少步？两端的改进能否叠加？ |

## 2. 已确定的结论和决定

**时间节点**：课堂报告 **11/10（周二）**，30 分钟，两人讲；final report **12/16 23:59**，英文，正文约 5 页。

**决定**
- 课程方法只要求**一种**（11/07 再次确认）：用聚类，即 k-means + χ²。
- 报告 PPT 已定稿，不再改结构。
- Final 实验收窄到 CFM 的效率和质量检测（C1–C7），不做新的 diffusion 实验。
- Report 用 NeurIPS 风格 LaTeX，结构为 Introduction / Related Work / Methods / Experiments / Conclusion / References，另加 Individual Contributions（课程硬性要求）。

**实验结论（均为我们自己跑出的结果）**
- 2D（种子 0）：OT-CFM 从 1 步起就接近最好；I-CFM 走 1 步会落在数据均值上；DDIM 在 1–2 步时失败。
- MNIST FID，Euler 步数 NFE = 1 / 2 / 5 / 10 / 20 / 50 / 100，每格是“种子 0 / 种子 1”：

  | 模型 | NFE 1 | 2 | 5 | 10 | 20 | 50 | 100 |
  |---|---|---|---|---|---|---|---|
  | OT-CFM | 223 / 207 | 76.0 / 83.2 | 26.7 / 36.9 | 19.0 / 25.1 | 17.0 / 20.1 | 15.6 / 18.2 | 13.9 / 16.8 |
  | I-CFM | 332 / 318 | 132 / 136 | 45.3 / 37.9 | 27.9 / 22.8 | 20.7 / 17.4 | 16.9 / 15.0 | 14.5 / 12.9 |
  | DDIM | 293 / 356 | 291 / 356 | 99.3 / 106 | 41.0 / 53.0 | 28.4 / 40.1 | 23.1 / 34.4 | 12.2 / 32.8 |

  - 种子 0 用 dopri5 时，OT-CFM 11.5、I-CFM 12.1（约 275 NFE）。
  - 真实 vs 真实的 FID 是 0.54。
- OT-CFM 只在 1–2 步时两个种子都更好：FID 低 33–42%。5 步时两个种子分别低 41% 和 3%；从 10 步起两个种子不一致。
- 在 784 维中，minibatch OT（batch 128）的配对代价只比随机配对低 4%，2D 中低 85–94%。可节省的上限是 2 tr Σ^½ = 386，总代价约 1,509。
- E3（k-means，k = 10，100 步，10,000 张）：
  - DDIM 种子 0：FID 最好（12.2），但类别最不均衡（TV 0.19；数字 1 占 18.7%，3 只占 4.5%）。
  - DDIM 种子 1：类别均衡（TV 0.06），FID 却是 32.8。
  - 结论：只看 FID、只看一个种子都会误判。
  - 真实测试集基线 TV 0.017。所有模型的 TV 是基线的 2.6–11 倍。换采样噪声，TV 最多变 0.012；效应量 w 在 100 步时为 0.11–0.46。
- DDIM 两个种子的训练损失几乎相同（0.0215 / 0.0213），FID 却差很多：训练损失不能用来比较模型质量。

## 3. 当前版本的关键内容

**方法**
- I-CFM：独立配对 x₀ ~ N(0, I)、x₁ ~ data；x_t = (1 − t)x₀ + t·x₁；回归目标是 x₁ − x₀；σ = 0。
- OT-CFM：在每个 batch 内用精确 OT 重新配对。
- DDPM：线性 β，T = 1000，ε 预测；采样用 DDIM（η = 0）。
- 记号：t = 0 是噪声，t = 1 是数据。

**MNIST 设定（三个模型完全相同）**
- 网络：torchcfm U-Net，32 通道，1.08M 参数。
- 训练：50 epoch（23,400 步），batch 128，Adam 默认学习率，无 EMA；每个模型约 33 分钟（不限速）。

**评估协议**
- FID：clean-fid，10,000 张，参考集是 MNIST 训练集，采样噪声 `torch.manual_seed(100)`。
- 类别覆盖：k-means 只在 30,000 张真实图上拟合，生成图分到最近的中心；报告 TV、χ²、w = √(χ²/n)。
- 数字标签：用 RBF-SVM（准确率 97%）给生成图打标签，只用来说明“哪个数字偏多或偏少”。
- 直线度：路径长度比（100 步 Euler 走过的路径长度 / 首尾直线距离）。

**代码**
- `code/e1_2d/`：2D 训练（`train_fm.py`、`train_ddpm.py`）和汇总。
- `code/e2_mnist/train_mnist.py`：`--method icfm|otcfm|ddpm --seed N`。
- `code/e4_cifar/eval_images.py`：MNIST / CIFAR 的 FID、一步行为、路径长度比。
- `code/e3_kmeans/cluster_coverage.py`：E3 的 k-means 实验。
- `code/e3_kmeans/fid_seed.py`：其他种子的 FID。
- `code/common/throttle.py`：限速。
- 结果：`results/e1/`、`results/e2/`、`results/e3/`、`results/e4/`。权重、日志、FID 统计量不进 git。

**报告材料**
- `slides/FM_vs_Diffusion_Talk.pptx`：37 页（正文 31 + B1–B6），带讲者备注，已用 PowerPoint 逐页检查。
- `slides/FM_vs_Diffusion_Talk.pdf`、讲稿 `slides/FM_vs_Diffusion_Speech_Script.docx`。
- slide 源文件：`slides/deck_v2/project/`。
- 生成方法：
  - PDF：`slides/build_pdf.py`；
  - PPT：`slides/build_pptx.py`，用 `D:\Anaconda\python.exe`（3.13）运行，因为 flow 环境缺 lxml；
  - 讲稿：`slides/speech/`；
  - 图：`slides/make_image_figs.py`。
- 在线 slide：https://claude.ai/artifact/7Lpjq76Dw2ywYrmNCLgUDR
- 在线讲稿：https://claude.ai/artifact/SsCxbkUcGE1DdD8HQvhD8r

## 4. 被否定的方案和原因

| 方案 | 原因 |
|---|---|
| TSTR（在生成图上训练、在真实图上测试，逻辑回归） | 预实验中所有模型的准确率都是 0.89–0.90，和真实数据（0.90）一样，区分不了模型 |
| real-vs-fake 分类器（随机森林） | 所有模型都被识别出 89–95%，主要靠背景像素不是纯黑，不反映生成质量 |
| PCA + Fréchet 距离 | 能用，但又是一个和 FID 类似的单一分数，看不出缺了哪类数字 |
| 四类课程方法都做 | 课程只要求一种 |
| 第一版实验计划 R1–R7（包括 DDPM 1000 步、DDIM 网格消融、Fashion-MNIST、补齐 2D 三个种子、GMM / KDE 基线） | 偏离主线；收窄为 CFM 的效率和质量检测（C1–C7） |
| EMA、cosine 调度、classifier-free guidance、MNIST-FID | 和少步效率无关，或需要额外模型 |
| Consistency model、MeanFlow | 训练目标不同，超出 CFM 框架和时间预算，只在 Related Work 中提及 |
| VAE / GAN 四模型对比、表格数据合成 | 选题阶段否定：范围太宽 / 网络部分太少 |

## 5. 下一步待办

**Git**
- PR #3（报告定稿、C1–C7 计划、项目全流程、report 格式、推导笔记 D1–D8）已于 11/08 合并进 main。
- 推导笔记 D9 和本文件的更新走新分支 `update/derivations-d9` 的 PR。
- `slides/FM_vs_Diffusion_Talk.pptx` 本地被 PowerPoint 重新保存过，没有提交；需要时由用户决定是否提交。

**报告前（11/08–11/10）**
- [ ] 用 PowerPoint 放映（F5），确认第 3、15 页的动图在动
- [ ] 回复在线讲稿上的评论 `1ef6e105-1ea5`
- [ ] 确认 slide 3 的措辞 “Only the data and the network change”
- [ ] Kaiming 过一遍自己负责的页和讲稿

**Final 实验（11/11–12/02，详见 `docs/FINAL_REPORT_PLAN.md`）**
- **先做（必做）**
  - C1：I-CFM、OT-CFM 训练第 3 个种子；测采样噪声和样本数带来的 FID 误差；TV 的 bootstrap 区间。
  - C2：Euler / Heun / RK4 × 均匀 / shift / 按曲率分配 / 校准网格，不需要训练。1D 核对显示按曲率分配的网格在目标平滑时略好、目标集中时更差，所以实际方法是校准网格（推导和核对见 `docs/notes/derivations.md` D5）。
- **再做**
  - C5 reflow（必做）。
  - C3 OT batch 128–4,096（应做）。
  - C4 训练时间分布：logit-normal 和按曲率加权（应做）。
  - C6 综合分析（必做）。
  - C7 CIFAR-10 验证（选做）。
- **需要先改的代码**
  - 给 `train_mnist.py` 加 `--ot-batch` 和 `--t-dist`；
  - 新建 `code/c2_solvers/sample_grid.py`；
  - `cluster_coverage.py` 改为额外保存每簇计数。

**Report（12/03–12/16，格式见 `report/README.md`）**
- [ ] 向老师确认附录是否允许、是否计入 5 页
- [ ] 12/07 初稿 → 12/09 交叉审阅 → 12/15 定稿 → 12/16 提交

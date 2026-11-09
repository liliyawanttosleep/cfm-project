# Flow Matching vs. Diffusion — Data Mining Final Project

GitHub：https://github.com/liliyawanttosleep/cfm-project （私有）· 协作流程见 [CONTRIBUTING.md](CONTRIBUTING.md)

**截止：2026-12-16 23:59（final report）** · 两人组 · 30 分钟 presentation

## 文档

> **👉 先读 [TEAM_PLAN.md](TEAM_PLAN.md)：总计划（分工、Pre 准备、实验步骤、预期结果、公式、report、工作规则）。** 下面的文档是补充细节。

| 文件 | 内容 |
|---|---|
| [TEAM_PLAN.md](TEAM_PLAN.md) | **总计划，以它为准** |
| [docs/FINAL_REPORT_PLAN.md](docs/FINAL_REPORT_PLAN.md) | **报告之后的实验计划 C1–C7**（final report 用，主线：CFM 的生成效率与质量检测）：设计流程、理想结果、要验证的问题 |
| [docs/notes/derivations.md](docs/notes/derivations.md) | Report 用的公式推导 D1–D9（英文），含数值核对 |
| [report/README.md](report/README.md) | **Final report 的格式、提纲、页数分配和提交检查清单** |
| [docs/SLIDE_PLAN.md](docs/SLIDE_PLAN.md) | 11/10 课堂报告的 slide 规划 |
| [slides/README.md](slides/README.md) | slide、PPT、PDF、讲稿的生成方法 |
| [code/README.md](code/README.md) | 代码结构与运行命令 |
| [docs/meetings.md](docs/meetings.md) | 周会记录 |
| [docs/00_选题讨论记录.md](docs/00_选题讨论记录.md) | 课程要求、为什么选这个题 |
| [docs/01_项目计划.md](docs/01_项目计划.md) | 研究问题、实验设计 E1–E4、时间线、report 结构 |
| [docs/02_任务清单_两人分工.md](docs/02_任务清单_两人分工.md) | **每周任务清单（成员 A / 成员 B）** |
| [docs/03_参考资料.md](docs/03_参考资料.md) | 论文（已下载到 `papers/`）、开源库、本地可用例子 |

## 目录

```
docs/        计划与记录          papers/   论文 PDF（编号与 03_参考资料 对应）
code/        我们自己的代码      results/  图、表、日志
report/      LaTeX               slides/   presentation
data/        数据集（MNIST）
conditional-flow-matching-main/   torchcfm 原仓库（只读参考）
```

## 环境

本机已有可用环境 `flow`（D:\Anaconda\envs\flow）：Python 3.12、torch 2.9.0+cu130，可用 GPU RTX 5070 Ti Laptop（12 GB），已装 torchcfm / POT / torchdyn。

```bash
conda activate flow
pip install clean-fid umap-learn scikit-learn matplotlib pandas
```

> 注意：系统默认的 `python`（3.13）装的是 **CPU 版 torch**，也没有 POT，不要用它跑实验。
> 另一位成员的电脑：先装 PyTorch 官网对应 CUDA 版本，然后在 `conditional-flow-matching-main/` 下执行 `pip install -e .`，再执行上面的 pip 命令。

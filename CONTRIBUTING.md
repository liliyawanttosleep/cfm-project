# 协作流程（GitHub）

## 第一次设置（每人一次）

```bash
git clone https://github.com/liliyawanttosleep/cfm-project.git cfm_project
cd cfm_project
git clone https://github.com/atong01/conditional-flow-matching conditional-flow-matching-main   # 上游库，不进我们的 git
bash scripts/download_papers.sh                                                             # 论文，不进 git
```
然后按 `TEAM_PLAN.md` §3.3 配置环境，并按 §3.4 跑冒烟测试。

## 日常流程：分支 → Pull Request → 合并

```bash
git switch main && git pull                     # 1. 先同步最新的 main
git switch -c a/e1-8gaussians                   # 2. 新建自己的分支：a/... 或 b/...
# ... 改代码、跑实验、改文档 ...
git add <改过的文件> && git commit -m "E1: 补跑 8gaussians"
git push -u origin a/e1-8gaussians              # 3. 推送分支
# 4. 在 GitHub 上开 Pull Request，请对方看一眼，然后 Merge
```

- **不要直接推送到 main**。小改动（错别字、勾选任务清单）可以例外。
- 一个分支只做一件事，做完就合并，不要拖很多天。
- 合并后删除分支；下次开始工作前先 `git switch main && git pull`。
- 冲突时：`git switch <你的分支> && git merge main`，解决冲突后再提交。

## 什么提交、什么不提交

| 提交 | 不提交（已在 `.gitignore` 中） |
|---|---|
| `code/`、`notebooks/`（清空输出）、`docs/`、`TEAM_PLAN.md` | `papers/`、`data/`、`*.pdf` |
| `results/<实验>/` 下的 `*.json`、`*.csv`、`*.png`、`README.md` | 模型权重 `*.pt`、`results/**/ckpt/`、`results/**/weights/` |
| `slides/`（源文件和图） | 上游库 `conditional-flow-matching-main/` |
| `report/`（LaTeX 源文件） | 生成的大批样本 `results/**/samples/` |

单个文件超过约 5 MB 时，先在群里商量再提交。

## 与 Claude 协作

两人的 Claude 都先读 `TEAM_PLAN.md`（总计划和第 10 节工作规则）。让 Claude 改完代码后，也按上面的分支流程提交并开 PR。

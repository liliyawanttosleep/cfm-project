# Slides

## 课堂报告（2026-11-10）：当前版本

| 内容 | 位置 |
|---|---|
| **全部幻灯片合成一个文件** | [`FM_vs_Diffusion_Talk.pdf`](FM_vs_Diffusion_Talk.pdf)，共 37 页：主体 31 页（Opening 到 Close，第 28 页是 E3）和备份 B1–B6 |
| 在线幻灯片（演示用，以它为准） | https://claude.ai/artifact/7Lpjq76Dw2ywYrmNCLgUDR |
| 逐页源文件和讲者备注 | `deck_v2/project/deck.json` + `deck_v2/project/slides/<id>.html` |
| **PowerPoint（演讲用，动图能播放，讲稿在备注里）** | [`FM_vs_Diffusion_Talk.pptx`](FM_vs_Diffusion_Talk.pptx)，由 `python slides/build_pptx.py` 生成（需要 `pip install python-pptx`；讲者备注来自 `speech/script.md`），改完 HTML 后重新运行 |
| Word 讲稿的源文件 | `speech/script.md`（`speech/build.py` + `speech/finish.ps1` 生成 Word 版） |
| 重新生成合并文件 | `python slides/build_pdf.py --deck deck_v2`；加 `--notes` 另外生成带讲者备注的 `build/FM_vs_Diffusion_Talk_notes.pdf` |

## 旧版 Pre（开题报告）

以下内容只针对旧版 `deck/`。

| 内容 | 位置 |
|---|---|
| **在线幻灯片（演示用，以它为准）** | https://claude.ai/artifact/GRZZAyn8ir3RXQE1Hc8P5e （可导出 PPTX/PDF） |
| 幻灯片源文件（版本记录） | `deck/project/deck.json` + `deck/project/slides/<id>.html`（每页一个文件） |
| 幻灯片用图 | `figs/`，用 `python slides/make_slide_figs.py slides/figs` 重新生成（需先跑完 E1 的 moons 实验） |
| **PDF（一个文件，审阅和发给别人用）** | `python slides/build_pdf.py` → `build/FM_vs_Diffusion_Pre.pdf`；加 `--notes` 另外生成每页后面附讲稿的 `build/FM_vs_Diffusion_Pre_notes.pdf` |

## 修改流程（看 PDF → 提意见 → 改源文件 → 重新生成）

1. 打开 `build/FM_vs_Diffusion_Pre.pdf`，按**页码**写意见，例如“第 7 页：公式太小”“第 19 页：右图放大”。
2. Claude 修改对应的源文件 `deck/project/slides/<id>.html`。页码与 id 的对应关系见 `deck/project/deck.json` 的 `order`：第 1 页 = cover，第 2 页 = motivation，以此类推。
3. 运行 `python slides/build_pdf.py`，重新生成 PDF 后再看。
4. 定稿后，把改过的页发布到在线幻灯片（见下文方式 2），并提交到 git。

PDF 用无头 Chrome/Edge 渲染，需要联网加载 Google Fonts。在线版和 PDF 用的是同一份源文件，排版基本一致。

## 怎么改幻灯片

**方式 1：直接在线改（推荐，适合改文字和排版）**
在线幻灯片的拥有者是 Wentao，队友需要先在页面的 Share 菜单中被授予**编辑权限**。之后打开链接就能直接编辑，改动实时保存。

**方式 2：让 Claude 改源文件（适合大段内容或批量修改）**
1. 修改 `deck/project/slides/<id>.html`。格式规则：每页只有一个 `<section>`，样式全部内联；讲稿写在最后的 `<aside>` 里。
2. 让 Claude 把改过的文件发布到上面的链接（Artifact publish，`url` 设为上面的链接，`root` 设为 `slides/deck`）。只有拥有编辑权限的人才能发布。
3. 提交到 git。

**同步规则**：在线版本是演示时的最终版本。每次在线改完一轮，请 Claude 用 Artifact `read` 把在线的 `project/` 文件拉回 `slides/deck/project/` 并提交，这样 git 里始终有一份最新快照，也能看到每次改了什么。

## 页面编号与讲者

页面顺序见 `deck/project/deck.json` 中的 `order`；每页讲稿的开头标有【A 讲】或【B 讲】以及时长。完整大纲见 `TEAM_PLAN.md` §4 P6。

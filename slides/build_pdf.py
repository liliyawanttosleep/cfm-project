"""把 slides/deck/project/ 下的逐页源文件合并成一个 HTML，再用无头 Chrome/Edge 导出成一个 PDF。

用法（在 cfm_project 根目录下）：
    python slides/build_pdf.py            # → slides/build/deck.html 和 slides/build/FM_vs_Diffusion_Pre.pdf
    python slides/build_pdf.py --notes    # 另外导出带讲稿的版本（每页下方附上 <aside> 中的讲稿）
    python slides/build_pdf.py --deck deck_v2 --notes    # 课堂报告 deck（slides/deck_v2/）：整合成一个文件 slides/FM_vs_Diffusion_Talk.pdf（进 git），带讲稿版在 build/

源文件是唯一需要修改的地方：slides/deck/project/slides/<id>.html（每页一个文件），页序在 deck.json 的 order 中。
在线 Slides 支持的几个自定义标签（x-icon、x-shape）在这里被替换成普通的 SVG，以便浏览器直接渲染。
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"

# 在线幻灯片中上传的图片（/_blob/<id>）对应的本地文件
BLOBS = {
    "a8e9b6c8b3d8aa10f9d1a061e2a10984": "figs/traj_compare.png",
    "a93672477e02b5236eeef6d8da86da4f": "figs/samples_grid.png",
    "76c7c974902916d90acd95af5ed91833": "figs/prob_paths.png",
    "2d822b3f4aa50320b1d583e727fb9b59": "figs/w2_vs_nfe.png",
    "cfc22190ee49a30793e8d3a7d5dc3af8": "figs/datasets_grid.png",
    "543670184f905869343c19211535706e": "figs/w2_all.png",
    # 动图在 PDF 里换成静态版本
    "9b1fbf15b2eef9b9d2d61fac2895a6dd": "figs/sampling_strip.png",   # sampling.gif
    "79ebb7a368ab48e00ce231ffb0c852fc": "figs/prob_paths.png",       # paths.gif
    "472b9aa80ca74f7148454c5dad4d91aa": "figs/sampling_strip.png",
    "be4c17ff6fc412b5fe26d77620e46972": "figs/one_pair.png",
    "0291b03702f2917e163ceaff62a61f2c": "figs/all_pairs.png",
    "bffa23c02ded1621214080cd7a18bed5": "figs/posterior_mean.png",
    "2ddc4d1679b83d6865551db26e7affc1": "figs/closed_form.png",
    "d5f5c8efe3a4fe583327e895c583896c": "figs/loss_parabola.png",
    "fb76f3c1c1dbd7b15ae41b3b397240bc": "figs/ode_euler.png",
    "907654b7514a9e66192368e471751fe1": "figs/continuity_1d.png",
    "3e8821a5d1a3f7577ec473cb06f60602": "figs/many_paths.png",
    "c19a0e36849f86e0425b237545d99450": "figs/opening_2d.png",       # opening_2d.gif
    "50da418141580e4b7e19e82cf42791ee": "figs/169_generated_samples_otcfm.png",  # 169_generated_samples_otcfm.gif
    "073bf32d4d70f5499dff57d452afca30": "figs/icfm_demo.png",        # icfm_demo.gif
    "97bd47f265527af9903a725ec1be896a": "figs/bend_1d.png",
    "f30123a64e3c54a5c7c38d8938da844a": "figs/pairs_swap.png",
    "708f4b6eef5ddc4b05aa46cfe2ecdb63": "figs/traj_ot_vs_i.png",
    "4391ed3690eb42eedb5b28afd743deaf": "figs/w2_four.png",
    "d7fac07126b7b98f1dd01c85dece77f9": "figs/moons_5steps.png",
    "626b4664653456a853619f9c7b6003f8": "figs/vp_paths.png",
    "49bf1e0c59ada058fc990d91de5e30ae": "figs/vp_overlay.png",
}

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser", "microsoft-edge",
]

# 在线 Slides 渲染器的默认样式（标题、段落、表格），这里尽量复现
BASE_CSS = """
@page { size: 1920px 1080px; margin: 0; }
* { margin: 0; box-sizing: border-box; }
html, body { background: #555; }
body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
section { position: relative; width: 1920px; height: 1080px; overflow: hidden;
          display: flex; flex-direction: column; page-break-after: always; break-after: page; }
section > aside { display: none; }
h1 { font-size: 96px; font-weight: 600; line-height: 1.1; }
h2 { font-size: 64px; font-weight: 600; line-height: 1.15; }
h3 { font-size: 44px; font-weight: 600; line-height: 1.2; }
p, li { line-height: 1.4; }
img { display: block; }
table { border-collapse: collapse; width: 100%; }
th, td { padding: 0.35em 0.6em; border-bottom: 1px solid #D9DDE3; vertical-align: top; }
th { font-weight: 600; color: #14213D; border-bottom: 2px solid #14213D; }
.notes { width: 1920px; min-height: 1080px; background: #FAF8F3; color: #14213D; padding: 96px 128px;
         font: 30px/1.6 'IBM Plex Sans', Arial, sans-serif; white-space: pre-wrap; page-break-after: always; break-after: page; }
.notes h4 { font-size: 36px; margin-bottom: 32px; color: #5B6B82; }
"""

ICONS = {
    "Check": '<path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>',
}


def style_of(attrs: str) -> str:
    m = re.search(r'style="([^"]*)"', attrs)
    return m.group(1) if m else ""


def replace_custom_tags(html: str) -> str:
    def icon(m):
        name, st = re.search(r'name="([^"]+)"', m.group(1)).group(1), style_of(m.group(1))
        return f'<svg viewBox="0 0 24 24" style="{st};flex:none">{ICONS.get(name, "")}</svg>'

    def shape(m):
        kind, st = re.search(r'kind="([^"]+)"', m.group(1)).group(1), style_of(m.group(1))
        fill = (re.search(r"background:\s*([^;]+)", st) or [None, "#888"])[1]
        size = re.sub(r"background:[^;]+;?", "", st)
        if kind == "arrow-right":  # 块状箭头：箭头头部占长度的后 40%
            return (f'<svg viewBox="0 0 100 50" preserveAspectRatio="none" style="{size};flex:none">'
                    f'<polygon points="0,15 60,15 60,0 100,25 60,50 60,35 0,35" fill="{fill}"/></svg>')
        return f'<div style="{st}"></div>'

    html = re.sub(r"<x-icon([^>]*)>\s*</x-icon>", icon, html)
    html = re.sub(r"<x-shape([^>]*)>\s*</x-shape>", shape, html)
    for blob, path in BLOBS.items():
        html = html.replace(f"/_blob/{blob}", f"../{path}")
    return html


def build_html(deck_dir: Path, with_notes: bool) -> Path:
    deck = json.loads((deck_dir / "deck.json").read_text(encoding="utf-8"))
    fonts = "\n".join(f'<link rel="stylesheet" href="{f["href"]}">' for f in deck["faces"].values() if "href" in f)
    pages = []
    for i, sid in enumerate(deck["order"], 1):
        src = (deck_dir / "slides" / f"{sid}.html").read_text(encoding="utf-8")
        pages.append(replace_custom_tags(src))
        if with_notes:
            note = re.search(r"<aside>(.*?)</aside>", src, re.S)
            pages.append(f'<div class="notes"><h4>Slide {i} · {sid} · speaker notes</h4>{note.group(1).strip() if note else "(no notes)"}</div>')
    out = BUILD / ("deck_notes.html" if with_notes else "deck.html")
    out.write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><title>{deck["title"]}</title>{fonts}'
        f"<style>{BASE_CSS}</style></head><body>{''.join(pages)}</body></html>",
        encoding="utf-8",
    )
    return out


def find_browser() -> str:
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    sys.exit("找不到 Chrome 或 Edge，请安装其中一个")


def to_pdf(browser: str, html: Path, pdf: Path):
    subprocess.run(
        [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=15000",
         f"--print-to-pdf={pdf}", html.as_uri()],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180,
    )
    print(f"saved {pdf}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--notes", action="store_true", help="另外导出带讲稿的版本")
    p.add_argument("--deck", default="deck", help="slides/ 下的 deck 目录：deck（旧版）或 deck_v2（重做版）")
    args = p.parse_args()
    BUILD.mkdir(exist_ok=True)
    browser = find_browser()
    deck_dir = HERE / args.deck / "project"
    name = "FM_vs_Diffusion_Pre" if args.deck == "deck" else "FM_vs_Diffusion_Talk"
    out_dir = BUILD if args.deck == "deck" else HERE  # 报告 deck 的合并 PDF 直接放在 slides/ 下
    to_pdf(browser, build_html(deck_dir, False), out_dir / f"{name}.pdf")
    if args.notes:
        to_pdf(browser, build_html(deck_dir, True), BUILD / f"{name}_notes.pdf")


if __name__ == "__main__":
    main()

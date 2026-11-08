"""把课堂报告 deck（slides/deck_v2/）转成一个可编辑的 PowerPoint，GIF 保留动画，讲稿写入备注。

    pip install python-pptx
    python slides/build_pptx.py          # → slides/FM_vs_Diffusion_Talk.pptx

做法
    1. 用无头 Chrome 打开逐页的 HTML，量出每个文字块、色块和图片的位置。量之前把字体换成 Office 自带字体
       （IBM Plex Sans → Calibri，Source Serif 4、STIX Two Text → Cambria），换行才会和 PowerPoint 接近。
    2. 用 python-pptx 按这些位置画成原生文本框、形状和图片，所以每页都能直接编辑。
    3. 第 1–31 页的备注来自 slides/speech/script.md 的 Script，备份页的备注来自 HTML 的 <aside>。

逐页源文件仍是 deck_v2/project/slides/<id>.html：改完内容后重新运行本脚本。
"""
import html as htmllib
import json
import re
import subprocess
import uuid
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Emu, Pt

import build_pdf as bp

HERE = Path(__file__).resolve().parent
DECK = HERE / "deck_v2" / "project"
OUT = HERE / "FM_vs_Diffusion_Talk.pptx"
SCRIPT = HERE / "speech" / "script.md"
PX = 12700 / 2  # 1920 px 的画布对应 13.333 英寸宽，1 px = 0.5 pt
BODY_SCALE = 1.06  # Calibri 比 IBM Plex Sans 窄且显小，正文放大一点，保持相近的观感

# 在线版的动图：PDF 用静态图，PowerPoint 用原始 GIF
GIFS = {
    "c19a0e36849f86e0425b237545d99450": "figs/opening_2d.gif",
    "1b8a087d9b7b868374d763e90dcb47bf": "figs/mnist_flow.gif",
    "073bf32d4d70f5499dff57d452afca30": "figs/icfm_demo.gif",
}
FONTS = [
    (r"'IBM Plex Sans', Arial, sans-serif", "Calibri, sans-serif"),
    (r"'Source Serif 4', Georgia, serif", "Cambria, serif"),
    (r"'STIX Two Text', 'Times New Roman', serif", "Cambria, serif"),
]

MEASURE_JS = r"""
window.addEventListener('load', function () {
  const BLOCK = new Set(['P', 'DIV', 'H1', 'H2', 'H3', 'UL', 'OL', 'TABLE', 'TR', 'LI', 'TD', 'TH']);
  const TEXT = new Set(['P', 'H1', 'H2', 'H3', 'LI', 'TD', 'TH']);
  const px = s => parseFloat(s) || 0;
  function hex(c) {
    const m = c.match(/rgba?\((\d+), (\d+), (\d+)(?:, ([\d.]+))?\)/);
    if (!m || m[4] === '0') return null;
    return [1, 2, 3].map(i => (+m[i]).toString(16).padStart(2, '0')).join('').toUpperCase();
  }
  document.querySelectorAll('section *').forEach(el => {
    const cs = getComputedStyle(el);
    if (el.style.fontSize && cs.fontFamily.startsWith('Calibri')) el.style.fontSize = (px(cs.fontSize) * SCALE) + 'px';
  });
  function runs(node, out) {
    node.childNodes.forEach(n => {
      if (n.nodeType === 3) {
        let t = n.textContent.replace(/\s+/g, ' ');
        const cs = getComputedStyle(n.parentElement);
        if (cs.textTransform === 'uppercase') t = t.toUpperCase();
        if (t) out.push({text: t, font: cs.fontFamily.split(',')[0].replace(/['"]/g, ''), size: px(cs.fontSize),
                         bold: parseInt(cs.fontWeight) >= 600, italic: cs.fontStyle === 'italic', color: hex(cs.color),
                         spacing: px(cs.letterSpacing)});
      } else if (n.nodeName === 'BR') out.push({br: true});
      else if (n.nodeType === 1 && n.nodeName !== 'ASIDE') runs(n, out);
    });
    return out;
  }
  const slides = [];
  document.querySelectorAll('section').forEach(sec => {
    const o = sec.getBoundingClientRect();
    const items = [];
    function box(r) { return {x: r.left - o.left, y: r.top - o.top, w: r.width, h: r.height}; }
    function walk(el) {
      if (el.nodeName === 'ASIDE') return;
      const cs = getComputedStyle(el), r = el.getBoundingClientRect();
      if (el !== sec) {
        const bg = hex(cs.backgroundColor);
        if (el.nodeName === 'svg') {
          const poly = el.querySelector('polygon');
          items.push({type: 'arrow', ...box(r), color: poly ? hex(getComputedStyle(poly).fill) : '8A94A3'});
          return;
        }
        if (bg && r.width && r.height) items.push({type: 'rect', ...box(r), color: bg, radius: px(cs.borderTopLeftRadius)});
        for (const side of ['Top', 'Left', 'Bottom', 'Right']) {
          const w = px(cs['border' + side + 'Width']), c = hex(cs['border' + side + 'Color']);
          if (w && c && cs['border' + side + 'Style'] !== 'none') {
            const b = box(r);
            if (side === 'Bottom') b.y += b.h - w;
            if (side === 'Right') b.x += b.w - w;
            if (side === 'Top' || side === 'Bottom') b.h = w; else b.w = w;
            items.push({type: 'rect', ...b, color: c, radius: 0});
          }
        }
      }
      if (el.nodeName === 'IMG') {
        items.push({type: 'img', ...box(r), src: el.getAttribute('src'), alt: el.getAttribute('alt') || '',
                    nw: el.naturalWidth, nh: el.naturalHeight});
        return;
      }
      const blockKid = [...el.children].some(c => BLOCK.has(c.nodeName));
      if (TEXT.has(el.nodeName) && !blockKid) {
        const rs = runs(el, []);
        if (rs.some(x => x.text && x.text.trim())) {
          const rng = document.createRange(); rng.selectNodeContents(el);
          const tops = [...rng.getClientRects()].map(q => q.top).sort((a, b) => a - b);
          const lh = px(cs.lineHeight) || (rs[0].size * 1.22);
          let lines = 1; for (let i = 1; i < tops.length; i++) if (tops[i] - tops[i - 1] > lh * 0.6) lines++;
          const pl = px(cs.paddingLeft), pr = px(cs.paddingRight), pt = px(cs.paddingTop), pb = px(cs.paddingBottom);
          items.push({type: 'text', tag: el.nodeName, x: r.left - o.left + pl, y: r.top - o.top + pt, w: r.width - pl - pr, h: r.height - pt - pb,
                      align: cs.textAlign, lh: px(cs.lineHeight), lines, runs: rs});
        }
        return;
      }
      [...el.children].forEach(walk);
    }
    walk(sec);
    const foot = [...sec.children].find(c => c.nodeName === 'P' && getComputedStyle(c).position === 'absolute' && c.style.bottom);
    const flowEnd = Math.max(...[...sec.children].filter(c => getComputedStyle(c).position !== 'absolute' && c.nodeName !== 'ASIDE').map(c => c.getBoundingClientRect().bottom)) - o.top;
    const aside = sec.querySelector('aside');
    slides.push({id: sec.id, bg: hex(getComputedStyle(sec).backgroundColor), items, flowEnd, footTop: foot ? foot.getBoundingClientRect().top - o.top : null,
                 notes: aside ? aside.textContent.trim() : ''});
  });
  const pre = document.createElement('pre'); pre.id = 'out'; pre.textContent = JSON.stringify(slides);
  document.body.appendChild(pre);
});
"""


def measure(order):
    pages = []
    for sid in order:
        src = (DECK / "slides" / f"{sid}.html").read_text(encoding="utf-8")
        for a, b in FONTS:
            src = src.replace(a, b)
        for blob, path in GIFS.items():
            src = src.replace(f"/_blob/{blob}", f"../{path}")
        pages.append(bp.replace_custom_tags(src))
    page = bp.BUILD / "deck_pptx.html"
    bp.BUILD.mkdir(exist_ok=True)
    js = MEASURE_JS.replace("SCALE", str(BODY_SCALE))
    page.write_text(f'<!doctype html><html><head><meta charset="utf-8"><style>{bp.BASE_CSS}</style></head><body>{"".join(pages)}<script>{js}</script></body></html>', encoding="utf-8")
    res = subprocess.run([bp.find_browser(), "--headless=new", "--disable-gpu", "--allow-file-access-from-files", "--virtual-time-budget=20000", "--window-size=1920,1080",
                          "--dump-dom", page.as_uri()], capture_output=True, timeout=240)
    m = re.search(r'<pre id="out">(.*?)</pre>', res.stdout.decode("utf-8", "replace"), re.S)
    if not m:
        raise SystemExit("measuring failed: " + res.stderr.decode("utf-8", "replace")[-500:])
    return json.loads(htmllib.unescape(m.group(1)))


def script_notes():
    """{slide number: speaker notes} from the Script parts of the speech source."""
    src = SCRIPT.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"^## Slide (\d+) · (\w+) · ([\d.]+) min · .*?\n\n### Script\n\n(.*?)\n\n### Formulas", src, re.S | re.M):
        body = re.sub(r"\*\[([^\]]*)\]\*", r"[\1]", m[4])
        out[int(m[1])] = f"[{m[2]} · {m[3]} min]\n" + re.sub(r"\n\s*\n", "\n", body).strip()
    return out


def emu(v):
    return Emu(int(round(v * PX)))


def rgb(h):
    return RGBColor.from_string(h)


def fill_text(tf, it):
    tf.word_wrap = it["lines"] > 1 or it["align"] in ("center", "right")
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    paras, cur = [], []
    for r in it["runs"]:
        if r.get("br"):
            paras.append(cur)
            cur = []
        else:
            cur.append(r)
    paras.append(cur)
    for i, runs in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}.get(it["align"], PP_ALIGN.LEFT)
        if it["lh"]:
            p.line_spacing = Pt(round(it["lh"] / 2, 1))
        if runs:
            runs[0]["text"] = runs[0]["text"].lstrip()
            runs[-1]["text"] = runs[-1]["text"].rstrip()
        for r in runs:
            if not r["text"]:
                continue
            run = p.add_run()
            run.text = r["text"]
            f = run.font
            f.name, f.size, f.bold, f.italic = r["font"], Pt(round(r["size"] * 2) / 4), r["bold"], r["italic"]
            if r["color"]:
                f.color.rgb = rgb(r["color"])
            if r["spacing"]:
                run._r.get_or_add_rPr().set("spc", str(int(r["spacing"] / 2 * 100)))


def add_sections(prs, deck, order):
    """PowerPoint sections, one per part of the talk."""
    names = {"s1": "Opening", "s2": "Part I", "s3": "Part II", "s4": "Part III", "s5": "Part IV", "s6": "Part V", "s7": "Close", "s8": "Backup"}
    starts = {v["start"]: names.get(k, k) for k, v in deck["sections"].items()}
    ids = [s.get("id") for s in prs.part._element.sldIdLst]
    groups = []
    for sid, num in zip(order, ids):
        if sid in starts:
            groups.append((starts[sid], []))
        groups[-1][1].append(num)
    P14, P = "http://schemas.microsoft.com/office/powerpoint/2010/main", "http://schemas.openxmlformats.org/presentationml/2006/main"
    root = prs.part._element
    ext_lst = root.find(f"{{{P}}}extLst")
    if ext_lst is None:
        ext_lst = etree.SubElement(root, f"{{{P}}}extLst")
    ext = etree.SubElement(ext_lst, f"{{{P}}}ext", uri="{521415D9-36F7-43E2-AB2F-B90AF26B5E84}")
    lst = etree.SubElement(ext, f"{{{P14}}}sectionLst", nsmap={"p14": P14})
    for name, nums in groups:
        sec = etree.SubElement(lst, f"{{{P14}}}section", name=name, id="{" + str(uuid.uuid4()).upper() + "}")
        sl = etree.SubElement(sec, f"{{{P14}}}sldIdLst")
        for n in nums:
            etree.SubElement(sl, f"{{{P14}}}sldId", id=n)


def main():
    deck = json.loads((DECK / "deck.json").read_text(encoding="utf-8"))
    order = deck["order"]
    data = measure(order)
    notes = script_notes()

    prs = Presentation()
    ratio = 1920 * PX / prs.slide_width
    prs.slide_width, prs.slide_height = Emu(int(1920 * PX)), Emu(int(1080 * PX))
    for layout in [prs.slide_master, *prs.slide_layouts]:  # the default template is 4:3
        for sh in layout.placeholders:
            sh.left, sh.width = int(sh.left * ratio), int(sh.width * ratio)
    prs.core_properties.title = deck["title"]
    prs.core_properties.author = "Wentao Yang, Kaiming Zhang"
    layout = prs.slide_layouts[5]  # Title Only

    warnings = []
    for num, s in enumerate(data, 1):
        slide = prs.slides.add_slide(layout)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = rgb(s["bg"])
        title = slide.shapes.title
        used = False
        for it in s["items"]:
            if it["type"] == "rect":
                radius = it["radius"]
                sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, emu(it["x"]), emu(it["y"]), emu(it["w"]), emu(it["h"]))
                if radius:
                    sh.adjustments[0] = min(0.5, radius / min(it["w"], it["h"]))
                sh.fill.solid()
                sh.fill.fore_color.rgb = rgb(it["color"])
                sh.line.fill.background()
                sh.shadow.inherit = False
            elif it["type"] == "arrow":
                sh = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, emu(it["x"]), emu(it["y"]), emu(it["w"]), emu(it["h"]))
                sh.fill.solid()
                sh.fill.fore_color.rgb = rgb(it["color"])
                sh.line.fill.background()
                sh.shadow.inherit = False
            elif it["type"] == "img":
                src = it["src"]
                if not src.startswith("../"):
                    raise SystemExit(f"slide {num}: image {src} has no local file; add it to BLOBS in build_pdf.py")
                scale = min(it["w"] / it["nw"], it["h"] / it["nh"])  # object-fit: contain
                w, h = it["nw"] * scale, it["nh"] * scale
                pic = slide.shapes.add_picture(str(HERE / src[3:]), emu(it["x"] + (it["w"] - w) / 2), emu(it["y"] + (it["h"] - h) / 2), emu(w), emu(h))
                pic._element.nvPicPr.cNvPr.set("descr", it["alt"])
            else:
                slack = 4 if it["lines"] > 1 else 0
                if it["tag"] in ("H1", "H2") and not used:
                    used = True
                    title.left, title.top, title.width, title.height = emu(it["x"]), emu(it["y"]), emu(it["w"] + slack), emu(it["h"])
                    fill_text(title.text_frame, it)
                else:
                    tb = slide.shapes.add_textbox(emu(it["x"]), emu(it["y"]), emu(it["w"] + slack), emu(it["h"]))
                    fill_text(tb.text_frame, it)
        if not used:
            title._element.getparent().remove(title._element)
        slide.notes_slide.notes_text_frame.text = notes.get(num, s["notes"])
        if s["footTop"] is not None and s["flowEnd"] > s["footTop"] - 4:
            warnings.append(f"slide {num} {s['id']}: content ends at {s['flowEnd']:.0f}, footer starts at {s['footTop']:.0f}")
    add_sections(prs, deck, order)
    prs.save(OUT)
    print("saved", OUT, f"({OUT.stat().st_size / 1e6:.1f} MB,", len(data), "slides)")
    print("\n".join(warnings) or "no content runs into a footer")


if __name__ == "__main__":
    main()

"""Count spoken words per slide, fill the part times, build the .docx with pandoc (native Word equations)."""
import re
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
OUT = Path(sys.argv[1])
src = (HERE / "script.md").read_text(encoding="utf-8")

totals, part = {}, None
for m in re.finditer(r"^## Slide (\d+) · (\w+) · ([\d.]+) min.*?\n### Script\n(.*?)\n### Formulas", src, re.S | re.M):
    n, who, tag, body = int(m[1]), m[2], float(m[3]), m[4]
    words = len(re.sub(r"\*\[[^\]]*\]\*", "", body).split())
    totals[n] = words
    print(f"{n:>2} {who:8} tag {tag:>3} words {words:>4} = {words / 140:.2f} min")
p3 = sum(totals[i] for i in range(15, 20)) / 140
p4 = sum(totals[i] for i in range(20, 23)) / 140
print("I..IV words", sum(totals.values()), "minutes", round(sum(totals.values()) / 140, 2), "p3", round(p3, 2), "p4", round(p4, 2))
p5 = sum(totals[i] for i in range(23, 29)) / 140
p6 = sum(totals[i] for i in range(29, 32)) / 140
print("p5", round(p5, 2), "p6", round(p6, 2))
src = src.replace("@P3@", f"{p3:.1f}").replace("@P4@", f"{p4:.1f}").replace("@P5@", f"{p5:.1f}").replace("@P6@", f"{p6:.1f}").replace("@TOTAL@", f"{sum(totals.values()) / 140:.1f}").replace("@OPEN@", f"{sum(totals[i] for i in range(1, 5)) / 140:.1f}")
md = HERE / "script_filled.md"
md.write_text(src, encoding="utf-8")

# reference document: fonts, sizes and colors for the styles pandoc uses
ref = HERE / "reference.docx"
ref.write_bytes(subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"], capture_output=True, check=True).stdout)
tmp = HERE / "reference_styled.docx"
with zipfile.ZipFile(ref) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == "word/styles.xml":
            x = data.decode("utf-8")
            x = re.sub(r'<w:rFonts [^>]*/>', '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri" w:eastAsia="Microsoft YaHei"/>', x)
            x = x.replace('w:color w:val="4F81BD"', 'w:color w:val="14213D"').replace('w:color w:val="345A8A"', 'w:color w:val="14213D"')
            zout.writestr(item, x.encode("utf-8"))
        else:
            zout.writestr(item, data)
subprocess.run(["pandoc", str(md), "-f", "markdown+pipe_tables+tex_math_dollars", "-t", "docx", "--reference-doc", str(tmp), "-o", str(OUT)], check=True)
print("saved", OUT)

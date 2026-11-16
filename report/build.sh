#!/bin/bash
# Rebuild the final report: figures and tables from results/, then LaTeX (pdflatex + bibtex + pdflatex x2).
#   bash report/build.sh
set -e
cd "$(dirname "$0")"
PY=${PY:-D:/Anaconda/envs/flow/python.exe}
PYTHONIOENCODING=utf-8 "$PY" make_figs.py
PYTHONIOENCODING=utf-8 "$PY" make_tables.py
pdflatex -interaction=nonstopmode main.tex > build.log
bibtex main >> build.log
pdflatex -interaction=nonstopmode main.tex >> build.log
pdflatex -interaction=nonstopmode main.tex >> build.log
cp main.pdf FM_vs_Diffusion_Final_Report.pdf
grep -E "^!|undefined" main.log || true
echo "pending markers left: $(grep -o '\pending{' sections/*.tex | wc -l)"
pdfinfo main.pdf | grep Pages

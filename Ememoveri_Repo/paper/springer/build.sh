#!/usr/bin/env bash
# Rebuild the Springer LNCS paper from the result files.   Usage: bash paper/springer/build.sh   (needs pdflatex, bibtex, latexmk)
set -euo pipefail
cd "$(dirname "$0")/../.."
python paper/make_tables.py                                                   # tables + numbers.tex for the main benchmark (-> paper/)
python paper/make_ext.py results/real_ext/summary_ext.json paper/springer       # ladder, baselines, backends, LORO, lifecycle, figures
python paper/make_heldout.py results/real_ext/summary_heldout.json paper/springer
cp paper/numbers.tex paper/springer/numbers.tex
for t in data main ablation_short; do [ -f paper/tables/$t.tex ] && cp paper/tables/$t.tex paper/springer/tables/$t.tex; done
cp paper/figures/*.pdf paper/springer/figures/ 2>/dev/null || true
cd paper/springer && latexmk -pdf -interaction=nonstopmode ememoveri_springer.tex && pdfinfo ememoveri_springer.pdf | grep Pages

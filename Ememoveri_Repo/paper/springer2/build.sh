#!/usr/bin/env bash
# Springer LNCS version (<= 15 pages incl. references), converted from the IEEE source.   Usage: bash paper/springer2/build.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
bash paper/ieee/build.sh                                   # regenerates tables/figures and the IEEE source from the result files
cp paper/ieee/tables/*.tex paper/springer2/tables/ && cp paper/ieee/figures/*.pdf paper/springer2/figures/ && cp paper/ieee/numbers.tex paper/springer2/numbers.tex
python paper/springer2/to_lncs.py
cd paper/springer2 && latexmk -pdf -interaction=nonstopmode ememoveri_lncs.tex && pdfinfo ememoveri_lncs.pdf | grep Pages

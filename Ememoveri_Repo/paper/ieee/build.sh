#!/usr/bin/env bash
# Rebuild the IEEE-format paper (IEEEtran, conference mode) from the result files.   Usage: bash paper/ieee/build.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
python paper/make_tables.py                                                      # data.tex (development repositories) -> paper/tables
cp paper/tables/data.tex paper/ieee/tables/data.tex
python paper/make_ext.py results/real_ext/summary_ext.json paper/ieee            # ladder, strong baselines, backends, LORO, lifecycle, sweeps
python paper/make_heldout.py results/real_ext/summary_heldout.json paper/ieee
python paper/make_v06.py results/real_ext paper/ieee                             # EmemoVeri-2 tables and figures (current paper)
python paper/ieee/gen/fill_paper.py                                                  # fills the text template from the result files
cd paper/ieee && latexmk -pdf -interaction=nonstopmode ememoveri_ieee.tex && pdfinfo ememoveri_ieee.pdf | grep Pages

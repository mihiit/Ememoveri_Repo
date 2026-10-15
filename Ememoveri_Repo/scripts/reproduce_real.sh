#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-repos}; mkdir -p "$ROOT"
for r in fastapi/fastapi yt-dlp/yt-dlp browser-use/browser-use; do
  n=$(basename "$r")
  [ -d "$ROOT/$n.git" ] || git clone --bare --filter=blob:none -q "https://github.com/$r.git" "$ROOT/$n.git"
done
# NOTE: histories move on; results depend on the clone date. Pin with:  git --git-dir=$ROOT/<n>.git rev-parse HEAD
for n in fastapi yt-dlp browser-use; do echo "$n HEAD $(git --git-dir=$ROOT/$n.git rev-parse HEAD)"; done | tee results_heads.txt
python experiments/real_bench.py --git-dir-root "$ROOT" --out results/real
python experiments/analyze_real.py results/real
python experiments/figures_real.py results/real
# --- v0.3 extended analyses (ladder, history-prior baselines, backends, LORO, degradation, staleness, lifecycle) ---
python experiments/collect_ext.py --git-dir-root "$ROOT" --set dev --out collected.pkl
python experiments/analyze_ext.py collected.pkl results/real_ext/summary_ext.json
python paper/make_ext.py results/real_ext/summary_ext.json paper
# --- v0.4 held-out confirmation (protocol: docs/PROTOCOL_V04.md); ~1.8 GB RAM, ~10 min ---
for r in pallets/flask psf/requests encode/httpx Textualize/rich psf/black python-poetry/poetry pydantic/pydantic scrapy/scrapy pytest-dev/pytest aio-libs/aiohttp celery/celery sphinx-doc/sphinx; do
  n=$(basename $r); [ -d "$ROOT/$n.git" ] || git clone --bare --filter=blob:none -q "https://github.com/$r.git" "$ROOT/$n.git"
done
# results/real_ext/repo_heads_new.txt lists the pinned heads used in the paper (check out with: git --git-dir=$ROOT/<n>.git update-ref refs/heads/pinned <sha>; git --git-dir=$ROOT/<n>.git symbolic-ref HEAD refs/heads/pinned)
python experiments/collect_ext.py --git-dir-root "$ROOT" --set heldout --slim --out collected_heldout.pkl
python experiments/analyze_heldout.py collected_heldout.pkl results/real_ext/summary_ext.json results/real_ext/summary_heldout.json
python paper/make_heldout.py results/real_ext/summary_heldout.json paper

# --- v0.5: multi-language set, established baselines, learned rankers, repository-level statistics (docs/PROTOCOL_V05.md) ---
# Pin the multi-language mirrors first (heads in results/real_ext/repo_heads_multilang.txt; see docs/REPRODUCIBILITY.md).
python experiments/collect_ext.py --git-dir-root "$ROOT" --set multilang --slim --out collected_multilang.pkl
python experiments/collect_ext.py --git-dir-root "$ROOT" --set dev --out collected.pkl            # re-run: now also stores similar-history / co-change channels
python experiments/collect_ext.py --git-dir-root "$ROOT" --set heldout --slim --out collected_heldout.pkl
python experiments/task_flags.py --git-dir-root "$ROOT" --collected collected.pkl collected_heldout.pkl collected_multilang.pkl --out results/real_ext/task_flags.json
python experiments/analyze_v05.py collected.pkl collected_heldout.pkl collected_multilang.pkl results/real_ext/summary_ext.json \
       results/real_ext/summary_v05.json results/real_ext/ltr_models.pkl results/real_ext/task_flags.json
python paper/make_v05.py results/real_ext/summary_v05.json paper/ieee

# --- v0.6: EmemoVeri-2 (evidence-level retrieval), tuned on the original dev split, confirmed on 16 fresh repositories (docs/PROTOCOL_V06.md) ---
# Pin the fresh mirrors first (heads in results/real_ext/repo_heads_fresh.txt).  The collections below store the evidence ledger; --slim keeps memory low.
python experiments/collect_ext.py --git-dir-root "$ROOT" --set dev --out collected.pkl
python experiments/collect_ext.py --git-dir-root "$ROOT" --set fresh --slim --no-events --out collected_fresh.pkl
python experiments/collect_ext.py --git-dir-root "$ROOT" --set multilang --slim --out collected_multilang.pkl
python experiments/collect_ext.py --git-dir-root "$ROOT" --set heldout --slim --out collected_heldout.pkl
python experiments/task_flags.py --git-dir-root "$ROOT" --collected collected.pkl collected_heldout.pkl collected_multilang.pkl collected_fresh.pkl --out results/real_ext/task_flags.json
python experiments/tune_ev2.py collected.pkl results/real_ext/summary_v05.json results/real_ext/ev2_config.json results/real_ext/ev2_dev.json   # dev split only
python experiments/analyze_v06.py collected.pkl collected_heldout.pkl collected_multilang.pkl collected_fresh.pkl results/real_ext/summary_ext.json \
       results/real_ext/summary_v05.json results/real_ext/ltr_models.pkl results/real_ext/ev2_config.json results/real_ext/ev2_dev.json \
       results/real_ext/task_flags.json results/real_ext/summary_v06.json
python experiments/analyze_v06b.py collected.pkl collected_heldout.pkl collected_multilang.pkl collected_fresh.pkl results/real_ext/summary_ext.json \
       results/real_ext/summary_v05.json results/real_ext/ev2_config.json results/real_ext/ev2_dev.json results/real_ext/summary_v06b.json
# Out-of-sample component analysis and blend sensitivity of the frozen EmemoVeri-2 (descriptive; needs a no-carry replay for the carry-over row)
python experiments/collect_ext.py --git-dir-root "$ROOT" --set fresh --slim --no-events --no-carry --out collected_fresh_nocarry.pkl
python experiments/ablate_ev2.py ablate results/real_ext/ev2_config.json results/real_ext/ablate_ev2.json fresh16=collected_fresh.pkl \
       heldout12=collected_heldout.pkl multilang8=collected_multilang.pkl --nocarry collected_fresh_nocarry.pkl
python experiments/ablate_ev2.py sens results/real_ext/ev2_config.json results/real_ext/sens_ev2.json collected_fresh.pkl results/real_ext/summary_v06.json
bash paper/ieee/build.sh
bash paper/springer2/build.sh

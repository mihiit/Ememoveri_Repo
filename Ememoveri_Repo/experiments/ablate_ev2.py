"""Out-of-sample component analysis and sensitivity of the FROZEN EmemoVeri-2 (descriptive; nothing is re-tuned).

  python experiments/ablate_ev2.py ablate  <ev2_config.json> <out.json> <collected.pkl[:tag]> ... [--nocarry collected_nocarry.pkl]
  python experiments/ablate_ev2.py sens    <ev2_config.json> <out.json> <collected_fresh.pkl> <summary_v06.json>

`ablate` switches one ingredient off at a time (state filter, rename carry-over, frequency prior, recency prior, claim similarity,
evidence-item term, strength weighting) on each repository set and reports repository-level effects.
`sens` evaluates every point of the dev-tuning blend grid on the fresh set to show how fragile the frozen choice is."""
import json, pickle, sys, time
from dataclasses import replace
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, ".")
from ememoveri.evidence_retrieval import EvidenceConfig, claim_scores
from ememoveri.realbench_ext import METRICS, metrics_vec
from ememoveri.stats import holm, signflip_p, cohens_dz

mode, cfg_p, out_p = sys.argv[1:4]
raw = json.load(open(cfg_p))
EV2 = EvidenceConfig(**{k: v for k, v in raw.items() if k in EvidenceConfig.__dataclass_fields__})
IDX = {m: k for k, m in enumerate(METRICS)}
rng = np.random.default_rng(2)


def ranked(L, t, cfg, state=True):
    s = claim_scores(L, t, cfg)
    idx = np.flatnonzero(t.local & ~t.F.sup) if state else np.flatnonzero(t.local)
    return idx[np.lexsort((-t.F.updated[idx], -s[idx]))[:20]]


def run(D, cfg, state=True, tasks=None):
    shim = SimpleNamespace(path_of=D["path_of"], repo_of=D["repo_of"]); L = D["ledger"]
    ts = D["tasks"] if tasks is None else tasks
    return np.array([metrics_vec(shim, t.F, ranked(L, t, cfg, state), t.gold, t.q_repo) for t in ts])


def macro(A, rep, repos, met):
    return np.array([A[rep == r, IDX[met]].mean() for r in repos])


def rboot(d, B=10000):
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


if mode == "ablate":
    nocarry = None
    args = sys.argv[4:]
    if "--nocarry" in args:
        i = args.index("--nocarry"); nocarry = args[i + 1]; args = args[:i] + args[i + 2:]
    R = {}
    for spec in args:
        tag, path = spec.split("=")
        D = pickle.load(open(path, "rb"))
        rep = np.array([t.repo for t in D["tasks"]]); repos = sorted(set(rep))
        V = {"EmemoVeri-2 (full)": run(D, EV2), "- state filter": run(D, EV2, state=False),
             "- frequency prior": run(D, replace(EV2, l_pop=0.0)), "- recency prior": run(D, replace(EV2, l_rec=0.0)),
             "- both priors": run(D, replace(EV2, l_pop=0.0, l_rec=0.0)), "- claim similarity": run(D, replace(EV2, w_sim=0.0)),
             "- evidence-item term": run(D, replace(EV2, w_ev=0.0)), "- strength weighting": run(D, replace(EV2, strength_exp=0.0))}
        if nocarry and tag == "fresh16":
            Dn = pickle.load(open(nocarry, "rb")); assert [t.sha for t in Dn["tasks"]] == [t.sha for t in D["tasks"]]
            V["- rename carry-over"] = run(Dn, EV2); del Dn
        block = {"repos": repos, "n_total": len(rep), "variants": {}}
        pv = {}
        for nm, A in V.items():
            row = {f"macro_{m}": float(macro(A, rep, repos, m).mean()) for m in ("hit1", "hit5", "rr", "ndcg5", "rec5")}
            if nm != "EmemoVeri-2 (full)":
                d = macro(V["EmemoVeri-2 (full)"], rep, repos, "hit5") - macro(A, rep, repos, "hit5")
                row["full_minus"] = {"macro": rboot(d), "wins": int((d > 1e-9).sum()), "losses": int((d < -1e-9).sum()), "p": signflip_p(d), "dz": cohens_dz(d),
                                     "mrr": rboot(macro(V["EmemoVeri-2 (full)"], rep, repos, "rr") - macro(A, rep, repos, "rr"))}
                pv[nm] = row["full_minus"]["p"]
            block["variants"][nm] = row
        for nm, p in holm(pv).items():
            block["variants"][nm]["full_minus"]["p_holm"] = p
        R[tag] = block
        print(tag, {k: round(v["macro_hit5"], 3) for k, v in block["variants"].items()}, flush=True)
        for nm, v in block["variants"].items():
            if "full_minus" in v:
                f = v["full_minus"]; print(f"   {nm:24s} {f['macro'][0]:+.3f} [{f['macro'][1]:+.3f},{f['macro'][2]:+.3f}] W/L {f['wins']}/{f['losses']} holm {f['p_holm']:.4f}", flush=True)
        del D
    json.dump(R, open(out_p, "w"), indent=1)
elif mode == "sens":
    fresh_p, v06_p = sys.argv[4:6]
    D = pickle.load(open(fresh_p, "rb")); V06 = json.load(open(v06_p))
    rep = np.array([t.repo for t in D["tasks"]]); repos = sorted(set(rep))
    BLEND = [dict(w_sim=ws, l_pop=lp, l_rec=lr, H=h) for ws in (0.5, 1.0, 2.0) for lp in (0.0, 0.05, 0.15, 0.4) for lr, h in ((0.0, 90.0), (0.05, 30.0), (0.05, 365.0), (0.15, 30.0))]
    out = []
    t0 = time.time()
    for b in BLEND:
        A = run(D, replace(EV2, **b))
        out.append({"blend": b, "macro_hit5": float(macro(A, rep, repos, "hit5").mean()), "macro_rr": float(macro(A, rep, repos, "rr").mean())})
        print(b, round(out[-1]["macro_hit5"], 4), f"{time.time()-t0:.0f}s", flush=True)
    h5 = np.array([o["macro_hit5"] for o in out])
    ref = {k: V06["fresh16"]["systems"][k]["macro_hit5"] for k in ("Snapshot RAG", "EmemoVeri-T", "Sim+State+Rec+Pop", "EmemoVeri-2", "EmemoVeri-2 plain")}
    frozen = {"w_sim": EV2.w_sim, "l_pop": EV2.l_pop, "l_rec": EV2.l_rec, "H": EV2.H}
    fz = [o for o in out if all(abs(o["blend"][k] - v) < 1e-9 for k, v in frozen.items())][0]["macro_hit5"]
    R = {"grid": out, "min": float(h5.min()), "median": float(np.median(h5)), "max": float(h5.max()), "frozen": fz, "frozen_rank": int((h5 > fz + 1e-12).sum()) + 1,
         "n": len(out), "share_beating": {k: float((h5 > v).mean()) for k, v in ref.items() if k not in ("EmemoVeri-2", "EmemoVeri-2 plain")}, "reference": ref}
    json.dump(R, open(out_p, "w"), indent=1)
    print("min/median/max", R["min"], R["median"], R["max"], "frozen", fz, "rank", R["frozen_rank"], "of", len(out), R["share_beating"])

"""Analyses added in v0.3 (ladder, strong history baselines, backends, LORO, degradation, staleness
sweep, cluster bootstrap, lifecycle).  Input: pickle written by collect_ext.py.  Output: summary_ext.json."""
import itertools, json, pickle, sys, time, os
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, ".")
from ememoveri.realbench import Cond, rank_claims, tuning_grid
from ememoveri.realbench_ext import METRICS, metrics_vec

src = sys.argv[1] if len(sys.argv) > 1 else "collected.pkl"
out_path = sys.argv[2] if len(sys.argv) > 2 else "summary_ext.json"
D = pickle.load(open(src, "rb"))
tasks = D["tasks"]
shim = SimpleNamespace(path_of=D["path_of"], repo_of=D["repo_of"])
IDX = {m: k for k, m in enumerate(METRICS)}
dev = [t for t in tasks if t.split == "dev"]
test = [t for t in tasks if t.split == "test"]
print(len(dev), len(test), flush=True)
rng = np.random.default_rng(11)
T0 = time.time()

# ---------------------------------------------------------------- scoring family ----
LAMS = (0.0, 0.05, 0.15, 0.4)
HS = (30.0, 90.0, 365.0)


def family_grid(terms):
    """sim + lp*pop + le*ev + lt*trust + lr*recency(H); terms subset of {pop, ev, trust, rec}."""
    opts = []
    for k in ("pop", "ev", "trust"):
        opts.append(LAMS if k in terms else (0.0,))
    rec_opts = [(0.0, 90.0)] + [(l, h) for l in LAMS[1:] for h in HS] if "rec" in terms else [(0.0, 90.0)]
    return [dict(lp=a, le=b, lt=c, lr=lr, H=h) for a in opts[0] for b in opts[1] for c in opts[2] for lr, h in rec_opts]


def top_idx(t, cfg, backend, state=True):
    sim = t.sims[backend].astype(np.float64)
    F = t.F
    s = sim.copy()
    if cfg.get("lp"):
        s += cfg["lp"] * (t.n_touch / (t.n_touch + 5.0))
    if cfg.get("le"):
        s += cfg["le"] * (F.ev_total / (F.ev_total + 5.0))
    if cfg.get("lt"):
        s += cfg["lt"] * F.trust_seq
    if cfg.get("lr"):
        s += cfg["lr"] * np.power(2.0, -F.age_days / cfg["H"])
    mask = t.local & (~F.sup if state else True)
    idx = np.flatnonzero(mask)
    order = np.lexsort((-F.updated[idx], -s[idx]))[:20]
    return idx[order]


def eval_cfgs(ts, cfgs, backend="tfidf", state=True):
    A = np.zeros((len(cfgs), len(ts), len(METRICS)))
    for j, t in enumerate(ts):
        for g, cfg in enumerate(cfgs):
            A[g, j] = metrics_vec(shim, t.F, top_idx(t, cfg, backend, state), t.gold, t.q_repo)
    return A


def pick(A):
    """Best config index by mean MRR then Hit@5 (matches the published tuner)."""
    m = A.mean(1)
    return max(range(A.shape[0]), key=lambda g: (round(m[g, IDX["rr"]], 9), m[g, IDX["hit5"]]))


def cond_eval(ts, cond: Cond, backend="tfidf"):
    A = np.zeros((len(ts), len(METRICS)))
    for j, t in enumerate(ts):
        idx, _ = rank_claims(cond, t.sims[backend].astype(np.float64), t.F, t.local)
        A[j] = metrics_vec(shim, t.F, idx, t.gold, t.q_repo)
    return A


def summ(A):
    return {m: float(A[:, k].mean()) for m, k in IDX.items()}


# ---------------------------------------------------------------- bootstrap ----
def boot_task(d, B=10000):
    n = len(d)
    i = rng.integers(0, n, (B, n))
    m = d[i].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_cluster(d, labels, B=10000):
    u, inv = np.unique(labels, return_inverse=True)
    s = np.bincount(inv, weights=d, minlength=len(u))
    c = np.bincount(inv, minlength=len(u)).astype(float)
    k = rng.integers(0, len(u), (B, len(u)))
    m = s[k].sum(1) / c[k].sum(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), int(len(u))


lab_month = np.array([t.month for t in test])
lab_file = np.array([f"{t.repo}:{sorted(t.gold)[0][1]}" for t in test])

# ---------------------------------------------------------------- published conditions on new metrics ----
SIM = dict(w_sim=1.0, w_trust=0.0, w_ev=0.0, w_scope=0.0, w_stale=0.0)
C = {
    "RECENCY": Cond("RECENCY", kind="recency"), "POPULARITY": Cond("POPULARITY", kind="popularity"),
    "PLAIN_RAG": Cond("PLAIN_RAG", policy="open", state_aware=False, **SIM),
    "NAIVE": Cond("NAIVE", policy="local_only", state_aware=False, **SIM),
    "SNAPSHOT": Cond("SNAPSHOT", policy="local_only", state_aware=True, **SIM),
    "EV_DEFAULT": Cond("EV_DEFAULT"),
    "EV_TUNED": Cond("EV_TUNED", 0.8, 0.0, 0.1, 0.1, 0.05, 365.0, "seq"),
}
R = {"meta": {"n_dev": len(dev), "n_test": len(test), "counts": D["counts"], "resurrected": D["resurrected"],
              "n_superseded_final": D["n_superseded"], "n_events": len(D["events"])}}
PUB = {k: cond_eval(test, c) for k, c in C.items() if k != "PLAIN_RAG"}
PUB["PLAIN_RAG"] = None
R["published_test"] = {k: summ(v) for k, v in PUB.items() if v is not None}
print("published", {k: round(v["hit5"], 3) for k, v in R["published_test"].items()}, flush=True)

# ---------------------------------------------------------------- 1. component ladder ----
ladder_terms = [("R0 similarity (dead files kept)", None), ("R1 + state filter", []),
                ("R2 + frequency prior", ["pop"]), ("R3 + evidence", ["pop", "ev"]),
                ("R4 + trust", ["pop", "ev", "trust"]), ("R5 + recency", ["pop", "ev", "trust", "rec"])]
LAD = {}
ladder_rows = []
for name, terms in ladder_terms:
    if terms is None:
        A = eval_cfgs(test, [dict()], state=False)[0]
        cfg = {}
    else:
        grid = family_grid(terms)
        Ad = eval_cfgs(dev, grid)
        g = pick(Ad)
        cfg = grid[g]
        A = eval_cfgs(test, [cfg])[0]
    LAD[name] = A
    ladder_rows.append({"name": name, "cfg": cfg, **summ(A)})
    print("ladder", name, round(A[:, IDX["hit5"]].mean(), 3), cfg, f"{time.time()-T0:.0f}s", flush=True)
R["ladder"] = ladder_rows
names = [n for n, _ in ladder_terms]
R["ladder_steps"] = []
for a, b in zip(names[:-1], names[1:]):
    d = LAD[b][:, IDX["hit5"]] - LAD[a][:, IDX["hit5"]]
    R["ladder_steps"].append({"from": a, "to": b, "d_hit5": boot_task(d), "d_hit5_cluster": boot_cluster(d, lab_month)[:3],
                              "d_mrr": boot_task(LAD[b][:, IDX["rr"]] - LAD[a][:, IDX["rr"]])})

# ---------------------------------------------------------------- 2. strong history baselines ----
FAM = {"Sim+Recency": ["rec"], "Sim+Popularity": ["pop"], "Sim+Recency+Popularity": ["rec", "pop"]}
STRONG = {}
strong_rows = []
for fam, terms in FAM.items():
    for state in (False, True):
        grid = family_grid(terms)
        Ad = eval_cfgs(dev, grid, state=state)
        g = pick(Ad)
        A = eval_cfgs(test, [grid[g]], state=state)[0]
        nm = ("Sim+State" + fam[3:]) if state else fam
        STRONG[nm] = A
        strong_rows.append({"name": nm, "cfg": grid[g], **summ(A)})
        print("strong", nm, round(A[:, IDX["hit5"]].mean(), 3), flush=True)
R["strong"] = strong_rows
R["ladder_full_vs_strong"] = {}
ev_t = PUB["EV_TUNED"]
snap = PUB["SNAPSHOT"]
for nm, A in list(STRONG.items()) + [("R5", LAD[names[-1]])]:
    for ref_name, ref in (("EV_TUNED", ev_t),):
        d = ref[:, IDX["hit5"]] - A[:, IDX["hit5"]]
        R["ladder_full_vs_strong"][f"{ref_name}-{nm}"] = {"task": boot_task(d), "month": boot_cluster(d, lab_month),
                                                          "file": boot_cluster(d, lab_file)}
    d = A[:, IDX["hit5"]] - snap[:, IDX["hit5"]]
    R["ladder_full_vs_strong"][f"{nm}-SNAPSHOT"] = {"task": boot_task(d), "month": boot_cluster(d, lab_month),
                                                    "file": boot_cluster(d, lab_file)}

# ---------------------------------------------------------------- 3. cluster bootstrap for headline contrasts ----
R["cluster_boot"] = {}
for nm, (a, b) in {"EV_TUNED-SNAPSHOT": ("EV_TUNED", "SNAPSHOT"), "EV_DEFAULT-SNAPSHOT": ("EV_DEFAULT", "SNAPSHOT"),
                   "EV_TUNED-EV_DEFAULT": ("EV_TUNED", "EV_DEFAULT")}.items():
    for met in ("hit5", "rr", "ndcg5"):
        d = PUB[a][:, IDX[met]] - PUB[b][:, IDX[met]]
        R["cluster_boot"][f"{nm}:{met}"] = {"task": boot_task(d), "month": boot_cluster(d, lab_month),
                                            "file": boot_cluster(d, lab_file)}
print("boot done", f"{time.time()-T0:.0f}s", flush=True)

# ---------------------------------------------------------------- 4. retrieval backends ----
R["backends"] = {}
full_grid = family_grid(["pop", "ev", "trust", "rec"])
rp_grid = family_grid(["rec", "pop"])
for be in ("tfidf", "bm25", "dense"):
    row = {}
    A_s = eval_cfgs(test, [dict()], backend=be)[0]
    row["snapshot"] = summ(A_s)
    res = {"snapshot": A_s}
    for nm, grid in (("full", full_grid), ("recpop", rp_grid)):
        Ad = eval_cfgs(dev, grid, backend=be)
        g = pick(Ad)
        A = eval_cfgs(test, [grid[g]], backend=be)[0]
        res[nm] = A
        row[nm] = {**summ(A), "cfg": grid[g]}
    for nm in ("full", "recpop"):
        d = res[nm][:, IDX["hit5"]] - A_s[:, IDX["hit5"]]
        row[f"{nm}-snapshot"] = {"task": boot_task(d), "month": boot_cluster(d, lab_month)[:3]}
    d = res["full"][:, IDX["hit5"]] - res["recpop"][:, IDX["hit5"]]
    row["full-recpop"] = {"task": boot_task(d), "month": boot_cluster(d, lab_month)[:3]}
    R["backends"][be] = row
    print("backend", be, {k: round(v["hit5"], 3) for k, v in row.items() if isinstance(v, dict) and "hit5" in v}, flush=True)

# ---------------------------------------------------------------- 5. leave-one-repository-out ----
grid_pub = tuning_grid()


def pub_cfg(g):
    return Cond("g", g.w_sim, g.w_trust, g.w_ev, g.w_scope, g.w_stale, g.half_life, g.trust_mode)


def eval_pub(ts, conds):
    A = np.zeros((len(conds), len(ts), len(METRICS)))
    for j, t in enumerate(ts):
        sim = t.sims["tfidf"].astype(np.float64)
        for g, c in enumerate(conds):
            idx, _ = rank_claims(c, sim, t.F, t.local)
            A[g, j] = metrics_vec(shim, t.F, idx, t.gold, t.q_repo)
    return A


conds_pub = [pub_cfg(g) for g in grid_pub]
Aall = eval_pub(tasks, conds_pub)                              # [G, all tasks, metrics]
rep_of = np.array([t.repo for t in tasks])
split_of = np.array([t.split for t in tasks])
Arp_grid = family_grid(["rec", "pop"])
Arp_all = eval_cfgs(tasks, Arp_grid)
R["loro"] = {}
test_mask = split_of == "test"
for h in ("fastapi", "yt-dlp", "browser-use"):
    train = rep_of != h
    ev_g = pick(Aall[:, train])
    rp_g = pick(Arp_all[:, train])
    ho = (rep_of == h) & test_mask
    A_ev, A_rp = Aall[ev_g][ho], Arp_all[rp_g][ho]
    tl = [t for t in tasks if t.repo == h and t.split == "test"]
    A_sn = cond_eval(tl, C["SNAPSHOT"])
    A_df = cond_eval(tl, C["EV_DEFAULT"])
    gsel = grid_pub[ev_g]
    entry = {"n": int(ho.sum()), "n_train": int(train.sum()),
             "selected_weights": {"w_sim": gsel.w_sim, "w_trust": gsel.w_trust, "w_ev": gsel.w_ev, "w_stale": gsel.w_stale,
                                  "half_life": gsel.half_life, "trust_mode": gsel.trust_mode},
             "snapshot": summ(A_sn), "default": summ(A_df), "loro_tuned": summ(A_ev), "loro_recpop": {**summ(A_rp), "cfg": Arp_grid[rp_g]}}
    d = A_ev[:, IDX["hit5"]] - A_sn[:, IDX["hit5"]]
    entry["tuned-snapshot"] = {"task": boot_task(d), "month": boot_cluster(d, np.array([t.month for t in tl]))[:3]}
    d = A_rp[:, IDX["hit5"]] - A_sn[:, IDX["hit5"]]
    entry["recpop-snapshot"] = {"task": boot_task(d), "month": boot_cluster(d, np.array([t.month for t in tl]))[:3]}
    d = A_ev[:, IDX["hit5"]] - A_rp[:, IDX["hit5"]]
    entry["tuned-recpop"] = {"task": boot_task(d)}
    R["loro"][h] = entry
    print("loro", h, round(entry["snapshot"]["hit5"], 3), round(entry["loro_tuned"]["hit5"], 3), round(entry["loro_recpop"]["hit5"], 3), flush=True)

# ---------------------------------------------------------------- 6. query degradation ----
R["degrade"] = {}
rp_cfg = strong_rows[-1]["cfg"]
for lvl, be in (("0.0", "tfidf"), ("0.3", "tfidf@0.3"), ("0.5", "tfidf@0.5"), ("0.7", "tfidf@0.7")):
    row = {}
    for nm, c in (("snapshot", C["SNAPSHOT"]), ("default", C["EV_DEFAULT"]), ("tuned", C["EV_TUNED"])):
        row[nm] = summ(cond_eval(test, c, be))
    row["recpop"] = summ(eval_cfgs(test, [rp_cfg], backend=be)[0])
    A_t, A_s = cond_eval(test, C["EV_TUNED"], be), cond_eval(test, C["SNAPSHOT"], be)
    row["tuned-snapshot"] = boot_task(A_t[:, IDX["hit5"]] - A_s[:, IDX["hit5"]])
    row["tuned-snapshot-month"] = boot_cluster(A_t[:, IDX["hit5"]] - A_s[:, IDX["hit5"]], lab_month)[:3]
    R["degrade"][lvl] = row
    print("degrade", lvl, {k: round(v["hit5"], 3) for k, v in row.items() if isinstance(v, dict)}, flush=True)

# ---------------------------------------------------------------- 7. staleness sweep ----
R["stale_sweep"] = {}
for nm, base in (("blueprint", C["EV_DEFAULT"]), ("tuned", C["EV_TUNED"])):
    rows = []
    for hl in (None, 7.0, 30.0, 90.0, 180.0, 365.0, 730.0, 3650.0):
        c = Cond("s", base.w_sim, base.w_trust, base.w_ev, base.w_scope, 0.0 if hl is None else base.w_stale,
                 90.0 if hl is None else hl, base.trust_mode)
        A = cond_eval(test, c)
        Ad = cond_eval(dev, c)
        rows.append({"half_life": hl, "test": summ(A), "dev": summ(Ad)})
    R["stale_sweep"][nm] = rows
    print("stale", nm, [(r["half_life"], round(r["test"]["hit5"], 3)) for r in rows], flush=True)

# ---------------------------------------------------------------- 8. lifecycle ----
ev = D["events"]
life = {}
for typ in ("delete", "rename"):
    es = [e for e in ev if e["type"] == typ and "post" in e]
    n = len(es)
    entry = {"n": n}
    for sys_ in ("naive", "snapshot", "tuned"):
        entry[f"pre_hit_{sys_}"] = float(np.mean([e["pre"][sys_] for e in es]))
        entry[f"post_old_in_top5_{sys_}"] = float(np.mean([e["post"][f"carry:{sys_}"]["old_in_top5"] for e in es]))
    if typ == "rename":
        for k in ("carry:naive", "carry:snapshot", "carry:tuned", "nocarry:snapshot"):
            entry[f"post_new_in_top5_{k}"] = float(np.mean([e["post"][k]["new_in_top5"] for e in es]))
        entry["pre_hit_snapshot_given"] = entry["pre_hit_snapshot"]
        entry["n_pre_hit_snapshot"] = int(sum(e["pre"]["snapshot"] for e in es))
        entry["continuity_given_pre_hit_carry"] = float(np.mean(
            [e["post"]["carry:snapshot"]["new_in_top5"] for e in es if e["pre"]["snapshot"]])) if entry["n_pre_hit_snapshot"] else None
        entry["continuity_given_pre_hit_nocarry"] = float(np.mean(
            [e["post"]["nocarry:snapshot"]["new_in_top5"] for e in es if e["pre"]["snapshot"]])) if entry["n_pre_hit_snapshot"] else None
    life[typ] = entry
R["lifecycle"] = life
R["test_dead5"] = {"naive": float(np.mean([metrics_vec(shim, t.F, rank_claims(C["NAIVE"], t.sims["tfidf"].astype(float), t.F, t.local)[0], t.gold, t.q_repo)[IDX["dead5"]] for t in test])),
                   "snapshot": float(PUB["SNAPSHOT"][:, IDX["dead5"]].mean())}
json.dump(R, open(out_path, "w"), indent=1, default=float)
print("wrote", out_path, f"{time.time()-T0:.0f}s")

"""Top-5 files per task and memory condition (for agent_eval.py solve).
Usage: python experiments/export_retrievals.py issue_collected.pkl results/real_ext/summary_ext.json retrievals.json [query-name]
Conditions use the frozen weights chosen on the original dev split."""
import json, pickle, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ememoveri.realbench import Cond, rank_claims

D = pickle.load(open(sys.argv[1], "rb")); FZ = json.load(open(sys.argv[2])); qn = sys.argv[4] if len(sys.argv) > 4 else "full"
ssrp = {r["name"]: r["cfg"] for r in FZ["strong"]}["Sim+State+Recency+Popularity"]
SIM = dict(w_sim=1.0, w_trust=0.0, w_ev=0.0, w_scope=0.0, w_stale=0.0)
snap = Cond("SNAP", policy="local_only", state_aware=True, **SIM)
ev_t = Cond("EV_TUNED", 0.8, 0.0, 0.1, 0.1, 0.05, 365.0, "seq")


def top5(t, idx):
    seen, out = set(), []
    for i in idx:
        p = D["path_of"][i]
        if p not in seen:
            seen.add(p); out.append(p)
        if len(out) == 5:
            break
    return out


res = {}
for t in D["tasks"]:
    sim = t.sims[f"tfidf:{qn}"].astype(np.float64); F = t.F
    s = sim + ssrp["lp"] * (t.n_touch / (t.n_touch + 5.0)) + ssrp["lr"] * np.power(2.0, -F.age_days / ssrp["H"])
    idx = np.flatnonzero(t.local & ~F.sup)
    order = idx[np.lexsort((-F.updated[idx], -s[idx]))[:20]]
    res[t.sha] = {"snapshot": top5(t, rank_claims(snap, sim, F, t.local)[0]), "sim_state_rec_pop": top5(t, order),
                  "ememoveri_tuned": top5(t, rank_claims(ev_t, sim, F, t.local)[0])}
json.dump(res, open(sys.argv[3], "w"), indent=1)
print("wrote", len(res), "tasks")

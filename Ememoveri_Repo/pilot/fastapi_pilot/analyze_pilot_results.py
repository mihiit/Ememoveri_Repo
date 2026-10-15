"""
analyze_pilot_results.py

Turns pilot_grid_results.csv into paper-ready summary tables. It fixes three
problems with the raw CSV that would otherwise silently corrupt any analysis:

  1. MIXED SCHEMA: rows written before the `provider` column was added have 13
     fields; later rows have 14. Rows are parsed by length, not by header.
  2. RETRY DUPLICATES: a retried cell appends a new row and leaves the old
     API-error row in place. Only the LATEST row per
     (task, model, provider, condition) is kept.
  3. FALSE SUCCESSES: `git apply --check` only proves a patch is well-formed.
     In full_file mode a response truncated by max_tokens (finish_reason
     'length') becomes a shortened file, i.e. a diff that deletes the tail. It
     applies cleanly and is worthless. Two success levels are reported:
        lenient = git apply passes
        strict  = git apply passes AND the response was not truncated
     Use STRICT numbers in the paper; report lenient only as a reference.

Also reports Wilson 95% confidence intervals and an exact McNemar test for
paired condition comparisons. With n this small these are honest uncertainty
statements, not evidence of significance.

Usage:  python analyze_pilot_results.py [pilot_grid_results.csv]
Writes: pilot_grid_results_clean.csv
"""

import csv
import math
import sys
from collections import defaultdict

CONDS = ["NO_MEMORY", "PLAIN_RAG", "NAIVE_MEMORY", "EMEMOVERI"]


def load_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            if not r:
                continue
            if len(r) == 13:
                (task, cat, model, cond, mode, sha, fin, _ld, _hw, _ha, git_ok, err, notes) = r
                provider = "groq"
            elif len(r) == 14:
                (task, cat, model, provider, cond, mode, sha, fin, _ld, _hw, _ha, git_ok, err, notes) = r
            else:
                print(f"WARNING: skipping row with {len(r)} fields", file=sys.stderr)
                continue
            rows.append(dict(task=task, category=cat, model=model, provider=provider,
                             condition=cond, finish_reason=fin,
                             git_apply_ok=(git_ok == "True"), error_category=err))
    return rows


def dedupe_latest(rows):
    latest = {}
    for r in rows:  # file order == chronological order
        latest[(r["task"], r["model"], r["provider"], r["condition"])] = r
    return list(latest.values())


def classify(r):
    if r["error_category"] == "API error":
        return "infra_failure"
    if r["git_apply_ok"] and r["finish_reason"] == "length":
        return "truncated_false_success"
    if r["git_apply_ok"]:
        return "success"
    return "failure"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt(k, n):
    if n == 0:
        return "n=0"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k/n:5.1%}  (95% CI {lo:.0%}-{hi:.0%})"


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "pilot_grid_results.csv"
    raw = load_rows(path)
    rows = dedupe_latest(raw)
    for r in rows:
        r["outcome"] = classify(r)
    print(f"Raw rows: {len(raw)}  ->  unique cells after keeping latest attempt: {len(rows)}\n")

    with open("pilot_grid_results_clean.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "category", "model", "provider", "condition",
                    "finish_reason", "git_apply_ok", "error_category", "outcome"])
        for r in rows:
            w.writerow([r["task"], r["category"], r["model"], r["provider"], r["condition"],
                        r["finish_reason"], r["git_apply_ok"], r["error_category"], r["outcome"]])

    real = [r for r in rows if r["outcome"] != "infra_failure"]
    print("=" * 96)
    print("OVERALL OUTCOME COUNTS (unique cells)")
    print("=" * 96)
    counts = defaultdict(int)
    for r in rows:
        counts[r["outcome"]] += 1
    for k in ["success", "truncated_false_success", "failure", "infra_failure"]:
        print(f"  {k:26s} {counts[k]:4d}")

    print("\n" + "=" * 96)
    print("PER-MODEL  (denominator = real attempts, infra failures excluded)")
    print("=" * 96)
    by_model = defaultdict(list)
    for r in rows:
        by_model[f"{r['provider']}:{r['model']}"].append(r)
    for m, rs in sorted(by_model.items()):
        real_m = [r for r in rs if r["outcome"] != "infra_failure"]
        lenient = sum(1 for r in real_m if r["git_apply_ok"])
        strict = sum(1 for r in real_m if r["outcome"] == "success")
        infra = len(rs) - len(real_m)
        print(f"{m}\n   cells={len(rs)}  infra_failures={infra}")
        print(f"   lenient (git apply): {fmt(lenient, len(real_m))}")
        print(f"   STRICT             : {fmt(strict, len(real_m))}")

    print("\n" + "=" * 96)
    print("PER-CONDITION, WITHIN EACH MODEL  (STRICT success, real attempts only)")
    print("=" * 96)
    for m, rs in sorted(by_model.items()):
        real_m = [r for r in rs if r["outcome"] != "infra_failure"]
        if len(real_m) < 8:
            print(f"\n{m}: only {len(real_m)} real attempts -- too few to break down, skipped")
            continue
        print(f"\n{m}")
        for c in CONDS:
            cs = [r for r in real_m if r["condition"] == c]
            print(f"   {c:14s} {fmt(sum(1 for r in cs if r['outcome']=='success'), len(cs))}")

    print("\n" + "=" * 96)
    print("PAIRED COMPARISON: each condition vs NO_MEMORY (same task + model, STRICT success)")
    print("Exact McNemar on discordant pairs. UNDERPOWERED: treat as descriptive only.")
    print("=" * 96)
    idx = {(r["task"], r["model"], r["provider"], r["condition"]): r for r in real}
    for c in ["PLAIN_RAG", "NAIVE_MEMORY", "EMEMOVERI"]:
        both = better = worse = same = 0
        for (t, m, p, cond), r in idx.items():
            if cond != c:
                continue
            base = idx.get((t, m, p, "NO_MEMORY"))
            if base is None:
                continue
            both += 1
            a, b = r["outcome"] == "success", base["outcome"] == "success"
            if a and not b:
                better += 1
            elif b and not a:
                worse += 1
            else:
                same += 1
        p = mcnemar_exact(better, worse)
        print(f"  {c:13s} vs NO_MEMORY: paired cells={both:3d}  better={better}  worse={worse}  "
              f"tied={same}  exact McNemar p={p:.3f}")

    print("\n" + "=" * 96)
    print("ERROR TAXONOMY (real attempts only)")
    print("=" * 96)
    tax = defaultdict(int)
    for r in real:
        key = "Truncated output (applies, but destructive)" if r["outcome"] == "truncated_false_success" \
            else r["error_category"]
        tax[key] += 1
    tot = sum(tax.values())
    for k, v in sorted(tax.items(), key=lambda x: -x[1]):
        print(f"  {k:52s} {v:3d} ({v/tot:.1%})")
    print("\nWrote pilot_grid_results_clean.csv")


if __name__ == "__main__":
    main()
"""
compare_paired.py — paired comparison between two conditions.

Matches runs by scenario_index so each pair faced the identical car and identical
hidden reservations. A paired test removes scenario difficulty as a source of
variance, which is exactly what made the earlier v3 vs v3.1 comparison
uninterpretable, so it detects a real difference with a much smaller sample than
comparing two independent batches would.

Usage:
    python compare_paired.py                                  # reads runs/
    python compare_paired.py runs seller_v4 seller_vanilla
"""

import os
import sys
import json
import math
import statistics as st
from collections import defaultdict


def load_runs(folder="runs"):
    runs = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(folder, name)) as f:
                runs.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return runs


def share(run, no_deal_is_zero=True):
    """Seller's share of the ZOPA. A no-deal captures zero surplus for the seller,
    which is the honest way to score it: the seller keeps the car and ends at its
    own reservation."""
    sc = run["scenario"]
    zopa = sc["buyer_reservation"] - sc["seller_reservation"]
    if run.get("outcome") != "deal" or run.get("final_price") is None:
        return 0.0 if no_deal_is_zero else None
    return (run["final_price"] - sc["seller_reservation"]) / zopa


def summarize(runs, label):
    deals = [r for r in runs if r.get("outcome") == "deal"]
    cond_share = [share(r) for r in deals]
    exp_share = [share(r) for r in runs]
    print(f"  {label}")
    print(f"    n = {len(runs)}, deals = {len(deals)} ({100*len(deals)/len(runs):.0f}%)")
    if cond_share:
        print(f"    share GIVEN a deal        : {100*st.mean(cond_share):.1f}%")
    print(f"    EXPECTED share per attempt: {100*st.mean(exp_share):.1f}%")
    return exp_share


def paired_test(a_runs, b_runs, a_label, b_label):
    a_by = {r.get("scenario_index"): r for r in a_runs if r.get("scenario_index") is not None}
    b_by = {r.get("scenario_index"): r for r in b_runs if r.get("scenario_index") is not None}
    common = sorted(set(a_by) & set(b_by))
    if not common:
        print("  no shared scenario_index values — cannot pair. "
              "Falling back to unpaired comparison only.")
        return

    diffs = [share(a_by[i]) - share(b_by[i]) for i in common]
    n = len(diffs)
    mean_d = st.mean(diffs)
    sd_d = st.stdev(diffs) if n > 1 else 0.0
    se_d = sd_d / math.sqrt(n) if n > 1 else 0.0

    print()
    print("=" * 70)
    print(f"PAIRED COMPARISON on {n} shared scenarios")
    print("=" * 70)
    print(f"  mean difference ({a_label} - {b_label}): {100*mean_d:+.1f} pts")
    print(f"  sd of differences: {100*sd_d:.1f}   paired SE: {100*se_d:.1f}")
    if se_d:
        t = mean_d / se_d
        print(f"  t = {t:.2f} on {n-1} df")
        verdict = ("clearly real" if abs(t) > 3
                   else "likely real" if abs(t) > 2
                   else "NOT distinguishable from noise")
        print(f"  -> {verdict}")
        lo, hi = mean_d - 1.96 * se_d, mean_d + 1.96 * se_d
        print(f"  95% CI on the difference: {100*lo:+.1f} to {100*hi:+.1f} pts")
    wins = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    print(f"  {a_label} won {wins}/{n} scenarios, lost {n-wins-ties}, tied {ties}")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "runs"
    a_label = sys.argv[2] if len(sys.argv) > 2 else "seller_v4"
    b_label = sys.argv[3] if len(sys.argv) > 3 else "seller_vanilla"

    runs = load_runs(folder)
    by_cond = defaultdict(list)
    for r in runs:
        by_cond[r.get("condition") or "unlabeled"].append(r)

    print("CONDITIONS FOUND:", {k: len(v) for k, v in by_cond.items()})
    print()
    a = by_cond.get(a_label, [])
    b = by_cond.get(b_label, [])
    if not a or not b:
        print(f"need both '{a_label}' and '{b_label}' present; found "
              f"{len(a)} and {len(b)}")
        sys.exit(0)

    print("UNPAIRED SUMMARIES (all runs in each condition)")
    summarize(a, a_label)
    summarize(b, b_label)
    paired_test(a, b, a_label, b_label)

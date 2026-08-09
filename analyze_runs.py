"""
analyze_runs.py — batch readout for the negotiation harness.

Reports the metrics that actually distinguish a good distributive negotiator from
a split-the-difference machine:

  seller share of ZOPA   how much of the bargaining zone the seller captured
  final vs midpoint      did it drag the deal off the midpoint of the two openings?
  hold economics         what a hold buys vs what a concession buys (the key number)
  anchor premium         how aggressive the opening was, and does it correlate with share
  estimate accuracy      how close the seller's guess at the buyer's ceiling actually was
  money left behind      buyer_reservation - final_price

Run:  python analyze_runs.py            (reads runs/)
      python analyze_runs.py runs_v4    (reads a specific folder)
"""

import os
import sys
import json
import re
import math
import statistics as st
from collections import defaultdict


def load_runs(folder="runs"):
    runs = []
    if not os.path.isdir(folder):
        print(f"no such folder: {folder}")
        return runs
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(folder, name)) as f:
                r = json.load(f)
            r["_filename"] = name
            runs.append(r)
        except (json.JSONDecodeError, OSError):
            continue
    return runs


def seller_share(run):
    if run.get("outcome") != "deal" or run.get("final_price") is None:
        return None
    sc = run["scenario"]
    zopa = sc["buyer_reservation"] - sc["seller_reservation"]
    if not zopa:
        return None
    return (run["final_price"] - sc["seller_reservation"]) / zopa


def first_dollar(text):
    """First plausible dollar figure in free text (requires a literal $ so we
    don't grab the model year or mileage)."""
    if not text:
        return None
    for m in re.findall(r'\$\s?([\d]{1,3}(?:,\d{3})*(?:\.\d+)?)', text):
        try:
            v = float(m.replace(",", ""))
        except ValueError:
            continue
        if v >= 1000:
            return v
    return None


def opening(run, who):
    return next((t["number"] for t in run["turns"]
                 if t["speaker"] == who and t["number"]), None)


def hold_economics(runs):
    """For every seller turn, pair the seller's move with the buyer's response.
    Returns (hold_gains, concession_pairs) where a pair is (seller_gave, buyer_moved)."""
    holds, concessions = [], []
    for r in runs:
        ts = r["turns"]
        for i in range(len(ts) - 1):
            a, b = ts[i], ts[i + 1]
            if a["speaker"] != "seller" or not a["number"]:
                continue
            if b["speaker"] != "buyer" or not b["number"]:
                continue
            s_prev = [t["number"] for t in ts[:i] if t["speaker"] == "seller" and t["number"]]
            b_prev = [t["number"] for t in ts[:i + 1] if t["speaker"] == "buyer" and t["number"]]
            if not s_prev or not b_prev:
                continue
            s_move = s_prev[-1] - a["number"]      # +ve = seller came down
            b_move = b["number"] - b_prev[-1]      # +ve = buyer came up
            if s_move == 0:
                holds.append(b_move)
            elif s_move > 0:
                concessions.append((s_move, b_move))
    return holds, concessions


def corr(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    sx, sy = st.pstdev(xs), st.pstdev(ys)
    if not sx or not sy:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / len(xs) / (sx * sy)


def report(runs, label):
    print("=" * 78)
    print(f"{label}   ({len(runs)} runs)")
    print("=" * 78)

    outcomes = defaultdict(int)
    for r in runs:
        outcomes[r.get("outcome", "unknown")] += 1
    print("  outcomes:", dict(outcomes))

    deals = [r for r in runs if r.get("outcome") == "deal"]
    shares = [seller_share(r) for r in deals]
    shares = [s for s in shares if s is not None]
    if not shares:
        print("  no deals to analyze")
        return None

    sd = st.pstdev(shares) if len(shares) > 1 else 0.0
    se = sd / math.sqrt(len(shares)) if shares else 0.0
    print(f"  seller share of ZOPA: {100*st.mean(shares):.1f}%  (sd {100*sd:.0f}, "
          f"se {100*se:.1f}, n={len(shares)})")
    print(f"    95% CI roughly {100*(st.mean(shares)-1.96*se):.1f}% to {100*(st.mean(shares)+1.96*se):.1f}%")

    # final vs midpoint of openings
    vs_mid = []
    prem, sh_for_prem = [], []
    for r in deals:
        bo, so = opening(r, "buyer"), opening(r, "seller")
        if bo and so:
            vs_mid.append(r["final_price"] - (bo + so) / 2)
        fv = r["scenario"].get("fair_value")
        if so and fv:
            prem.append(100 * (so - fv) / fv)
            sh_for_prem.append(seller_share(r))
    if vs_mid:
        print(f"  final vs midpoint of openings: {st.mean(vs_mid):+.0f}   "
              f"(0 = pure split-the-difference)")
    if prem:
        c = corr(prem, sh_for_prem)
        print(f"  seller opening premium over fair: {st.mean(prem):.1f}%  "
              f"(range {min(prem):.1f}–{max(prem):.1f}%)"
              + (f"   corr with share: {c:+.2f}" if c is not None else ""))

    # hold vs concede economics
    holds, conc = hold_economics(runs)
    if holds:
        moved = sum(1 for h in holds if h > 0)
        print(f"  HOLD:    cost $0    -> buyer moves {st.mean(holds):+.0f}   "
              f"(n={len(holds)}, buyer moved {moved}/{len(holds)})")
    if conc:
        g = st.mean([s for s, _ in conc])
        b = st.mean([m for _, m in conc])
        print(f"  CONCEDE: cost ${g:.0f}  -> buyer moves {b:+.0f}   net {b-g:+.0f}   (n={len(conc)})")
    if holds and conc:
        g = st.mean([s for s, _ in conc]); b = st.mean([m for _, m in conc])
        print(f"  => holding is worth ${st.mean(holds)-(b-g):.0f} more per turn than conceding")

    # seller action mix
    acts = defaultdict(int)
    for r in runs:
        for t in r["turns"]:
            if t["speaker"] == "seller":
                acts[t["action"]] += 1
    print("  seller actions:", dict(acts))

    # round numbers
    s_round = s_tot = 0
    for r in runs:
        for t in r["turns"]:
            if t["speaker"] == "seller" and t["number"]:
                s_tot += 1
                s_round += (t["number"] % 100 == 0)
    if s_tot:
        print(f"  seller round-hundred offers: {s_round}/{s_tot} ({100*s_round/s_tot:.0f}%)")

    # estimate accuracy vs the hidden buyer reservation
    errs = []
    for r in deals:
        real = r["scenario"]["buyer_reservation"]
        seq = [first_dollar(t.get("opponent_limit_estimate"))
               for t in r["turns"] if t["speaker"] == "seller"]
        seq = [v for v in seq if v]
        if seq:
            errs.append(seq[-1] - real)
    if errs:
        print(f"  final estimate of buyer ceiling vs truth: {st.mean(errs):+.0f} "
              f"(negative = underestimating), n={len(errs)}")

    # turn budget + money left
    print(f"  turns used: mean {st.mean([r['num_turns'] for r in deals]):.1f} of 21 available")
    left = [r["scenario"]["buyer_reservation"] - r["final_price"] for r in deals]
    print(f"  money left on table: ${st.mean(left):.0f} of a $750 zone "
          f"({100*st.mean(left)/750:.0f}%)")
    return shares


def compare(a_shares, b_shares, a_label, b_label):
    if not a_shares or not b_shares:
        return
    print("=" * 78)
    print(f"COMPARISON: {a_label} vs {b_label}")
    print("=" * 78)
    ma, mb = st.mean(a_shares), st.mean(b_shares)
    sa = st.pstdev(a_shares) if len(a_shares) > 1 else 0
    sb = st.pstdev(b_shares) if len(b_shares) > 1 else 0
    se = math.sqrt(sa**2 / len(a_shares) + sb**2 / len(b_shares))
    diff = ma - mb
    print(f"  {a_label}: {100*ma:.1f}%   {b_label}: {100*mb:.1f}%")
    print(f"  difference: {100*diff:+.1f} pts,  pooled SE {100*se:.1f} pts")
    if se:
        z = diff / se
        print(f"  that is {z:.2f} standard errors "
              f"({'likely real' if abs(z) > 2 else 'NOT distinguishable from noise'})")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "runs"
    runs = load_runs(folder)
    if not runs:
        print("no runs found")
        sys.exit(0)

    # If records carry a "condition" label, report each separately and compare.
    by_cond = defaultdict(list)
    for r in runs:
        by_cond[r.get("condition") or "unlabeled"].append(r)

    results = {}
    for cond, rs in by_cond.items():
        results[cond] = report(rs, f"CONDITION: {cond}")
        print()

    conds = [c for c in results if results[c]]
    if len(conds) == 2:
        compare(results[conds[0]], results[conds[1]], conds[0], conds[1])
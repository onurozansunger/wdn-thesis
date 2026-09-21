"""Paired significance tests over seeds.

Every routing rule is scored on the *same* checkpoints, so the comparison
is paired and a paired test is both valid and far more powerful than
eyeballing whether two error bars overlap. The paper previously asserted a
0.005 gap was "real" without testing it; this supplies the test.

Reports, for each pair, the mean paired difference, a Wilcoxon signed-rank
p-value (no normality assumption) and a paired t-test p-value.

Usage:
    python3 scripts/paired_tests.py runs/temporal_moe/workshop_routing.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PAIRS = [("soft", "cascade"), ("soft", "router_top1"),
         ("cascade", "router_top1"), ("oracle", "cascade")]


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    out = {"mean_diff": float(d.mean()), "n": int(len(d))}
    # Wilcoxon is undefined when every difference is zero.
    if np.allclose(d, 0):
        out["wilcoxon_p"] = 1.0
        out["ttest_p"] = 1.0
        return out
    out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
    out["ttest_p"] = float(stats.ttest_rel(a, b).pvalue)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="?",
                    default="runs/temporal_moe/workshop_routing.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p = ROOT / args.json
    runs = json.load(open(p))["runs"]
    by = {k: np.array([r[k]["f1"] for r in runs])
          for k in ("soft", "router_top1", "cascade", "oracle")}

    res = {}
    print(f"n = {len(runs)} paired seeds\n")
    print(f"  {'comparison':<26}{'mean diff':>11}{'Wilcoxon p':>13}{'t-test p':>11}")
    for a, b in PAIRS:
        r = paired(by[a], by[b])
        res[f"{a}_vs_{b}"] = r
        star = "*" if r["wilcoxon_p"] < 0.05 else " "
        print(f"  {a} - {b:<18}{r['mean_diff']:>+11.4f}"
              f"{r['wilcoxon_p']:>13.4f}{r['ttest_p']:>11.4f} {star}")

    out = ROOT / (args.out or str(p).replace(".json", "_tests.json"))
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

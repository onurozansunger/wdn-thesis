"""Aggregate Modena Part-1 10-seed runs for replay_weight=1.0 vs 2.5.

Produces a single table summarising the Pareto trade-off between overall
pressure F1 and per-attack replay F1.  Reads runs/temporal_moe/* and
filters by (hidden_dim=64, router_hidden_dim=24) for both replay-weight
configurations.

Usage:
    python3 scripts/compare_rw_sweep.py [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict


def collect():
    """Return {(rw,): {seed: test_results_dict}}."""
    groups: dict[float, dict[int, dict]] = defaultdict(dict)
    for d in sorted(glob.glob("runs/temporal_moe/2026*")):
        args_p = os.path.join(d, "args.json")
        res_p = os.path.join(d, "test_results.json")
        if not (os.path.exists(args_p) and os.path.exists(res_p)):
            continue
        a = json.load(open(args_p))
        if a.get("hidden_dim") != 64 or a.get("router_hidden_dim") != 24:
            continue
        rw = float(a.get("replay_weight", 1.0))
        seed = a.get("seed")
        if seed is None or seed < 1 or seed > 10:
            continue
        # Keep the most recent run per (rw, seed).
        prev = groups[rw].get(seed)
        if prev is None or os.path.getmtime(res_p) > os.path.getmtime(
            os.path.join(prev["_dir"], "test_results.json")
        ):
            tr = json.load(open(res_p))
            tr["_dir"] = d
            groups[rw][seed] = tr
    return groups


def summarise(rw_results: dict[int, dict]) -> dict:
    def vals(path):
        out = []
        for tr in rw_results.values():
            x = tr
            for k in path:
                x = x[k]
            out.append(x)
        return out

    def ms(v):
        if len(v) < 2:
            return {"mean": v[0] if v else None, "std": 0.0, "n": len(v)}
        return {"mean": st.mean(v), "std": st.stdev(v), "n": len(v)}

    return {
        "n_seeds": len(rw_results),
        "seeds": sorted(rw_results.keys()),
        "pressure_f1": ms(vals(["anomaly_detection", "pressure", "f1"])),
        "pressure_auroc": ms(vals(["anomaly_detection", "pressure", "auroc"])),
        "flow_f1": ms(vals(["anomaly_detection", "flow", "f1"])),
        "router_acc": ms(vals(["router_acc"])),
        "per_attack": {
            attack: ms(vals(["per_attack_pressure", attack, "f1"]))
            for attack in ["random", "replay", "stealthy", "noise", "targeted"]
        },
    }


def fmt(ms_dict):
    if ms_dict["mean"] is None:
        return "    ---"
    return f"{ms_dict['mean']:.4f} ± {ms_dict['std']:.4f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default=None, help="write summary to JSON")
    args = p.parse_args()

    groups = collect()
    rws = sorted(groups.keys())
    summaries = {rw: summarise(groups[rw]) for rw in rws}

    print(f"{'metric':<30} " + " ".join(f"{f'rw={rw}':>22}" for rw in rws))
    print("-" * (32 + 23 * len(rws)))
    print(
        f"{'n_seeds':<30} "
        + " ".join(f"{str(summaries[rw]['n_seeds']):>22}" for rw in rws)
    )
    for metric in ["pressure_f1", "pressure_auroc", "flow_f1", "router_acc"]:
        print(
            f"{metric:<30} "
            + " ".join(f"{fmt(summaries[rw][metric]):>22}" for rw in rws)
        )
    print("\nper-attack pressure F1:")
    for attack in ["random", "replay", "stealthy", "noise", "targeted"]:
        print(
            f"  {attack:<28} "
            + " ".join(
                f"{fmt(summaries[rw]['per_attack'][attack]):>22}" for rw in rws
            )
        )

    if len(rws) >= 2:
        rw_lo, rw_hi = rws[0], rws[-1]
        df1 = (
            summaries[rw_hi]["pressure_f1"]["mean"]
            - summaries[rw_lo]["pressure_f1"]["mean"]
        )
        drep = (
            summaries[rw_hi]["per_attack"]["replay"]["mean"]
            - summaries[rw_lo]["per_attack"]["replay"]["mean"]
        )
        print(
            f"\nTrade-off (rw {rw_lo} -> {rw_hi}): "
            f"overall F1 Δ={df1:+.4f}, replay F1 Δ={drep:+.4f}"
        )

    if args.json:
        json.dump(summaries, open(args.json, "w"), indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()

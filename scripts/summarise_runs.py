"""Print a compact summary table of all training runs.

Walks ``runs/`` and pulls the headline metrics out of each ``test_results.json``
so the latest set of experiments can be compared at a glance.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
RUNS = ROOT / "runs"


def fmt(x: float | None, n: int = 3) -> str:
    if x is None:
        return "  -"
    return f"{x:.{n}f}"


def get_replay(res: dict) -> tuple[float | None, float | None]:
    pa = res.get("per_attack_pressure", {}) or {}
    rep = pa.get("replay") or {}
    return rep.get("f1"), rep.get("auroc")


def get_overall(res: dict) -> tuple[float | None, float | None, float | None]:
    rec = (res.get("reconstruction", {}) or {}).get("pressure_unobs", {}) or {}
    anom = (res.get("anomaly_detection", {}) or {}).get("pressure", {}) or {}
    return rec.get("mae"), anom.get("f1"), anom.get("auroc")


def collect():
    rows = []
    for category in sorted(RUNS.iterdir()):
        if not category.is_dir():
            continue
        for run in sorted(category.iterdir()):
            tr = run / "test_results.json"
            ar = run / "args.json"
            if not tr.exists():
                continue
            try:
                res = json.load(open(tr))
            except json.JSONDecodeError:
                continue
            args = {}
            if ar.exists():
                try:
                    args = json.load(open(ar))
                except json.JSONDecodeError:
                    pass
            mae, f1, auroc = get_overall(res)
            rep_f1, rep_auroc = get_replay(res)
            rows.append({
                "category": category.name,
                "run": run.name,
                "data": Path(args.get("data_dir", "?")).name,
                "gnn": args.get("gnn_type", "?"),
                "mae": mae,
                "f1": f1,
                "auroc": auroc,
                "rep_f1": rep_f1,
                "rep_auroc": rep_auroc,
            })
    return rows


def print_table(rows):
    header = f"{'category':18} {'run':18} {'data':24} {'gnn':12} " \
             f"{'P_MAE':>7} {'F1':>6} {'AUROC':>6} " \
             f"{'rep_F1':>7} {'rep_AUR':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['category']:18} {r['run']:18} {r['data']:24} {r['gnn']:12} "
            f"{fmt(r['mae']):>7} {fmt(r['f1']):>6} {fmt(r['auroc']):>6} "
            f"{fmt(r['rep_f1']):>7} {fmt(r['rep_auroc']):>8}"
        )


if __name__ == "__main__":
    rows = collect()
    print_table(rows)
    print(f"\n{len(rows)} runs total.")

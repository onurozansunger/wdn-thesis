"""Turn the architecture sweep into the JSONs the deliverables script reads.

Reads /tmp/wdn_queue/arch_*.log to map each experiment name to its run
directory, evaluates every routing scheme on those checkpoints, and writes:

    runs/temporal_moe/expert_sweep.json   lambda_expert -> {soft, oracle, cascade, top1}
    runs/temporal_moe/norm_compare.json   [{config, overall_f1, replay_f1}, ...]
    runs/temporal_moe/cascade_eval.json   full per-run cascade evaluation

    python3 scripts/aggregate_arch_experiments.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = Path("/tmp/wdn_queue")
RUNS = ROOT / "runs" / "temporal_moe"


def run_dir_for(name: str) -> Path | None:
    """Find the run directory a given experiment wrote to."""
    lg = LOG / f"arch_{name}.log"
    if not lg.exists():
        return None
    m = re.findall(r"runs/temporal_moe/[0-9_]+", lg.read_text())
    if not m:
        return None
    d = ROOT / m[-1]
    return d if (d / "best_model.pt").exists() else None


def test_metrics(d: Path) -> dict | None:
    p = d / "test_results.json"
    if not p.exists():
        return None
    t = json.load(open(p))
    return {
        "overall_f1": t["anomaly_detection"]["pressure"]["f1"],
        "replay_f1": t["per_attack_pressure"]["replay"]["f1"],
        "auroc": t["anomaly_detection"]["pressure"]["auroc"],
        "router_acc": t.get("router_acc"),
    }


def eval_cascade(dirs: list[Path], norm_mode: str, out: Path) -> list[dict]:
    """Shell out to eval_cascade.py for the routing-scheme comparison."""
    if not dirs:
        return []
    cmd = [sys.executable, str(ROOT / "scripts" / "eval_cascade.py"),
           "--data_dir", "data/temporal_moe_modena",
           "--norm_mode", norm_mode,
           "--out", str(out.relative_to(ROOT)),
           "--runs", *[str(d) for d in dirs]]
    subprocess.run(cmd, cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.load(open(out))


def main():
    # --- A) lambda_expert sweep (global normalisation) ---
    sweep_cfg = {"0.5": "le0.5_s1", "2.0": "le2_s1",
                 "4.0": "le4_s1", "8.0": "le8_s1"}
    dirs, keys = [], []
    for lam, name in sweep_cfg.items():
        d = run_dir_for(name)
        if d:
            dirs.append(d); keys.append(lam)
        else:
            print(f"  missing run for lambda_expert={lam} ({name})")

    sweep = {}
    if dirs:
        res = eval_cascade(dirs, "global", RUNS / "cascade_eval.json")
        for lam, r in zip(keys, res):
            sweep[lam] = {v: r[v]["f1"]
                          for v in ("soft", "router_top1", "cascade", "oracle")}
        json.dump(sweep, open(RUNS / "expert_sweep.json", "w"), indent=2)
        print(f"wrote expert_sweep.json ({len(sweep)} points)")
        for lam in keys:
            s = sweep[lam]
            print(f"  lambda_expert={lam}: soft {s['soft']:.3f} | "
                  f"top1 {s['router_top1']:.3f} | cascade {s['cascade']:.3f} | "
                  f"oracle {s['oracle']:.3f}")

    # --- B) normalisation comparison ---
    norm_cfg = [
        ("global, lambda_e=0.5", "le0.5_s1"),
        ("global, lambda_e=4",   "le4_s1"),
        ("per-node, lambda_e=0.5", "norm_s1"),
        ("per-node, lambda_e=4",   "norm_le4_s1"),
    ]
    rows = []
    for label, name in norm_cfg:
        d = run_dir_for(name)
        if not d:
            print(f"  missing run for {label} ({name})")
            continue
        m = test_metrics(d)
        if m:
            rows.append({"config": label,
                         "overall_f1": round(m["overall_f1"], 4),
                         "replay_f1": round(m["replay_f1"], 4),
                         "auroc": round(m["auroc"], 4)})
    if rows:
        json.dump(rows, open(RUNS / "norm_compare.json", "w"), indent=2)
        print(f"\nwrote norm_compare.json ({len(rows)} configs)")
        for r in rows:
            print(f"  {r['config']:<24} overall {r['overall_f1']:.3f} | "
                  f"replay {r['replay_f1']:.3f}")


if __name__ == "__main__":
    main()

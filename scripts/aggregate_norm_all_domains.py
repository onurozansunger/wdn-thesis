"""Aggregate global vs per-node results for all 3 domains into one JSON.

Reads the 5-seed runs per (domain, norm_mode) and writes
runs/temporal_moe/norm_all_domains.json with mean overall F1, replay F1
and replay AUROC — the data behind the 'does the normalisation fix
generalise?' figure.
"""
import json, glob, os, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOMAINS = {"data/temporal_moe_modena": "Water",
           "data/temporal_moe_power": "Power",
           "data/temporal_moe_traffic": "Traffic"}

groups = {}
for d in sorted(glob.glob(str(ROOT / "runs/temporal_moe/2026*"))):
    ap, tp = os.path.join(d, "args.json"), os.path.join(d, "test_results.json")
    if not (os.path.exists(ap) and os.path.exists(tp)):
        continue
    a = json.load(open(ap))
    dom = DOMAINS.get(a.get("data_dir"))
    if dom is None or a.get("epochs") != 60 or a.get("seed") not in range(1, 6):
        continue
    nm = a.get("norm_mode", "global")
    groups.setdefault((dom, nm), {})[a["seed"]] = json.load(open(tp))

out = {}
for (dom, nm), runs in groups.items():
    r = list(runs.values())
    out.setdefault(dom, {})[nm] = {
        "n_seeds": len(r),
        "overall_f1": st.mean([x["anomaly_detection"]["pressure"]["f1"] for x in r]),
        "overall_f1_std": st.stdev([x["anomaly_detection"]["pressure"]["f1"] for x in r]) if len(r) > 1 else 0.0,
        "replay_f1": st.mean([x["per_attack_pressure"]["replay"]["f1"] for x in r]),
        "replay_auroc": st.mean([x["per_attack_pressure"]["replay"]["auroc"] for x in r]),
    }

json.dump(out, open(ROOT / "runs/temporal_moe/norm_all_domains.json", "w"), indent=2)
for dom, modes in out.items():
    for nm, v in modes.items():
        print(f"  {dom:<8} {nm:<9} n={v['n_seeds']} F1={v['overall_f1']:.3f} "
              f"replay_AUROC={v['replay_auroc']:.3f}")

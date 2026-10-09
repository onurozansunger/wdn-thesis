"""Measure each expert independently, using calibration and validation only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
import torch
from torch.utils.data import DataLoader

from wdn.models.mechanism_moe import FAMILY_NAMES
from wdn.temporal_dataset import TemporalWDNDataset, temporal_collate_fn
from wdn.train_operational_moe import TrainingConfig, build_model, _forward, _binary_counts, summarise
from wdn.train_temporal_moe import _to_device


def best_f1_threshold(scores, labels):
    """Exact F1 threshold on calibration; caller decides the diagnostic slice."""
    if not np.any(labels):
        return 1.
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], 1e-12)
    # precision_recall_curve uses >=; the deployment convention is >.
    selected = thresholds[np.argmax(f1)]
    return float(np.nextafter(selected, np.array(-np.inf, dtype=scores.dtype)))


@torch.no_grad()
def collect_experts(model, dataset, cfg):
    loader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=temporal_collate_fn)
    model.eval()
    values = {k: [] for k in ("experts", "mixture", "uniform", "oracle", "labels", "families", "routing")}
    for raw in loader:
        batch = _to_device(raw, torch.device(cfg.device))
        batch["x_seq"] = batch["x_seq"][-cfg.long_window:]
        out = _forward(model, batch)
        observed = batch["pressure_mask"] > 0
        families = batch["attack_type"].repeat_interleave(batch["num_nodes"])
        logits = out["expert_pressure_anomaly_logits"]
        values["experts"].append(logits[observed].sigmoid().cpu().numpy())
        values["mixture"].append(out["pressure_anomaly_logits"][observed].sigmoid().cpu().numpy())
        values["uniform"].append(logits.mean(-1)[observed].sigmoid().cpu().numpy())
        owner = logits.gather(1, families[:, None]).squeeze(1)
        values["oracle"].append(owner[observed].sigmoid().cpu().numpy())
        values["labels"].append(batch["pressure_anomaly"][observed].cpu().numpy())
        values["families"].append(families[observed].cpu().numpy())
        values["routing"].append(out["router_probs"].repeat_interleave(batch["num_nodes"], 0)[observed].cpu().numpy())
    return {k: np.concatenate(v) for k, v in values.items()}


def write_audit_markdown(report, run_dir):
    lines = ["# Independent expert audit", "",
             "Calibration and validation only; the locked test is not evaluated.",
             "Each expert uses a single global threshold selected on calibration.", "",
             "## Per-family AUPRC (threshold independent)", "",
             "| Expert | Random | Replay | Drift | Noise | Targeted | Clean FPR |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in report["expert_matrix"].items():
        fields = [f"{row['per_family'][family]['auprc']:.4f}" for family in FAMILY_NAMES[1:]]
        lines.append(f"| {name} | " + " | ".join(fields) +
                     f" | {row['per_family']['clean']['fpr']:.5f} |")
    wins = 0
    for family in FAMILY_NAMES[1:]:
        best = max(row["per_family"][family]["auprc"] for row in report["expert_matrix"].values())
        wins += report["expert_matrix"][family]["per_family"][family]["auprc"] >= best
    lines += ["", f"Owning experts rank first (including ties): **{wins}/5**.", "",
              "## Complete detectors", "",
              "| Detector | Overall F1 | Replay F1 | AUPRC |",
              "|---|---:|---:|---:|"]
    for name in ("mixture", "uniform", "oracle"):
        row = report[name]
        lines.append(f"| {name} | {row['overall']['f1']:.4f} | "
                     f"{row['per_family']['replay']['f1']:.4f} | {row['overall']['auprc']:.4f} |")
    lines += ["", "Oracle uses true family labels and is NOT deployable or a guaranteed ceiling.",
              "These are one-checkpoint development diagnostics; no seed robustness is established.", ""]
    Path(run_dir, "expert_audit.md").write_text("\n".join(lines))


def audit(run_dir):
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir/"best.pt", map_location="cpu", weights_only=True)
    cfg = TrainingConfig(**checkpoint["config"])
    cfg.device = "cpu"
    torch.set_num_threads(cfg.threads)
    if cfg.model == "single":
        raise ValueError("Use a multi-expert checkpoint for expert audits")
    splits = json.loads((run_dir/"splits.json").read_text())
    data = Path(cfg.data_dir)
    with (data/"snapshots.pkl").open("rb") as f: snapshots = pickle.load(f)
    with (data/"corrupted.pkl").open("rb") as f: corrupted = pickle.load(f)
    with (run_dir/"normalizer.pkl").open("rb") as f: normalizer = pickle.load(f)
    sets = {}
    for name in ("calibration", "validation"):
        selected = [i for i, s in enumerate(snapshots) if s.scenario_id in splits[name]]
        sets[name] = TemporalWDNDataset([snapshots[i] for i in selected],
            [corrupted[i] for i in selected], cfg.endpoint_window, normalizer)
    model = build_model(cfg, sets["calibration"][0])
    model.load_state_dict(checkpoint["state_dict"])
    arrays = {name: collect_experts(model, ds, cfg) for name, ds in sets.items()}
    cal, val = arrays["calibration"], arrays["validation"]
    report = {"checkpoint_epoch": checkpoint["epoch"], "test_evaluated": False,
              "threshold_protocol": "expert-specific global-F1 calibration, fixed across validation families",
              "expert_matrix": {}, "routing_by_family": {}}
    for expert, name in enumerate(FAMILY_NAMES):
        threshold = best_f1_threshold(cal["experts"][:, expert], cal["labels"])
        row = {"threshold": threshold, "per_family": {}}
        for family, label in enumerate(FAMILY_NAMES):
            selected = val["families"] == family
            scores, labels = val["experts"][selected, expert], val["labels"][selected]
            metrics = _binary_counts(scores, labels, threshold)
            metrics["auprc"] = float(average_precision_score(labels, scores)) if labels.any() else None
            row["per_family"][label] = metrics
        report["expert_matrix"][name] = row
    for family, label in enumerate(FAMILY_NAMES):
        selected = val["families"] == family
        report["routing_by_family"][label] = val["routing"][selected].mean(0).tolist()
    from wdn.train_operational_moe import select_threshold
    for name in ("mixture", "uniform", "oracle"):
        threshold = select_threshold(cal[name], cal["labels"], cal["families"])
        report[name] = summarise(val[name], val["labels"], val["families"], threshold)
    report["oracle_note"] = "Uses true family labels: diagnostic only, not deployable or a guaranteed ceiling"
    path = run_dir/"expert_audit.json"
    path.write_text(json.dumps(report, indent=2))
    write_audit_markdown(report, run_dir)
    for name, data in arrays.items():
        np.savez_compressed(run_dir/f"expert_scores_{name}.npz", **data)
    print("Expert / random / replay / drift / noise / targeted AUPRC")
    for name, row in report["expert_matrix"].items():
        print(name, *[round(row["per_family"][family]["auprc"] or 0., 3) for family in FAMILY_NAMES[1:]])
    for name in ("mixture", "uniform", "oracle"):
        print(name, "F1", round(report[name]["overall"]["f1"],4),
              "replay", round(report[name]["per_family"]["replay"]["f1"],4))
    print(path)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    audit(parser.parse_args().run_dir)

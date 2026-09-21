"""Test whether the Modena MoE learned stable, useful specialists.

The performance gate alone is insufficient: a six-expert model can improve by
capacity while ignoring its intended family decomposition.  This script
compares the soft mixture, router top-1 expert and true-family (oracle) expert,
then checks whether each family's owning expert ranks first on that family's
sensor labels.  Thresholds are selected on validation; AUPRC is threshold-free.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_workshop_routing import load_split, to_device  # noqa: E402
from wdn.models.temporal_moe import TemporalMixtureOfExpertsGNN  # noqa: E402


def make_model(args: dict, sample: dict, device: torch.device):
    return TemporalMixtureOfExpertsGNN(
        node_in_dim=sample["x_seq"][0].shape[1],
        edge_in_dim=sample["edge_attr"].shape[1],
        hidden_dim=int(args["hidden_dim"]),
        num_experts=int(args["num_experts"]),
        router_hidden_dim=int(args["router_hidden_dim"]),
        num_layers=2,
        num_temporal_layers=int(args.get("num_temporal_layers", 1)),
        window_size=int(args["window_size"]),
        dropout=0.1,
        gnn_type=args["gnn_type"],
        heads=4,
        use_pattern_features=not bool(args.get("no_pattern_features", False)),
        use_topology=not bool(args.get("no_topology", False)),
    ).to(device)


def forward(model, batch: dict):
    return model(
        x_seq=batch["x_seq"], edge_index=batch["edge_index"],
        edge_attr=batch["edge_attr"], is_original_edge=batch["is_original_edge"],
        batch_size=batch["batch_size"], num_nodes_per_graph=batch["num_nodes"],
        pressure_obs=batch["pressure_obs"], flow_obs=batch["flow_obs"],
        pressure_mask=batch["pressure_mask"], flow_mask=batch["flow_mask"],
    )


@torch.no_grad()
def collect(model, loader, device):
    soft, top1, oracle, labels, families, expert_scores = [], [], [], [], [], []
    router_true, router_pred = [], []
    for raw in loader:
        batch = to_device(raw, device)
        out = forward(model, batch)
        score_soft = torch.sigmoid(out["pressure_anomaly_logits"])
        score_experts = torch.sigmoid(out["expert_pressure_anomaly_logits"])
        n_nodes = int(batch["num_nodes"])
        mask = batch["pressure_mask"] > 0
        top = out["router_probs"].argmax(dim=-1)
        owner = batch["attack_report"].long()

        def choose(index: torch.Tensor):
            expanded = index.repeat_interleave(n_nodes).unsqueeze(-1)
            return score_experts.gather(-1, expanded).squeeze(-1)

        soft.append(score_soft[mask].cpu().numpy())
        top1.append(choose(top)[mask].cpu().numpy())
        oracle.append(choose(owner)[mask].cpu().numpy())
        expert_scores.append(score_experts[mask].cpu().numpy())
        labels.append(batch["pressure_anomaly"][mask].cpu().numpy().astype(int))
        families.append(
            owner.repeat_interleave(n_nodes)[mask].cpu().numpy().astype(int)
        )
        router_true.append(owner.cpu().numpy())
        router_pred.append(top.cpu().numpy())
    concatenate = lambda rows: np.concatenate(rows, axis=0)
    return {
        "soft": concatenate(soft),
        "top1": concatenate(top1),
        "oracle": concatenate(oracle),
        "expert_scores": concatenate(expert_scores),
        "labels": concatenate(labels),
        "families": concatenate(families),
        "router_true": concatenate(router_true),
        "router_pred": concatenate(router_pred),
    }


def threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    best_threshold, best_f1 = 0.5, -1.0
    for candidate in np.linspace(0.02, 0.98, 97):
        value = f1_score(labels, scores > candidate, zero_division=0)
        if value > best_f1:
            best_threshold, best_f1 = float(candidate), float(value)
    return best_threshold


def metrics(scores: np.ndarray, labels: np.ndarray, selected_threshold: float) -> dict:
    return {
        "threshold": selected_threshold,
        "f1": float(f1_score(labels, scores > selected_threshold, zero_division=0)),
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
    }


def analyse_run(run_dir: Path, device: torch.device) -> dict:
    args = json.loads((run_dir / "args.json").read_text())
    (train_loader, val_loader, test_loader, _), _, class_names = load_split(
        ROOT / args["data_dir"],
        int(args["window_size"]),
        int(args["batch_size"]),
        args["norm_mode"],
    )
    del train_loader
    model = make_model(args, test_loader.dataset[0], device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()
    validation = collect(model, val_loader, device)
    test = collect(model, test_loader, device)

    variants = {}
    for name in ("soft", "top1", "oracle"):
        selected = threshold(validation[name], validation["labels"])
        variants[name] = metrics(test[name], test["labels"], selected)

    family_matrix = {}
    owner_wins = 0
    owner_ranks = []
    for family_id, family_name in enumerate(class_names):
        if family_id == 0:
            continue
        selected = test["families"] == family_id
        labels = test["labels"][selected]
        if labels.sum() == 0 or labels.sum() == labels.size:
            continue
        values = [
            float(average_precision_score(labels, test["expert_scores"][selected, expert]))
            for expert in range(len(class_names))
        ]
        ranking = np.argsort(values)[::-1].tolist()
        owner_rank = ranking.index(family_id) + 1
        owner_ranks.append(owner_rank)
        owner_wins += int(owner_rank == 1)
        family_matrix[family_name] = {
            "auprc_by_expert": {
                class_names[expert]: value for expert, value in enumerate(values)
            },
            "owner_rank": owner_rank,
            "best_expert": class_names[ranking[0]],
        }

    return {
        "run_id": run_dir.name,
        "data_seed": int(re.search(r"modena_episode_seed(\d+)$", args["data_dir"]).group(1)),
        "model_seed": int(args["seed"]),
        "variants": variants,
        "router_accuracy": float(
            (test["router_true"] == test["router_pred"]).mean()
        ),
        "mean_sensor_expert_score_sd": float(test["expert_scores"].std(axis=1).mean()),
        "families_with_owner_best": owner_wins,
        "families_evaluated": len(owner_ranks),
        "mean_owner_rank": float(np.mean(owner_ranks)),
        "per_family": family_matrix,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-roots", type=Path, nargs="+",
        default=[V2 / "outputs" / "runs" / "modena_screening"],
    )
    parser.add_argument(
        "--data-seeds", type=int, nargs="+", default=None,
        help="Optional generated-data seed filter.",
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()
    run_dirs = []
    for run_root in args.run_roots:
        for directory in sorted(run_root.iterdir()):
            args_path = directory / "args.json"
            if not args_path.exists() or not (directory / "best_model.pt").exists():
                continue
            run_args = json.loads(args_path.read_text())
            match = re.search(r"modena_episode_seed(\d+)$", run_args["data_dir"])
            data_seed = int(match.group(1)) if match else None
            if (
                int(run_args.get("num_experts", 1)) > 1
                and (args.data_seeds is None or data_seed in args.data_seeds)
            ):
                run_dirs.append(directory)
    if len(run_dirs) < 3 or len(run_dirs) % 3:
        raise SystemExit(
            f"expected complete three-model-seed MoE groups, found {len(run_dirs)} runs"
        )

    device = torch.device(args.device)
    rows = []
    for run_dir in run_dirs:
        row = analyse_run(run_dir, device)
        rows.append(row)
        print(
            f"{run_dir.name}: soft AUPRC={row['variants']['soft']['auprc']:.3f}, "
            f"oracle={row['variants']['oracle']['auprc']:.3f}, "
            f"owner best={row['families_with_owner_best']}/{row['families_evaluated']}"
        )

    summary = {
        "n_model_seeds": len(rows),
        "n_data_seeds": len({row["data_seed"] for row in rows}),
        "soft_auprc_mean": statistics.mean(row["variants"]["soft"]["auprc"] for row in rows),
        "oracle_minus_soft_auprc_mean": statistics.mean(
            row["variants"]["oracle"]["auprc"] - row["variants"]["soft"]["auprc"]
            for row in rows
        ),
        "router_accuracy_mean": statistics.mean(row["router_accuracy"] for row in rows),
        "owner_best_fraction": sum(row["families_with_owner_best"] for row in rows)
        / sum(row["families_evaluated"] for row in rows),
        "mean_owner_rank": statistics.mean(row["mean_owner_rank"] for row in rows),
        "family_owner_best": {
            family: {
                "wins": sum(
                    family in row["per_family"]
                    and row["per_family"][family]["owner_rank"] == 1
                    for row in rows
                ),
                "runs": sum(family in row["per_family"] for row in rows),
            }
            for family in sorted({
                family for row in rows for family in row["per_family"]
            })
        },
    }
    output = V2 / "outputs" / "runs" / "modena_moe_specialisation.json"
    output.write_text(json.dumps({"runs": rows, "summary": summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

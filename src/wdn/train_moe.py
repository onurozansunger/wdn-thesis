"""Training script for the Mixture-of-Experts GNN.

Joint training of the attack router plus K attack-specialized experts.
Evaluation reports overall reconstruction / anomaly metrics AND per-attack
F1 so we can see whether specialization helps each attack type (especially
replay) over the single MultiTaskGNN baseline.

Usage:
    python -m wdn.train_moe --data_dir data/moe_net1 --epochs 80
    python -m wdn.train_moe --data_dir data/moe_modena --epochs 60
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from wdn.corruption import ID_TO_ATTACK_TYPE, NUM_ATTACK_CLASSES
from wdn.dataset import train_val_test_split, create_dataloaders
from wdn.models.moe import MixtureOfExpertsGNN, moe_loss
from wdn.models.multitask import multitask_loss
from wdn.models.recon import physics_loss
from wdn.metrics import compute_recon_metrics, compute_anomaly_metrics


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, device, incidence, graph_num_edges,
    lambda_physics=0.1, lambda_anomaly=1.0,
    lambda_router=0.5, lambda_balance=0.01,
):
    model.train()
    totals = defaultdict(float)
    n = 0

    for batch in loader:
        batch = batch.to(device)

        out = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            is_original_edge=batch.is_original_edge,
            batch=batch.batch if hasattr(batch, 'batch') else None,
            pressure_obs=batch.pressure_obs,
            flow_obs=batch.flow_obs,
            pressure_mask=batch.pressure_mask,
            flow_mask=batch.flow_mask,
        )

        losses = moe_loss(
            out, batch, multitask_loss,
            lambda_router=lambda_router,
            lambda_balance=lambda_balance,
            lambda_anomaly=lambda_anomaly,
        )

        phys = torch.tensor(0.0, device=device)
        if lambda_physics > 0:
            bs = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
            phys = physics_loss(out["flow_pred"], incidence, bs, graph_num_edges)

        loss = losses["total_loss"] + lambda_physics * phys

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in ("recon_loss", "anomaly_loss", "router_ce", "balance"):
            totals[k] += losses[k].item()
        totals["total"] += loss.item()
        n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model, loader, device, normalizer=None,
    lambda_anomaly=1.0, lambda_router=0.5, lambda_balance=0.01,
):
    model.eval()
    totals = defaultdict(float)
    n = 0

    all_p_pred, all_p_true, all_p_mask = [], [], []
    all_q_pred, all_q_true, all_q_mask = [], [], []
    all_p_anom_logits, all_p_anom_true, all_p_anom_mask = [], [], []
    all_q_anom_logits, all_q_anom_true, all_q_anom_mask = [], [], []
    all_router_logits, all_router_targets = [], []

    # Per-attack aggregation: we replicate the graph-level attack label
    # onto each of its nodes / original edges so we can slice metrics by
    # attack class after the fact.
    all_p_attack_class, all_q_attack_class = [], []

    for batch in loader:
        batch = batch.to(device)

        out = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            is_original_edge=batch.is_original_edge,
            batch=batch.batch if hasattr(batch, 'batch') else None,
            pressure_obs=batch.pressure_obs,
            flow_obs=batch.flow_obs,
            pressure_mask=batch.pressure_mask,
            flow_mask=batch.flow_mask,
        )

        losses = moe_loss(
            out, batch, multitask_loss,
            lambda_router=lambda_router,
            lambda_balance=lambda_balance,
            lambda_anomaly=lambda_anomaly,
        )

        for k in ("recon_loss", "anomaly_loss", "router_ce", "balance"):
            totals[k] += losses[k].item()
        n += 1

        # --- Predictions ---
        p_pred = out["pressure_pred"].cpu()
        q_pred = out["flow_pred"].cpu()
        p_true = batch.y_pressure.cpu()
        q_true = batch.y_flow.cpu()

        if normalizer is not None:
            p_pred = normalizer.denormalize_pressure(p_pred)
            q_pred = normalizer.denormalize_flow(q_pred)
            p_true = normalizer.denormalize_pressure(p_true)
            q_true = normalizer.denormalize_flow(q_true)

        all_p_pred.append(p_pred)
        all_p_true.append(p_true)
        all_p_mask.append(batch.pressure_mask.cpu())
        all_q_pred.append(q_pred)
        all_q_true.append(q_true)
        all_q_mask.append(batch.flow_mask.cpu())

        if "pressure_anomaly_logits" in out:
            all_p_anom_logits.append(out["pressure_anomaly_logits"].cpu())
            all_p_anom_true.append(batch.pressure_anomaly.cpu())
            all_p_anom_mask.append(batch.pressure_mask.cpu())
        if "flow_anomaly_logits" in out:
            all_q_anom_logits.append(out["flow_anomaly_logits"].cpu())
            all_q_anom_true.append(batch.flow_anomaly.cpu())
            all_q_anom_mask.append(batch.flow_mask.cpu())

        all_router_logits.append(out["router_logits"].cpu())
        all_router_targets.append(batch.attack_type.view(-1).long().cpu())

        # Broadcast graph-level attack label to nodes / edges.
        node_batch = batch.batch if hasattr(batch, 'batch') else torch.zeros(
            batch.x.shape[0], dtype=torch.long, device=device,
        )
        attack_graph = batch.attack_type.view(-1).long()                    # (B,)
        node_attack = attack_graph[node_batch].cpu()                        # (N,)
        all_p_attack_class.append(node_attack)
        orig_src = batch.edge_index[0][batch.is_original_edge]
        edge_graph = node_batch[orig_src]
        edge_attack = attack_graph[edge_graph].cpu()                        # (NE,)
        all_q_attack_class.append(edge_attack)

    # ------- Global reconstruction metrics -------
    p_pred = torch.cat(all_p_pred)
    p_true = torch.cat(all_p_true)
    p_mask = torch.cat(all_p_mask)
    q_pred = torch.cat(all_q_pred)
    q_true = torch.cat(all_q_true)
    q_mask = torch.cat(all_q_mask)

    result = {k: v / max(n, 1) for k, v in totals.items()}
    result.update({
        "pressure_all": compute_recon_metrics(p_pred, p_true),
        "pressure_unobs": compute_recon_metrics(p_pred, p_true, p_mask, only_unobserved=True),
        "flow_all": compute_recon_metrics(q_pred, q_true),
        "flow_unobs": compute_recon_metrics(q_pred, q_true, q_mask, only_unobserved=True),
    })

    # ------- Global anomaly metrics -------
    p_logits_all = torch.cat(all_p_anom_logits) if all_p_anom_logits else None
    p_labels_all = torch.cat(all_p_anom_true) if all_p_anom_true else None
    p_amask_all = torch.cat(all_p_anom_mask) if all_p_anom_mask else None

    q_logits_all = torch.cat(all_q_anom_logits) if all_q_anom_logits else None
    q_labels_all = torch.cat(all_q_anom_true) if all_q_anom_true else None
    q_amask_all = torch.cat(all_q_anom_mask) if all_q_anom_mask else None

    if p_logits_all is not None:
        p_scores = torch.sigmoid(p_logits_all)
        result["pressure_anomaly"] = compute_anomaly_metrics(
            (p_scores > 0.5).float(), p_labels_all, p_scores, p_amask_all,
        )
    if q_logits_all is not None:
        q_scores = torch.sigmoid(q_logits_all)
        result["flow_anomaly"] = compute_anomaly_metrics(
            (q_scores > 0.5).float(), q_labels_all, q_scores, q_amask_all,
        )

    # ------- Router accuracy -------
    router_logits = torch.cat(all_router_logits)
    router_targets = torch.cat(all_router_targets)
    router_pred = router_logits.argmax(dim=-1)
    router_acc = (router_pred == router_targets).float().mean().item()
    result["router_acc"] = router_acc

    # Router confusion matrix (sparse dict: {(true, pred): count}).
    conf = defaultdict(int)
    for t, p in zip(router_targets.tolist(), router_pred.tolist()):
        conf[(t, p)] += 1
    result["router_confusion"] = dict(conf)

    # ------- Per-attack anomaly F1 -------
    per_attack = {}
    if p_logits_all is not None:
        p_attack_class = torch.cat(all_p_attack_class)                      # (N,)
        p_scores = torch.sigmoid(p_logits_all)
        p_pred_labels = (p_scores > 0.5).float()
        for cls_id in range(NUM_ATTACK_CLASSES):
            mask_class = (p_attack_class == cls_id) & (p_amask_all > 0)
            if mask_class.sum() == 0:
                continue
            m = compute_anomaly_metrics(
                p_pred_labels[mask_class],
                p_labels_all[mask_class],
                p_scores[mask_class],
            )
            per_attack[ID_TO_ATTACK_TYPE[cls_id]] = {
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "auroc": m.auroc,
                "n": int(mask_class.sum().item()),
            }
    result["per_attack_pressure"] = per_attack

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/moe_net1")
    parser.add_argument("--gnn_type", type=str, default="GraphSAGE")
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--num_experts", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_anomaly", type=float, default=1.0)
    parser.add_argument("--lambda_physics", type=float, default=0.1)
    parser.add_argument("--lambda_router", type=float, default=0.5)
    parser.add_argument("--lambda_balance", type=float, default=0.01)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    with open(data_dir / "graph.pkl", "rb") as f:
        graph = pickle.load(f)
    with open(data_dir / "snapshots.pkl", "rb") as f:
        snapshots = pickle.load(f)
    with open(data_dir / "corrupted.pkl", "rb") as f:
        corrupted = pickle.load(f)

    print(f"Loaded {len(snapshots)} snapshots from {data_dir}")
    print(f"Network: {graph.num_nodes} nodes, {graph.num_edges} edges")

    train_s, train_c, val_s, val_c, test_s, test_c = train_val_test_split(
        snapshots, corrupted, 0.70, 0.15, seed=42,
    )
    train_loader, val_loader, test_loader, normalizer = create_dataloaders(
        train_s, train_c, val_s, val_c, test_s, test_c,
        batch_size=args.batch_size, num_workers=0,
    )

    sample = train_loader.dataset[0]
    model = MixtureOfExpertsGNN(
        node_in_dim=sample.x.shape[1],
        edge_in_dim=sample.edge_attr.shape[1],
        hidden_dim=args.hidden_dim,
        num_experts=args.num_experts,
        num_layers=2,
        dropout=0.1,
        gnn_type=args.gnn_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: MixtureOfExpertsGNN ({args.num_experts} experts, "
          f"{args.gnn_type}) -> {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8,
    )
    incidence = torch.tensor(graph.incidence_matrix, dtype=torch.float32).to(device)

    best_val = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  lambda_anomaly={args.lambda_anomaly}  lambda_physics={args.lambda_physics}")
    print(f"  lambda_router={args.lambda_router}  lambda_balance={args.lambda_balance}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(
            model, train_loader, optimizer, device, incidence,
            graph.num_edges, args.lambda_physics, args.lambda_anomaly,
            args.lambda_router, args.lambda_balance,
        )
        val_m = evaluate(
            model, val_loader, device, normalizer,
            args.lambda_anomaly, args.lambda_router, args.lambda_balance,
        )
        scheduler.step(val_m["recon_loss"])

        elapsed = time.time() - t0
        entry = {
            "epoch": epoch,
            "train_recon": train_m["recon_loss"],
            "train_anomaly": train_m["anomaly_loss"],
            "train_router_ce": train_m["router_ce"],
            "val_recon": val_m["recon_loss"],
            "val_p_mae_unobs": val_m["pressure_unobs"].mae,
            "val_router_acc": val_m["router_acc"],
        }
        if "pressure_anomaly" in val_m:
            entry["val_p_anomaly_f1"] = val_m["pressure_anomaly"].f1
            entry["val_p_anomaly_auroc"] = val_m["pressure_anomaly"].auroc
        history.append(entry)

        if epoch % 5 == 0 or epoch == 1:
            anom_str = ""
            if "pressure_anomaly" in val_m:
                anom_str = (f" | P_Anom F1={val_m['pressure_anomaly'].f1:.3f}"
                           f" AUROC={val_m['pressure_anomaly'].auroc:.3f}")
            print(
                f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s) | "
                f"Train: {train_m['total']:.4f} (router_ce={train_m['router_ce']:.3f}) | "
                f"Val Recon: {val_m['recon_loss']:.4f} | "
                f"P_MAE(unobs): {val_m['pressure_unobs'].mae:.3f} | "
                f"Router Acc: {val_m['router_acc']:.3f}"
                f"{anom_str}"
            )

        if val_m["recon_loss"] < best_val:
            best_val = val_m["recon_loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # ---- Test eval ----
    print(f"\n{'=' * 70}\nTEST SET EVALUATION\n{'=' * 70}")
    model.load_state_dict(best_state)
    test_m = evaluate(
        model, test_loader, device, normalizer,
        args.lambda_anomaly, args.lambda_router, args.lambda_balance,
    )

    print("  Reconstruction:")
    print(f"    Pressure (unobs): {test_m['pressure_unobs']}")
    print(f"    Flow (unobs):     {test_m['flow_unobs']}")
    if "pressure_anomaly" in test_m:
        print(f"  Pressure anomaly: {test_m['pressure_anomaly']}")
    if "flow_anomaly" in test_m:
        print(f"  Flow anomaly:     {test_m['flow_anomaly']}")
    print(f"  Router accuracy:  {test_m['router_acc']:.3f}")
    print("  Per-attack pressure F1:")
    for name, m in test_m["per_attack_pressure"].items():
        print(f"    {name:10s} n={m['n']:5d}  P={m['precision']:.3f}  "
              f"R={m['recall']:.3f}  F1={m['f1']:.3f}  AUROC={m['auroc']:.3f}")

    # ---- Save ----
    run_dir = Path("runs/moe") / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    to_dict_recon = lambda r: {"mae": r.mae, "mse": r.mse, "rmse": r.rmse}
    to_dict_anom = lambda m: {"precision": m.precision, "recall": m.recall,
                               "f1": m.f1, "auroc": m.auroc}

    payload = {
        "model": "MixtureOfExpertsGNN",
        "data_dir": str(data_dir),
        "n_params": n_params,
        "num_experts": args.num_experts,
        "hidden_dim": args.hidden_dim,
        "reconstruction": {
            "pressure_all": to_dict_recon(test_m["pressure_all"]),
            "pressure_unobs": to_dict_recon(test_m["pressure_unobs"]),
            "flow_all": to_dict_recon(test_m["flow_all"]),
            "flow_unobs": to_dict_recon(test_m["flow_unobs"]),
        },
        "router_acc": test_m["router_acc"],
        "router_confusion": {f"{k[0]}_{k[1]}": v for k, v in test_m["router_confusion"].items()},
        "per_attack_pressure": test_m["per_attack_pressure"],
    }
    if "pressure_anomaly" in test_m:
        payload["anomaly_detection"] = {"pressure": to_dict_anom(test_m["pressure_anomaly"])}
    if "flow_anomaly" in test_m:
        payload.setdefault("anomaly_detection", {})["flow"] = to_dict_anom(test_m["flow_anomaly"])

    with open(run_dir / "test_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), run_dir / "best_model.pt")
    torch.save(normalizer.state_dict(), run_dir / "normalizer.pt")

    # Also save args for reproducibility
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()

"""Training script for the Temporal Mixture-of-Experts GNN.

Same joint recipe as `train_moe` (router CE + recon + anomaly + balance)
but runs on temporal windows so each expert is a TemporalMultiTaskGNN with
a GRU. This is the variant that should actually solve replay detection:
a spatial-only expert has no way to see that a reading is *stale*, while a
temporal expert can compare against the last few timesteps.

Usage:
    python -m wdn.train_temporal_moe --data_dir data/moe_net1 --epochs 60
    python -m wdn.train_temporal_moe --data_dir data/moe_modena --epochs 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from wdn.corruption import active_attack_classes, compact_id_map
from wdn.temporal_dataset import create_temporal_dataloaders
from wdn.models.temporal_moe import TemporalMixtureOfExpertsGNN, temporal_moe_loss
from wdn.models.recon import physics_loss
from wdn.metrics import compute_recon_metrics, compute_anomaly_metrics


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def _to_device(batch: dict, device: torch.device) -> dict:
    """Move every tensor in a temporal batch dict to device (in place)."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [t.to(device) for t in v]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, device, incidence, graph_num_edges,
    lambda_physics=0.1, lambda_anomaly=1.0,
    lambda_router=0.5, lambda_balance=0.01, replay_weight=1.0,
    lambda_expert=0.5, replay_class_id=None,
):
    model.train()
    totals = defaultdict(float)
    n = 0

    for raw_batch in loader:
        batch = _to_device(raw_batch, device)

        out = model(
            x_seq=batch["x_seq"],
            edge_index=batch["edge_index"],
            edge_attr=batch["edge_attr"],
            is_original_edge=batch["is_original_edge"],
            batch_size=batch["batch_size"],
            num_nodes_per_graph=batch["num_nodes"],
            pressure_obs=batch["pressure_obs"],
            flow_obs=batch["flow_obs"],
            pressure_mask=batch["pressure_mask"],
            flow_mask=batch["flow_mask"],
        )

        losses = temporal_moe_loss(
            out, batch,
            lambda_router=lambda_router,
            lambda_balance=lambda_balance,
            lambda_anomaly=lambda_anomaly,
            replay_weight=replay_weight,
            lambda_expert=lambda_expert,
            replay_class_id=replay_class_id,
        )

        phys = torch.tensor(0.0, device=device)
        if lambda_physics > 0:
            bs = batch["batch_size"]
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
    replay_weight=1.0, lambda_expert=0.5,
    threshold=0.5, return_scores=False,
    class_names=None, replay_class_id=None,
):
    """Evaluate the model.

    ``threshold`` is the decision threshold applied to the sigmoid anomaly
    score. The default 0.5 is only correct if the score distribution happens
    to be centred there; per-node normalisation shifts it, so the caller
    should calibrate on validation (see ``calibrate_threshold``) and pass
    the result. ``return_scores`` additionally returns the pooled pressure
    (scores, labels, observed-mask) so a caller can run that calibration.
    """
    model.eval()
    totals = defaultdict(float)
    n = 0

    all_p_pred, all_p_true, all_p_mask = [], [], []
    all_q_pred, all_q_true, all_q_mask = [], [], []
    all_p_anom_logits, all_p_anom_true, all_p_anom_mask = [], [], []
    all_q_anom_logits, all_q_anom_true, all_q_anom_mask = [], [], []
    all_router_logits, all_router_targets = [], []
    all_p_attack_class, all_q_attack_class = [], []

    for raw_batch in loader:
        batch = _to_device(raw_batch, device)

        out = model(
            x_seq=batch["x_seq"],
            edge_index=batch["edge_index"],
            edge_attr=batch["edge_attr"],
            is_original_edge=batch["is_original_edge"],
            batch_size=batch["batch_size"],
            num_nodes_per_graph=batch["num_nodes"],
            pressure_obs=batch["pressure_obs"],
            flow_obs=batch["flow_obs"],
            pressure_mask=batch["pressure_mask"],
            flow_mask=batch["flow_mask"],
        )

        losses = temporal_moe_loss(
            out, batch,
            lambda_router=lambda_router,
            lambda_balance=lambda_balance,
            lambda_anomaly=lambda_anomaly,
            replay_weight=replay_weight,
            lambda_expert=lambda_expert,
            replay_class_id=replay_class_id,
        )

        for k in ("recon_loss", "anomaly_loss", "router_ce", "balance"):
            totals[k] += losses[k].item()
        n += 1

        # Predictions (denormalized)
        p_pred = out["pressure_pred"].cpu()
        q_pred = out["flow_pred"].cpu()
        p_true = batch["y_pressure"].cpu()
        q_true = batch["y_flow"].cpu()

        if normalizer is not None:
            p_pred = normalizer.denormalize_pressure(p_pred)
            q_pred = normalizer.denormalize_flow(q_pred)
            p_true = normalizer.denormalize_pressure(p_true)
            q_true = normalizer.denormalize_flow(q_true)

        all_p_pred.append(p_pred)
        all_p_true.append(p_true)
        all_p_mask.append(batch["pressure_mask"].cpu())
        all_q_pred.append(q_pred)
        all_q_true.append(q_true)
        all_q_mask.append(batch["flow_mask"].cpu())

        if "pressure_anomaly_logits" in out:
            all_p_anom_logits.append(out["pressure_anomaly_logits"].cpu())
            all_p_anom_true.append(batch["pressure_anomaly"].cpu())
            all_p_anom_mask.append(batch["pressure_mask"].cpu())
        if "flow_anomaly_logits" in out:
            all_q_anom_logits.append(out["flow_anomaly_logits"].cpu())
            all_q_anom_true.append(batch["flow_anomaly"].cpu())
            all_q_anom_mask.append(batch["flow_mask"].cpu())

        all_router_logits.append(out["router_logits"].cpu())
        all_router_targets.append(batch["attack_type"].cpu())

        # Broadcast graph-level attack label to nodes / original edges. All
        # graphs in the batch share the same topology, so we just repeat.
        B = batch["batch_size"]
        N = batch["num_nodes"]
        # Reporting label keeps full class resolution even when routing
        # collapsed the classes (single-expert ablation).
        attack_graph = batch.get("attack_report", batch["attack_type"]).cpu()
        NE_total = q_pred.shape[0]
        NE = NE_total // max(B, 1)
        all_p_attack_class.append(attack_graph.repeat_interleave(N))       # (B*N,)
        all_q_attack_class.append(attack_graph.repeat_interleave(NE))      # (B*NE,)

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
            (p_scores > threshold).float(), p_labels_all, p_scores, p_amask_all,
        )
    if q_logits_all is not None:
        q_scores = torch.sigmoid(q_logits_all)
        result["flow_anomaly"] = compute_anomaly_metrics(
            (q_scores > threshold).float(), q_labels_all, q_scores, q_amask_all,
        )
    result["threshold"] = float(threshold)

    # ------- Router accuracy + confusion -------
    router_logits = torch.cat(all_router_logits)
    router_targets = torch.cat(all_router_targets)
    router_pred = router_logits.argmax(dim=-1)
    result["router_acc"] = (router_pred == router_targets).float().mean().item()

    conf = defaultdict(int)
    for t, p in zip(router_targets.tolist(), router_pred.tolist()):
        conf[(t, p)] += 1
    result["router_confusion"] = dict(conf)

    # ------- Per-attack pressure anomaly F1 -------
    per_attack = {}
    if p_logits_all is not None:
        p_attack_class = torch.cat(all_p_attack_class)                     # (N_total,)
        p_scores = torch.sigmoid(p_logits_all)
        p_pred_labels = (p_scores > threshold).float()
        names = class_names or ["clean", "random", "replay", "stealthy",
                                "noise", "targeted"]
        for cls_id, cls_name in enumerate(names):
            mask_class = (p_attack_class == cls_id) & (p_amask_all > 0)
            if mask_class.sum() == 0:
                continue
            m = compute_anomaly_metrics(
                p_pred_labels[mask_class],
                p_labels_all[mask_class],
                p_scores[mask_class],
            )
            # False-positive rate on this slice. It is the only meaningful
            # number for the clean class, which has no positives at all
            # (precision/recall/F1 are degenerate there).
            pred_c = p_pred_labels[mask_class] > 0.5
            pos_c = p_labels_all[mask_class] > 0.5
            neg = (~pos_c).sum()
            per_attack[cls_name] = {
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "auroc": m.auroc,
                "auprc": m.auprc,
                "fpr": float((pred_c & ~pos_c).sum() / neg) if neg > 0 else 0.0,
                "n": int(mask_class.sum().item()),
            }
    result["per_attack_pressure"] = per_attack

    if return_scores:
        return result, (p_scores, p_labels_all, p_amask_all)
    return result


def calibrate_threshold(scores, labels, obs_mask, grid=None):
    """Pick the decision threshold that maximises F1 on the given split.

    Call this with *validation* scores and apply the result to test — the
    hard-coded 0.5 is only optimal when the score distribution happens to be
    centred there, which per-node normalisation breaks.

    Returns (best_threshold, best_f1).
    """
    if scores is None or labels is None:
        return 0.5, 0.0
    if obs_mask is not None:
        sel = obs_mask > 0
        scores, labels = scores[sel], labels[sel]
    if scores.numel() == 0:
        return 0.5, 0.0
    if grid is None:
        grid = torch.linspace(0.02, 0.98, 97)

    pos = labels > 0.5
    n_pos = pos.sum()
    best_t, best_f1 = 0.5, -1.0
    for t in grid.tolist():
        pred = scores > t
        tp = (pred & pos).sum()
        fp = (pred & ~pos).sum()
        denom = 2 * tp + fp + (n_pos - tp)
        f1 = (2 * tp / denom).item() if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/moe_net1")
    parser.add_argument("--gnn_type", type=str, default="GraphSAGE")
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--router_hidden_dim", type=int, default=32,
                        help="Router width — keep small relative to "
                             "hidden_dim (small classifier, bigger experts).")
    parser.add_argument("--num_experts", type=int, default=None,
                        help="Number of experts. Default: one per attack "
                             "class present in the data. Set to 1 to ablate "
                             "the mixture away into a single shared model.")
    parser.add_argument("--window_size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibrate_threshold", action="store_true",
                        default=True,
                        help="Pick the anomaly decision threshold on the "
                             "validation split instead of assuming 0.5.")
    parser.add_argument("--no_calibrate_threshold", dest="calibrate_threshold",
                        action="store_false")
    parser.add_argument("--lambda_expert", type=float, default=0.5,
                        help="Weight on direct per-expert supervision. Higher "
                             "values train each expert to be a competent "
                             "standalone detector, which is what hard/cascade "
                             "routing needs (the mixture only needs blend "
                             "components).")
    parser.add_argument("--norm_mode", type=str, default="global",
                        choices=["global", "per_node"],
                        help="Z-score scope: one scalar for the whole network "
                             "(global) or one mean/std per sensor (per_node). "
                             "per_node restores within-sensor variation that "
                             "global scaling squashes.")
    parser.add_argument("--replay_weight", type=float, default=1.0,
                        help="Per-node loss multiplier for replay windows.")
    parser.add_argument("--num_temporal_layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--output_root", type=str, default="runs/temporal_moe",
        help="Root directory for timestamped run artifacts.",
    )
    parser.add_argument(
        "--run_tag", type=str, default="",
        help="Short experiment label appended to the timestamped run name.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_anomaly", type=float, default=1.0)
    parser.add_argument("--lambda_physics", type=float, default=0.1)
    parser.add_argument("--lambda_router", type=float, default=0.5)
    parser.add_argument("--lambda_balance", type=float, default=0.01)
    parser.add_argument("--no_topology", action="store_true",
                        help="Disable message passing (empty edge set to the "
                             "spatial backbone): the topology-free baseline.")
    parser.add_argument("--no_pattern_features", action="store_true",
                        help="Disable pattern-detection features (autocorr, adj_diff_std, noise_ratio).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)

    device = get_device()
    print(f"Device: {device}, seed: {args.seed}")

    data_dir = Path(args.data_dir)
    with open(data_dir / "graph.pkl", "rb") as f:
        graph = pickle.load(f)
    with open(data_dir / "snapshots.pkl", "rb") as f:
        snapshots = pickle.load(f)
    with open(data_dir / "corrupted.pkl", "rb") as f:
        corrupted = pickle.load(f)

    n_attacked = sum(
        1 for c in corrupted
        if c.pressure_anomaly.sum() > 0 or c.flow_anomaly.sum() > 0
    )
    print(f"Loaded {len(snapshots)} snapshots ({n_attacked} with attacks)")
    print(f"Network: {graph.num_nodes} nodes, {graph.num_edges} edges")

    # Size the router / expert bank to the attack classes the data actually
    # contains. A dataset generated with a restricted attack pool leaves
    # gaps in the canonical ids, so they are compacted to 0..K-1 first.
    class_names = active_attack_classes(corrupted)
    attack_id_map = compact_id_map(class_names)
    replay_class_id = (class_names.index("replay")
                       if "replay" in class_names else None)
    print(f"Attack classes ({len(class_names)}): {', '.join(class_names)}")
    report_id_map = None

    if args.num_experts is None:
        args.num_experts = len(class_names)
    elif args.num_experts == 1:
        # Single-expert ablation: no mixture to route to, so every window
        # is folded onto the one expert and the router losses switch off.
        # This is the "before MoE" rung of the development trajectory.
        report_id_map = dict(attack_id_map)
        attack_id_map = {cid: 0 for cid in attack_id_map}
        replay_class_id = None
        args.lambda_router = 0.0
        args.lambda_balance = 0.0
        print("  single-expert ablation: router losses disabled")
    elif args.num_experts != len(class_names):
        raise SystemExit(
            f"--num_experts {args.num_experts} does not match the "
            f"{len(class_names)} classes in {data_dir}; expert i is defined "
            f"as the specialist for class i. Use 1 to ablate the mixture.")

    # Scenario-based split so windows never straddle train/val/test.
    scenarios = sorted({s.scenario_id for s in snapshots})
    timesteps_per_scenario = len(snapshots) // max(len(scenarios), 1)
    print(f"Scenarios: {len(scenarios)}, Timesteps per scenario: {timesteps_per_scenario}")
    print(f"Window size: {args.window_size}")

    if timesteps_per_scenario < args.window_size:
        print(f"ERROR: window_size ({args.window_size}) > timesteps per scenario "
              f"({timesteps_per_scenario}).")
        return

    rng = np.random.default_rng(42)
    shuffled = scenarios.copy()
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * 0.70)
    n_val = int(len(shuffled) * 0.15)
    train_scen = set(shuffled[:n_train])
    val_scen = set(shuffled[n_train:n_train + n_val])
    test_scen = set(shuffled[n_train + n_val:])

    def _filter(snaps, corrs, scen_set):
        s_out, c_out = [], []
        for s, c in zip(snaps, corrs):
            if s.scenario_id in scen_set:
                s_out.append(s)
                c_out.append(c)
        return s_out, c_out

    train_s, train_c = _filter(snapshots, corrupted, train_scen)
    val_s, val_c = _filter(snapshots, corrupted, val_scen)
    test_s, test_c = _filter(snapshots, corrupted, test_scen)
    print(f"Split: train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")

    train_loader, val_loader, test_loader, normalizer = create_temporal_dataloaders(
        train_s, train_c, val_s, val_c, test_s, test_c,
        window_size=args.window_size,
        batch_size=args.batch_size, num_workers=0,
        norm_mode=args.norm_mode, attack_id_map=attack_id_map,
        report_id_map=report_id_map,
    )

    sample = train_loader.dataset[0]
    model = TemporalMixtureOfExpertsGNN(
        node_in_dim=sample["x_seq"][0].shape[1],
        edge_in_dim=sample["edge_attr"].shape[1],
        hidden_dim=args.hidden_dim,
        num_experts=args.num_experts,
        router_hidden_dim=args.router_hidden_dim,
        num_layers=2,
        num_temporal_layers=args.num_temporal_layers,
        window_size=args.window_size,
        dropout=0.1,
        gnn_type=args.gnn_type,
        heads=4,
        use_pattern_features=not args.no_pattern_features,
        use_topology=not args.no_topology,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: TemporalMixtureOfExpertsGNN ({args.num_experts} experts, "
          f"{args.gnn_type}+GRU) -> {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8,
    )
    incidence = torch.tensor(graph.incidence_matrix, dtype=torch.float32).to(device)

    # Higher composite anomaly score is better (see selection logic below).
    best_val = float("-inf")
    best_state = None
    best_epoch = None
    patience_counter = 0
    history = []
    training_started = time.time()

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  lambda_anomaly={args.lambda_anomaly}  lambda_physics={args.lambda_physics}")
    print(f"  lambda_router={args.lambda_router}  lambda_balance={args.lambda_balance}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(
            model, train_loader, optimizer, device, incidence,
            graph.num_edges, args.lambda_physics, args.lambda_anomaly,
            args.lambda_router, args.lambda_balance, args.replay_weight,
            args.lambda_expert, replay_class_id,
        )
        val_m = evaluate(
            model, val_loader, device, normalizer,
            args.lambda_anomaly, args.lambda_router, args.lambda_balance,
            args.replay_weight, args.lambda_expert,
            class_names=class_names, replay_class_id=replay_class_id,
        )
        scheduler.step(val_m["recon_loss"])

        elapsed = time.time() - t0
        entry = {
            "epoch": epoch,
            "epoch_seconds": elapsed,
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
            entry["val_p_anomaly_auprc"] = val_m["pressure_anomaly"].auprc
        history.append(entry)

        if epoch % 5 == 0 or epoch == 1:
            anom_str = ""
            if "pressure_anomaly" in val_m:
                anom_str = (f" | P_Anom F1={val_m['pressure_anomaly'].f1:.3f}"
                            f" AUROC={val_m['pressure_anomaly'].auroc:.3f}")
            # Quick glimpse of the replay slice — the whole point of temporal.
            replay_str = ""
            if "replay" in val_m.get("per_attack_pressure", {}):
                replay_f1 = val_m["per_attack_pressure"]["replay"]["f1"]
                replay_str = f" | Replay F1={replay_f1:.3f}"
            print(
                f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s) | "
                f"Train: {train_m['total']:.4f} (router_ce={train_m['router_ce']:.3f}) | "
                f"Val Recon: {val_m['recon_loss']:.4f} | "
                f"P_MAE(unobs): {val_m['pressure_unobs'].mae:.3f} | "
                f"Router Acc: {val_m['router_acc']:.3f}"
                f"{anom_str}{replay_str}"
            )

        # Select checkpoints by the pre-registered primary metric on the
        # validation split. AUPRC is threshold-free and appropriate for the
        # imbalanced sensor-level task. The previous F1 + replay-F1 composite
        # privileged one family and made the selected model depend on an
        # arbitrary decision threshold.
        anomaly_metrics = val_m.get("pressure_anomaly")
        score = anomaly_metrics.auprc if anomaly_metrics is not None else 0.0
        if score > best_val:
            best_val = score
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # ---- Test eval ----
    print(f"\n{'=' * 70}\nTEST SET EVALUATION (Temporal MoE)\n{'=' * 70}")
    model.load_state_dict(best_state)

    # Calibrate the decision threshold on VALIDATION, then apply it to test.
    # The default 0.5 is mis-calibrated whenever the score distribution is
    # not centred there (per-node normalisation shifts it), which costs F1
    # even when ranking quality (AUROC) is better.
    threshold = 0.5
    if args.calibrate_threshold:
        _, (v_scores, v_labels, v_mask) = evaluate(
            model, val_loader, device, normalizer,
            args.lambda_anomaly, args.lambda_router, args.lambda_balance,
            args.replay_weight, args.lambda_expert, return_scores=True,
            class_names=class_names, replay_class_id=replay_class_id,
        )
        threshold, val_f1 = calibrate_threshold(v_scores, v_labels, v_mask)
        print(f"  Calibrated threshold on val: {threshold:.3f} "
              f"(val F1 {val_f1:.4f}, default 0.5)")

    test_m = evaluate(
        model, test_loader, device, normalizer,
        args.lambda_anomaly, args.lambda_router, args.lambda_balance,
        args.replay_weight, args.lambda_expert, threshold=threshold,
        class_names=class_names, replay_class_id=replay_class_id,
    )
    # Also record the uncalibrated numbers so the gain is auditable.
    if args.calibrate_threshold:
        test_default = evaluate(
            model, test_loader, device, normalizer,
            args.lambda_anomaly, args.lambda_router, args.lambda_balance,
            args.replay_weight, args.lambda_expert, threshold=0.5,
            class_names=class_names, replay_class_id=replay_class_id,
        )
        da = test_default.get("pressure_anomaly")
        test_m["uncalibrated"] = {
            "threshold": 0.5,
            "pressure_f1": da.f1 if da else None,
            "pressure_precision": da.precision if da else None,
            "pressure_recall": da.recall if da else None,
            "per_attack_replay_f1": test_default.get("per_attack_pressure", {})
                .get("replay", {}).get("f1"),
        }

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
              f"R={m['recall']:.3f}  F1={m['f1']:.3f}  AUROC={m['auroc']:.3f}  "
              f"FPR={m['fpr']:.3f}")

    # ---- Save ----
    # Second-resolution names collide when runs are launched in parallel:
    # two workers starting in the same second shared a directory and one
    # overwrote the other's args.json, which silently lost two runs from a
    # 40-run sweep. The pid disambiguates, and exist_ok stays off so a
    # collision fails loudly instead of corrupting a result.
    safe_tag = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in args.run_tag.strip()
    ).strip("-")
    run_name = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    if safe_tag:
        run_name += f"_{safe_tag}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    to_dict_recon = lambda r: {"mae": r.mae, "mse": r.mse, "rmse": r.rmse}
    to_dict_anom = lambda m: {"precision": m.precision, "recall": m.recall,
                               "f1": m.f1, "auroc": m.auroc,
                               "auprc": m.auprc}

    payload = {
        "model": "TemporalMixtureOfExpertsGNN",
        "data_dir": str(data_dir),
        "n_params": n_params,
        "num_experts": args.num_experts,
        "attack_classes": class_names,
        "hidden_dim": args.hidden_dim,
        "window_size": args.window_size,
        "reconstruction": {
            "pressure_all": to_dict_recon(test_m["pressure_all"]),
            "pressure_unobs": to_dict_recon(test_m["pressure_unobs"]),
            "flow_all": to_dict_recon(test_m["flow_all"]),
            "flow_unobs": to_dict_recon(test_m["flow_unobs"]),
        },
        "router_acc": test_m["router_acc"],
        "router_confusion": {
            f"{k[0]}_{k[1]}": v for k, v in test_m["router_confusion"].items()
        },
        "per_attack_pressure": test_m["per_attack_pressure"],
        # Decision threshold actually used (calibrated on validation), plus
        # the uncalibrated 0.5 numbers so the gain stays auditable.
        "threshold": test_m.get("threshold", 0.5),
        "norm_mode": getattr(args, "norm_mode", "global"),
        "run_tag": args.run_tag,
        "training": {
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_auprc": best_val,
            "wall_seconds": time.time() - training_started,
            "device": str(device),
        },
    }
    data_config_path = data_dir / "generate_config.yaml"
    if data_config_path.exists():
        payload["dataset_config_sha256"] = hashlib.sha256(
            data_config_path.read_bytes()
        ).hexdigest()
    if "uncalibrated" in test_m:
        payload["uncalibrated"] = test_m["uncalibrated"]
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
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()

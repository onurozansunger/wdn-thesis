"""Self-play training: an Attacker GNN against the Temporal MoE defender.

Phase 1 (MVP) of the self-play roadmap. The attacker observes the
last-timestep features of every sliding window and adds a sparse,
bounded perturbation to the pressure / flow observations. The defender
sees the modified window and is trained to reconstruct the true state
and flag the attacked sensors.

We optimise both networks with alternating gradient steps. The
attacker takes ``--attacker_steps`` updates per defender update so the
defender always faces an attacker that has had time to react to the
latest defender weights.

Outputs are written to ``runs/selfplay/<timestamp>/``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wdn.metrics import compute_anomaly_metrics, compute_recon_metrics
from wdn.models.attacker import AttackerGNN, StealthBudget, apply_stealth_budget
from wdn.models.attacker_moe import (
    MixtureOfAttackersGNN, diversity_loss, balance_loss,
)
from wdn.models.temporal_moe import TemporalMixtureOfExpertsGNN, temporal_moe_loss
from wdn.temporal_dataset import (
    TemporalWDNDataset, create_temporal_dataloaders, temporal_collate_fn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [t.to(device) for t in v]
        else:
            out[k] = v
    return out


def inject_attack(
    batch: dict,
    proj: dict[str, torch.Tensor],
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Replace the last-timestep observations in ``batch`` with the
    attacker's perturbed values, and return updated anomaly labels.

    The attack is added on top of whatever corruption the dataset
    already applied. The mask the defender is supposed to learn to
    flag is the union of (a) the original anomaly mask from the
    corruption pipeline and (b) the new sensors the attacker just
    touched. We force the observation mask to 1 on attacked sensors
    so the defender always *sees* the falsified value.
    """
    new_batch = {**batch}
    last = new_batch["x_seq"][-1].clone()

    # Apply pressure perturbation in-place. Attacked sensors are
    # always observed (mask = 1); the falsified value is the original
    # pressure_obs plus the attacker delta.
    delta_p = proj["delta_p"]
    mask_p = proj["mask_p"]
    last[:, 5] = last[:, 5] + delta_p
    last[:, 6] = torch.clamp(last[:, 6] + mask_p, max=1.0)

    new_x_seq = list(new_batch["x_seq"])
    new_x_seq[-1] = last
    new_batch["x_seq"] = new_x_seq

    # Update the observation tensors used by the defender's anomaly head.
    new_batch["pressure_obs"] = last[:, 5]
    new_batch["pressure_mask"] = last[:, 6]

    # Apply flow perturbation.
    delta_q = proj["delta_q"]
    mask_q = proj["mask_q"]
    new_batch["flow_obs"] = new_batch["flow_obs"] + delta_q
    new_batch["flow_mask"] = torch.clamp(
        new_batch["flow_mask"] + mask_q, max=1.0,
    )

    # Updated anomaly labels: union of original + attacker-touched.
    p_attacked = (proj["delta_p"].abs() > 1e-6).float()
    q_attacked = (proj["delta_q"].abs() > 1e-6).float()
    p_anom = torch.clamp(new_batch["pressure_anomaly"] + p_attacked, max=1.0)
    q_anom = torch.clamp(new_batch["flow_anomaly"] + q_attacked, max=1.0)

    return new_batch, p_anom, q_anom


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def attacker_step(
    attacker, defender, batch, budget, optimizer,
    lambda_recon, lambda_budget, lambda_physics, incidence,
    lambda_diversity=0.0, lambda_atk_balance=0.0,
):
    """One attacker SGD step. Defender weights are frozen."""
    attacker.train()
    defender.eval()
    for p in defender.parameters():
        p.requires_grad_(False)

    # Use the last-timestep clean-ish features as the attacker input.
    x_last = batch["x_seq"][-1]
    is_moe = isinstance(attacker, MixtureOfAttackersGNN)
    if is_moe:
        out = attacker(
            x_last, batch["edge_index"], batch["edge_attr"],
            batch["is_original_edge"],
            num_nodes_per_graph=batch["num_nodes"],
        )
    else:
        out = attacker(
            x_last, batch["edge_index"], batch["edge_attr"],
            batch["is_original_edge"],
        )
    proj = apply_stealth_budget(out, budget, hard=False)

    new_batch, _, _ = inject_attack(batch, proj)

    defender_out = defender(
        x_seq=new_batch["x_seq"],
        edge_index=new_batch["edge_index"],
        edge_attr=new_batch["edge_attr"],
        is_original_edge=new_batch["is_original_edge"],
        batch_size=new_batch["batch_size"],
        num_nodes_per_graph=new_batch["num_nodes"],
        pressure_obs=new_batch["pressure_obs"],
        flow_obs=new_batch["flow_obs"],
        pressure_mask=new_batch["pressure_mask"],
        flow_mask=new_batch["flow_mask"],
    )

    # Attacker wants:
    #   - defender to NOT flag the attacked sensors  (low logits)
    #   - defender's reconstruction to be wrong on the attacked sensors
    p_logits = defender_out.get("pressure_anomaly_logits")
    q_logits = defender_out.get("flow_anomaly_logits")

    p_attacked_mask = (proj["delta_p"].abs() > 1e-6).float()
    q_attacked_mask = (proj["delta_q"].abs() > 1e-6).float()

    stealth_loss = (
        F.binary_cross_entropy_with_logits(
            p_logits, p_attacked_mask.detach() * 0.0, reduction="none",
        ) * p_attacked_mask
    ).sum() / (p_attacked_mask.sum() + 1.0)
    if q_logits is not None:
        stealth_loss = stealth_loss + (
            F.binary_cross_entropy_with_logits(
                q_logits, q_attacked_mask.detach() * 0.0, reduction="none",
            ) * q_attacked_mask
        ).sum() / (q_attacked_mask.sum() + 1.0)

    # Damage loss: maximize defender's reconstruction error on attacked sensors.
    p_err = (defender_out["pressure_pred"] - new_batch["y_pressure"]).pow(2)
    q_err = (defender_out["flow_pred"] - new_batch["y_flow"]).pow(2)
    damage = (
        (p_err * p_attacked_mask).sum() / (p_attacked_mask.sum() + 1.0)
        + (q_err * q_attacked_mask).sum() / (q_attacked_mask.sum() + 1.0)
    )

    # Physics-aware penalty: a "real" cyber-attack must not violate
    # mass conservation, otherwise the attack would be trivially
    # detected by an analytical hydraulic check. We penalise the
    # residual B @ q on the falsified flow vector. ``q`` here is the
    # observed flow (clean + delta) on every original edge.
    physics_loss = batch["flow_obs"].new_zeros(())
    if incidence is not None and lambda_physics > 0:
        B = new_batch["batch_size"]
        NE = incidence.shape[1]
        q_obs = new_batch["flow_obs"].view(B, NE)
        residual = q_obs @ incidence.T          # (B, num_nodes)
        physics_loss = residual.pow(2).mean()

    # The attacker maximizes damage, minimizes stealth_loss, pays a
    # smooth budget penalty (already enforced via top-k + tanh, but we
    # discourage saturation so the gradient signal stays informative)
    # and respects mass conservation.
    loss = (
        stealth_loss
        - lambda_recon * damage
        + lambda_budget * (proj["delta_p"].pow(2).mean()
                           + proj["delta_q"].pow(2).mean())
        + lambda_physics * physics_loss
    )

    # MoE-attacker auxiliary losses: encourage experts to (a) disagree
    # on what to perturb (diversity) and (b) be evenly used by the
    # router (balance). Without these the population collapses to a
    # single mode.
    div_val = bal_val = 0.0
    if is_moe:
        div = diversity_loss(out["expert_delta_p"])
        bal = balance_loss(out["router_logits"])
        loss = loss + lambda_diversity * div + lambda_atk_balance * bal
        div_val = float(div.detach())
        bal_val = float(bal.detach())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(attacker.parameters(), 1.0)
    optimizer.step()

    for p in defender.parameters():
        p.requires_grad_(True)

    return {
        "atk_loss": loss.item(),
        "atk_stealth": stealth_loss.item(),
        "atk_damage": damage.item(),
        "atk_physics": float(physics_loss.detach()),
        "atk_diversity": div_val,
        "atk_balance": bal_val,
    }


def defender_step(
    attacker, defender, batch, budget, optimizer, lambdas,
):
    """One defender SGD step. Attacker weights are frozen."""
    attacker.eval()
    defender.train()
    for p in attacker.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        x_last = batch["x_seq"][-1]
        if isinstance(attacker, MixtureOfAttackersGNN):
            out = attacker(
                x_last, batch["edge_index"], batch["edge_attr"],
                batch["is_original_edge"],
                num_nodes_per_graph=batch["num_nodes"],
            )
        else:
            out = attacker(
                x_last, batch["edge_index"], batch["edge_attr"],
                batch["is_original_edge"],
            )
        proj = apply_stealth_budget(out, budget, hard=True)

    new_batch, p_anom, q_anom = inject_attack(batch, proj)
    new_batch["pressure_anomaly"] = p_anom
    new_batch["flow_anomaly"] = q_anom

    defender_out = defender(
        x_seq=new_batch["x_seq"],
        edge_index=new_batch["edge_index"],
        edge_attr=new_batch["edge_attr"],
        is_original_edge=new_batch["is_original_edge"],
        batch_size=new_batch["batch_size"],
        num_nodes_per_graph=new_batch["num_nodes"],
        pressure_obs=new_batch["pressure_obs"],
        flow_obs=new_batch["flow_obs"],
        pressure_mask=new_batch["pressure_mask"],
        flow_mask=new_batch["flow_mask"],
    )

    losses = temporal_moe_loss(
        defender_out, new_batch,
        lambda_router=lambdas["router"],
        lambda_balance=lambdas["balance"],
        lambda_anomaly=lambdas["anomaly"],
    )

    optimizer.zero_grad()
    losses["total_loss"].backward()
    torch.nn.utils.clip_grad_norm_(defender.parameters(), 1.0)
    optimizer.step()

    for p in attacker.parameters():
        p.requires_grad_(True)

    return {
        "def_loss": losses["total_loss"].item(),
        "def_recon": losses["recon_loss"].item(),
        "def_anomaly": losses["anomaly_loss"].item(),
    }


@torch.no_grad()
def evaluate(attacker, defender, loader, device, budget):
    """Evaluate the (frozen) defender on (a) the dataset's hand-crafted
    attacks and (b) the current attacker's adversarial perturbations.
    Returns anomaly F1 / AUROC on each setting."""
    attacker.eval(); defender.eval()
    p_logits_h, p_lab_h = [], []
    p_logits_a, p_lab_a = [], []

    for raw in loader:
        batch = to_device(raw, device)

        # (a) Hand-crafted attacks already in the corrupted snapshot.
        out_h = defender(
            x_seq=batch["x_seq"], edge_index=batch["edge_index"],
            edge_attr=batch["edge_attr"],
            is_original_edge=batch["is_original_edge"],
            batch_size=batch["batch_size"],
            num_nodes_per_graph=batch["num_nodes"],
            pressure_obs=batch["pressure_obs"], flow_obs=batch["flow_obs"],
            pressure_mask=batch["pressure_mask"], flow_mask=batch["flow_mask"],
        )
        m = batch["pressure_mask"] > 0
        p_logits_h.append(out_h["pressure_anomaly_logits"][m].cpu())
        p_lab_h.append(batch["pressure_anomaly"][m].cpu())

        # (b) Adversarial: attacker perturbs, defender forwards.
        x_last = batch["x_seq"][-1]
        if isinstance(attacker, MixtureOfAttackersGNN):
            atk_out = attacker(
                x_last, batch["edge_index"], batch["edge_attr"],
                batch["is_original_edge"],
                num_nodes_per_graph=batch["num_nodes"],
            )
        else:
            atk_out = attacker(
                x_last, batch["edge_index"], batch["edge_attr"],
                batch["is_original_edge"],
            )
        proj = apply_stealth_budget(atk_out, budget, hard=True)
        adv_batch, p_anom, _ = inject_attack(batch, proj)
        out_a = defender(
            x_seq=adv_batch["x_seq"], edge_index=adv_batch["edge_index"],
            edge_attr=adv_batch["edge_attr"],
            is_original_edge=adv_batch["is_original_edge"],
            batch_size=adv_batch["batch_size"],
            num_nodes_per_graph=adv_batch["num_nodes"],
            pressure_obs=adv_batch["pressure_obs"],
            flow_obs=adv_batch["flow_obs"],
            pressure_mask=adv_batch["pressure_mask"],
            flow_mask=adv_batch["flow_mask"],
        )
        m2 = adv_batch["pressure_mask"] > 0
        p_logits_a.append(out_a["pressure_anomaly_logits"][m2].cpu())
        p_lab_a.append(p_anom[m2].cpu())

    logits_h = torch.cat(p_logits_h); lab_h = torch.cat(p_lab_h)
    logits_a = torch.cat(p_logits_a); lab_a = torch.cat(p_lab_a)
    h = compute_anomaly_metrics(
        (logits_h > 0).long(), (lab_h > 0.5).long(), scores=torch.sigmoid(logits_h),
    )
    a = compute_anomaly_metrics(
        (logits_a > 0).long(), (lab_a > 0.5).long(), scores=torch.sigmoid(logits_a),
    )
    return {"hand_f1": h.f1, "hand_auroc": h.auroc,
            "adv_f1":  a.f1, "adv_auroc":  a.auroc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/temporal_moe_modena")
    parser.add_argument("--defender_ckpt", type=str, default=None,
                        help="Pretrained Temporal MoE checkpoint to fine-tune. "
                             "If omitted, defender is trained from scratch.")
    parser.add_argument("--gnn_type", type=str, default="GraphSAGE")
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--num_experts", type=int, default=6)
    parser.add_argument("--window_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--attacker_lr", type=float, default=1e-3)
    parser.add_argument("--defender_lr", type=float, default=5e-4)
    parser.add_argument("--attacker_steps", type=int, default=2)
    parser.add_argument("--defender_steps", type=int, default=1)
    parser.add_argument("--epsilon_p", type=float, default=2.0)
    parser.add_argument("--epsilon_q", type=float, default=0.05)
    parser.add_argument("--k_p", type=int, default=4)
    parser.add_argument("--k_q", type=int, default=4)
    parser.add_argument("--lambda_recon", type=float, default=0.5)
    parser.add_argument("--lambda_budget", type=float, default=0.01)
    parser.add_argument("--lambda_physics", type=float, default=0.1)
    parser.add_argument("--curriculum", action="store_true",
                        help="Grow epsilon and k as the defender improves.")
    parser.add_argument("--curriculum_threshold", type=float, default=0.85,
                        help="Bump budget once adv F1 exceeds this on val.")
    parser.add_argument("--epsilon_p_max", type=float, default=5.0)
    parser.add_argument("--k_p_max", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attacker_moe", action="store_true",
                        help="Use a population (Mixture-of-Attackers) instead "
                             "of a single attacker.")
    parser.add_argument("--num_attackers", type=int, default=4)
    parser.add_argument("--lambda_diversity", type=float, default=0.1)
    parser.add_argument("--lambda_atk_balance", type=float, default=0.05)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)

    device = get_device()
    print(f"Device: {device}, seed: {args.seed}")

    # ---- Data ----
    data_dir = Path(args.data_dir)
    with open(data_dir / "graph.pkl", "rb") as f:
        graph = pickle.load(f)
    with open(data_dir / "snapshots.pkl", "rb") as f:
        snapshots = pickle.load(f)
    with open(data_dir / "corrupted.pkl", "rb") as f:
        corrupted = pickle.load(f)

    print(f"Loaded {len(snapshots)} snapshots; {graph.num_nodes} nodes, "
          f"{graph.num_edges} edges")

    train_loader, val_loader, _, normalizer = create_temporal_dataloaders(
        snapshots[:int(0.7 * len(snapshots))],
        corrupted[:int(0.7 * len(corrupted))],
        snapshots[int(0.7 * len(snapshots)):int(0.85 * len(snapshots))],
        corrupted[int(0.7 * len(corrupted)):int(0.85 * len(corrupted))],
        snapshots[int(0.85 * len(snapshots)):],
        corrupted[int(0.85 * len(corrupted)):],
        window_size=args.window_size,
        batch_size=args.batch_size,
    )

    # ---- Models ----
    sample = next(iter(train_loader))
    node_in_dim = sample["x_seq"][0].shape[1]
    edge_in_dim = sample["edge_attr"].shape[1]

    defender = TemporalMixtureOfExpertsGNN(
        node_in_dim=node_in_dim, edge_in_dim=edge_in_dim,
        hidden_dim=args.hidden_dim, num_experts=args.num_experts,
        window_size=args.window_size, gnn_type=args.gnn_type,
    ).to(device)
    if args.defender_ckpt:
        sd = torch.load(args.defender_ckpt, map_location=device)
        defender.load_state_dict(sd)
        print(f"Loaded defender from {args.defender_ckpt}")

    if args.attacker_moe:
        attacker = MixtureOfAttackersGNN(
            node_in_dim=node_in_dim, edge_in_dim=edge_in_dim,
            hidden_dim=args.hidden_dim, gnn_type=args.gnn_type,
            num_experts=args.num_attackers,
        ).to(device)
        print(f"Using MixtureOfAttackersGNN with {args.num_attackers} experts")
    else:
        attacker = AttackerGNN(
            node_in_dim=node_in_dim, edge_in_dim=edge_in_dim,
            hidden_dim=args.hidden_dim, gnn_type=args.gnn_type,
        ).to(device)

    n_def = sum(p.numel() for p in defender.parameters())
    n_atk = sum(p.numel() for p in attacker.parameters())
    print(f"Defender: {n_def:,} params  |  Attacker: {n_atk:,} params")

    atk_opt = torch.optim.Adam(attacker.parameters(), lr=args.attacker_lr)
    def_opt = torch.optim.Adam(defender.parameters(), lr=args.defender_lr)

    budget = StealthBudget(
        epsilon_p=args.epsilon_p, epsilon_q=args.epsilon_q,
        k_p=args.k_p, k_q=args.k_q,
    )

    lambdas = dict(router=0.5, balance=0.01, anomaly=1.0)

    # Mass-conservation incidence matrix (B) on the original edges only.
    incidence = torch.tensor(graph.incidence_matrix, dtype=torch.float32).to(device)

    # ---- Training loop ----
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        atk_log = {"atk_loss": 0.0, "atk_stealth": 0.0,
                   "atk_damage": 0.0, "atk_physics": 0.0,
                   "atk_diversity": 0.0, "atk_balance": 0.0}
        def_log = {"def_loss": 0.0, "def_recon": 0.0, "def_anomaly": 0.0}
        n_atk_steps = n_def_steps = 0

        for raw in train_loader:
            batch = to_device(raw, device)

            for _ in range(args.attacker_steps):
                m = attacker_step(
                    attacker, defender, batch, budget, atk_opt,
                    args.lambda_recon, args.lambda_budget,
                    args.lambda_physics, incidence,
                    lambda_diversity=args.lambda_diversity,
                    lambda_atk_balance=args.lambda_atk_balance,
                )
                for k, v in m.items():
                    atk_log[k] += v
                n_atk_steps += 1

            for _ in range(args.defender_steps):
                m = defender_step(
                    attacker, defender, batch, budget, def_opt, lambdas,
                )
                for k, v in m.items():
                    def_log[k] += v
                n_def_steps += 1

        for k in atk_log:
            atk_log[k] /= max(n_atk_steps, 1)
        for k in def_log:
            def_log[k] /= max(n_def_steps, 1)

        elapsed = time.time() - t0

        # Validation: how good is the defender on (a) hand-crafted attacks
        # and (b) the current attacker's adversarial perturbations?
        val = evaluate(attacker, defender, val_loader, device, budget)

        # Auto-curriculum: if the defender is dominating the attacker on
        # the held-out set, raise the budget so the attacker stays in
        # contention.
        if args.curriculum and val["adv_f1"] > args.curriculum_threshold:
            new_eps = min(budget.epsilon_p + 0.5, args.epsilon_p_max)
            new_k = min(budget.k_p + 2, args.k_p_max)
            if new_eps > budget.epsilon_p or new_k > budget.k_p:
                budget = StealthBudget(
                    epsilon_p=new_eps, epsilon_q=budget.epsilon_q,
                    k_p=new_k, k_q=budget.k_q,
                )
                print(f"  [curriculum] -> epsilon_p={new_eps}, k_p={new_k}")

        entry = {
            "epoch": epoch, "elapsed_s": elapsed,
            **atk_log, **def_log, **val,
            "epsilon_p": budget.epsilon_p, "k_p": budget.k_p,
        }
        history.append(entry)
        print(
            f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s) | "
            f"atk dmg={atk_log['atk_damage']:.3f} steal={atk_log['atk_stealth']:.2f} "
            f"phys={atk_log['atk_physics']:.3f} | "
            f"def recon={def_log['def_recon']:.3f} anom={def_log['def_anomaly']:.3f} | "
            f"val hand_F1={val['hand_f1']:.3f} adv_F1={val['adv_f1']:.3f}"
        )

    # ---- Save ----
    run_dir = Path("runs/selfplay") / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(attacker.state_dict(), run_dir / "attacker.pt")
    torch.save(defender.state_dict(), run_dir / "defender.pt")
    torch.save(normalizer.state_dict(), run_dir / "normalizer.pt")
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()

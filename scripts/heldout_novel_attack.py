"""Hold-out evaluation: defenders are tested on a *novel* attack type
that was not in the training set.

We synthesise two attacks that don't match any of the five hand-crafted
classes the defenders have seen:
    - sinusoidal: superimpose A * sin(omega * t + phi) on the pressure
      reading over the window. Clearly different from step-like
      random / targeted / noise injection or copy-style replay.
    - sensor swap: swap the pressure observation of two attacked
      sensors. Cross-sensor confusion — none of the 5 baselines do this.

For each test snapshot we apply one of the novel attacks to a small
random subset of pressure sensors and feed the corrupted snapshot to
the defender. We report anomaly F1 / AUROC achieved by:
    - the supervised pretrained defender
    - the self-play single-attacker defender
    - the self-play attacker-MoE defender
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wdn.metrics import compute_anomaly_metrics
from wdn.models.temporal_moe import TemporalMixtureOfExpertsGNN
from wdn.temporal_dataset import create_temporal_dataloaders


DD = ROOT / "data" / "temporal_moe_modena"
CKPTS = {
    "pretrained": ROOT / "runs/temporal_moe/20260505_144409/best_model.pt",
    "sp_single":  ROOT / "runs/selfplay/20260505_223529/defender.pt",
    "sp_moe":     ROOT / "runs/selfplay/20260506_110630/defender.pt",
}


def make_loaders(seed: int = 0):
    snaps = pickle.load(open(DD / "snapshots.pkl", "rb"))
    corr = pickle.load(open(DD / "corrupted.pkl", "rb"))
    n = len(snaps); nt = int(0.7 * n); nv = int(0.15 * n)
    _, _, tl, _ = create_temporal_dataloaders(
        snaps[:nt], corr[:nt], snaps[nt:nt+nv], corr[nt:nt+nv],
        snaps[nt+nv:], corr[nt+nv:], window_size=6, batch_size=8,
    )
    return tl


def apply_novel_attack(batch: dict, kind: str, k: int, rng: np.random.Generator):
    """Apply ``kind`` to a random subset of k sensors per graph in the
    batch. Returns a new batch with modified ``x_seq[-1]`` columns 5
    (pressure_obs) and 6 (pressure_mask) and a fresh ``pressure_anomaly``
    label for the perturbed entries.
    """
    new_batch = {**batch}
    x_seq = [t.clone() for t in batch["x_seq"]]
    B = batch["batch_size"]
    N = batch["num_nodes"]
    p_anom = batch["pressure_anomaly"].clone()

    for b in range(B):
        # Pick k sensors for this graph that are observable.
        offset = b * N
        sensors = rng.choice(N, size=k, replace=False)

        if kind == "sinusoidal":
            # Inject sin(omega * t) with a random phase per sensor on
            # the pressure_obs column at every timestep.
            omega = 2 * np.pi / 6
            amp = 1.5  # in normalised units
            for s in sensors:
                phi = rng.uniform(0, 2 * np.pi)
                for t in range(len(x_seq)):
                    val = amp * np.sin(omega * t + phi)
                    x_seq[t][offset + s, 5] = (
                        x_seq[t][offset + s, 5] + float(val)
                    )
                    x_seq[t][offset + s, 6] = 1.0  # force observed
                p_anom[offset + s] = 1.0

        elif kind == "swap":
            # Swap the last-timestep pressure obs of pairs of sensors.
            sensors = sensors[:max(2, k - (k % 2))]
            for i in range(0, len(sensors), 2):
                a, b_ = sensors[i], sensors[i + 1]
                v_a = x_seq[-1][offset + a, 5].clone()
                v_b = x_seq[-1][offset + b_, 5].clone()
                x_seq[-1][offset + a, 5] = v_b
                x_seq[-1][offset + b_, 5] = v_a
                x_seq[-1][offset + a, 6] = 1.0
                x_seq[-1][offset + b_, 6] = 1.0
                p_anom[offset + a] = 1.0
                p_anom[offset + b_] = 1.0
        else:
            raise ValueError(kind)

    new_batch["x_seq"] = x_seq
    new_batch["pressure_obs"] = x_seq[-1][:, 5]
    new_batch["pressure_mask"] = x_seq[-1][:, 6]
    new_batch["pressure_anomaly"] = p_anom
    return new_batch


@torch.no_grad()
def evaluate_under(model, loader, device, kind, k, seed):
    rng = np.random.default_rng(seed)
    L, La = [], []
    for raw in loader:
        b = {k_: (v.to(device) if isinstance(v, torch.Tensor)
                  else [t.to(device) for t in v] if isinstance(v, list) else v)
             for k_, v in raw.items()}
        b = apply_novel_attack(b, kind, k, rng)
        o = model(
            x_seq=b["x_seq"], edge_index=b["edge_index"],
            edge_attr=b["edge_attr"],
            is_original_edge=b["is_original_edge"],
            batch_size=b["batch_size"],
            num_nodes_per_graph=b["num_nodes"],
            pressure_obs=b["pressure_obs"], flow_obs=b["flow_obs"],
            pressure_mask=b["pressure_mask"], flow_mask=b["flow_mask"],
        )
        sel = b["pressure_mask"] > 0
        L.append(o["pressure_anomaly_logits"][sel].cpu())
        La.append(b["pressure_anomaly"][sel].cpu())
    Lt = torch.cat(L); Lat = torch.cat(La)
    m = compute_anomaly_metrics(
        (Lt > 0).long(), (Lat > 0.5).long(), scores=torch.sigmoid(Lt))
    return {"f1": m.f1, "auroc": m.auroc,
            "precision": m.precision, "recall": m.recall}


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    loader = make_loaders()
    models = {}
    for name, path in CKPTS.items():
        m = TemporalMixtureOfExpertsGNN(
            node_in_dim=7, edge_in_dim=8, hidden_dim=48,
            num_experts=6, window_size=6, gnn_type="GraphSAGE",
        ).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models[name] = m

    results = {}
    for kind in ("sinusoidal", "swap"):
        print(f"\n=== Novel attack: {kind} (k=15 sensors per graph) ===")
        print(f"{'defender':12s} {'F1':>8s} {'AUROC':>8s} {'P':>8s} {'R':>8s}")
        results[kind] = {}
        for name, m in models.items():
            r = evaluate_under(m, loader, device, kind, k=15, seed=0)
            results[kind][name] = r
            print(f"{name:12s} {r['f1']:>8.3f} {r['auroc']:>8.3f} "
                  f"{r['precision']:>8.3f} {r['recall']:>8.3f}")

    out_path = ROOT / "runs/selfplay/eval_heldout.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()

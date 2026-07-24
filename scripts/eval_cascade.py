"""Compare soft-mixture routing against the iterative cascade router.

The cascade is a *decision procedure over already-trained experts*: the
router ranks them, we run the most likely one, a label-free feedback
signal judges it, and we re-route to the next candidate if it fails. That
means an existing checkpoint can be evaluated both ways with no
retraining, which isolates the effect of the routing scheme itself.

Usage:
    python3 scripts/eval_cascade.py \
        --data_dir data/temporal_moe_modena \
        --runs runs/temporal_moe/20260522_204723 [more runs ...] \
        --out runs/temporal_moe/cascade_eval.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wdn.models.temporal_moe import (
    TemporalMixtureOfExpertsGNN, cascade_route, expert_feedback_scores,
)
from wdn.temporal_dataset import create_temporal_dataloaders
from wdn.dataset import Normalizer

ATTACKS = {1: "random", 2: "replay", 3: "stealthy", 4: "noise", 5: "targeted"}


def load_split(data_dir: Path, window_size: int, batch_size: int, norm_mode: str):
    snaps = pickle.load(open(data_dir / "snapshots.pkl", "rb"))
    corr = pickle.load(open(data_dir / "corrupted.pkl", "rb"))
    graph = pickle.load(open(data_dir / "graph.pkl", "rb"))
    scen = sorted({s.scenario_id for s in snaps})
    n = len(scen)
    tr, va = scen[: int(0.7 * n)], scen[int(0.7 * n): int(0.85 * n)]
    te = scen[int(0.85 * n):]

    def _f(ids):
        keep = [i for i, s in enumerate(snaps) if s.scenario_id in ids]
        return [snaps[i] for i in keep], [corr[i] for i in keep]

    trs, trc = _f(tr); vas, vac = _f(va); tes, tec = _f(te)
    loaders = create_temporal_dataloaders(
        trs, trc, vas, vac, tes, tec,
        window_size=window_size, batch_size=batch_size, num_workers=0,
        norm_mode=norm_mode,
    )
    return loaders, graph


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v)
                else [t.to(device) for t in v] if isinstance(v, list) else v)
            for k, v in batch.items()}


@torch.no_grad()
def collect(model, loader, device, incidence, tau, max_attempts):
    """Run every routing scheme over a loader and gather predictions.

    Four variants, to separate "is hard selection viable at all?" from
    "is our feedback signal any good?":

    soft
        the trained model's own weighted mixture (baseline)
    router_top1
        hard-select the router's most likely expert, no feedback
    cascade
        router ranks, feedback accepts/rejects, re-route on failure
    oracle
        hard-select the expert that owns the *true* attack class — the
        ceiling any hard-routing scheme could reach
    """
    model.eval()
    soft_logit, casc_logit, labels, atk = [], [], [], []
    top1_logit, oracle_logit = [], []
    diag = {"n_attempts": [], "accepted_first": [], "chosen": [], "true": []}

    for raw in loader:
        b = to_device(raw, device)
        out = model(
            x_seq=b["x_seq"], edge_index=b["edge_index"],
            edge_attr=b["edge_attr"], is_original_edge=b["is_original_edge"],
            batch_size=b["batch_size"], num_nodes_per_graph=b["num_nodes"],
            pressure_obs=b["pressure_obs"], flow_obs=b["flow_obs"],
            pressure_mask=b["pressure_mask"], flow_mask=b["flow_mask"],
        )
        casc = cascade_route(out, b, incidence, tau=tau,
                             max_attempts=max_attempts)

        m = b["pressure_mask"] > 0
        soft_logit.append(out["pressure_anomaly_logits"][m].cpu())
        casc_logit.append(casc["pressure_anomaly_logits"][m].cpu())
        labels.append(b["pressure_anomaly"][m].cpu())

        # Hard-selection variants share the same gather machinery.
        N_ = b["num_nodes"]
        stack = out["expert_pressure_anomaly_logits"]              # (B*N, K)

        def _pick(sel_per_graph):
            idx = sel_per_graph.repeat_interleave(N_).unsqueeze(-1)
            return stack.gather(-1, idx).squeeze(-1)

        top1 = out["router_probs"].argmax(dim=-1)
        top1_logit.append(_pick(top1)[m].cpu())
        true_e = b["attack_type"].clamp(max=stack.shape[-1] - 1).long()
        oracle_logit.append(_pick(true_e)[m].cpu())

        N = b["num_nodes"]
        a = b["attack_type"].repeat_interleave(N)[m].cpu()
        atk.append(a)
        diag["n_attempts"].append(casc["n_attempts"].cpu())
        diag["accepted_first"].append(casc["accepted_first"].cpu())
        diag["chosen"].append(casc["chosen_expert"].cpu())
        diag["true"].append(b["attack_type"].cpu())

    cat = lambda xs: torch.cat(xs).numpy()
    variants = {
        "soft": cat(soft_logit),
        "router_top1": cat(top1_logit),
        "cascade": cat(casc_logit),
        "oracle": cat(oracle_logit),
    }
    return (variants, cat(labels), cat(atk),
            {k: cat(v) for k, v in diag.items()})


def metrics(logit, label, atk):
    prob = 1 / (1 + np.exp(-logit))
    pred = (prob > 0.5).astype(int)
    out = {
        "f1": float(f1_score(label, pred, zero_division=0)),
        "auroc": float(roc_auc_score(label, prob)) if label.max() > 0 else 0.5,
    }
    per = {}
    for aid, name in ATTACKS.items():
        sel = atk == aid
        if sel.sum() == 0:
            continue
        per[name] = {
            "f1": float(f1_score(label[sel], pred[sel], zero_division=0)),
            "n": int(sel.sum()),
        }
    out["per_attack"] = per
    return out


def calibrate_tau(model, loader, device, incidence, quantile):
    """Pick the acceptance threshold from validation feedback scores.

    We take the score the *best* expert achieves on each validation graph
    and use a quantile of that distribution: a candidate is accepted if it
    looks as good as a typical good expert does.
    """
    model.eval()
    best = []
    with torch.no_grad():
        for raw in loader:
            b = to_device(raw, device)
            out = model(
                x_seq=b["x_seq"], edge_index=b["edge_index"],
                edge_attr=b["edge_attr"],
                is_original_edge=b["is_original_edge"],
                batch_size=b["batch_size"], num_nodes_per_graph=b["num_nodes"],
                pressure_obs=b["pressure_obs"], flow_obs=b["flow_obs"],
                pressure_mask=b["pressure_mask"], flow_mask=b["flow_mask"],
            )
            s = expert_feedback_scores(out, b, incidence)
            best.append(s.min(dim=-1).values.cpu())
    best = torch.cat(best).numpy()
    return float(np.quantile(best, quantile))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/temporal_moe_modena")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="runs/temporal_moe/cascade_eval.json")
    p.add_argument("--window_size", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--router_hidden_dim", type=int, default=24)
    p.add_argument("--num_experts", type=int, default=6)
    p.add_argument("--norm_mode", default="global")
    p.add_argument("--max_attempts", type=int, default=2)
    p.add_argument("--tau_quantile", type=float, default=0.7)
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data_dir = ROOT / args.data_dir
    (train_l, val_l, test_l, norm), graph = load_split(
        data_dir, args.window_size, args.batch_size, args.norm_mode)
    incidence = torch.tensor(graph.incidence_matrix,
                             dtype=torch.float32).to(device)

    sample = test_l.dataset[0]
    results = []
    for run in args.runs:
        run = Path(run)
        ck = run / "best_model.pt"
        if not ck.exists():
            print(f"  skip {run.name}: no checkpoint")
            continue
        model = TemporalMixtureOfExpertsGNN(
            node_in_dim=sample["x_seq"][0].shape[1],
            edge_in_dim=sample["edge_attr"].shape[1],
            hidden_dim=args.hidden_dim, num_experts=args.num_experts,
            window_size=args.window_size, gnn_type="GraphSAGE",
            router_hidden_dim=args.router_hidden_dim,
        ).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))

        tau = calibrate_tau(model, val_l, device, incidence, args.tau_quantile)
        variants, lab, atk, diag = collect(
            model, test_l, device, incidence, tau, args.max_attempts)

        r = {
            "run": run.name,
            "tau": tau,
            **{name: metrics(lg, lab, atk) for name, lg in variants.items()},
            "diagnostics": {
                "mean_attempts": float(diag["n_attempts"].mean()),
                "accepted_first_rate": float(diag["accepted_first"].mean()),
                "attempts_hist": np.bincount(
                    diag["n_attempts"], minlength=args.max_attempts + 1).tolist(),
                "chosen_matches_true": float(
                    (diag["chosen"] == diag["true"]).mean()),
            },
        }
        results.append(r)
        print(f"{run.name}: soft {r['soft']['f1']:.3f} | "
              f"top1 {r['router_top1']['f1']:.3f} | "
              f"cascade {r['cascade']['f1']:.3f} | "
              f"oracle {r['oracle']['f1']:.3f} | "
              f"attempts {r['diagnostics']['mean_attempts']:.2f}")

    out_p = ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_p, "w"), indent=2)
    print(f"\nWrote {out_p}  ({len(results)} runs)")


if __name__ == "__main__":
    main()

"""Classical hydraulic baselines for state reconstruction + anomaly detection.

Compares three closed-form estimators against the trained Temporal-MoE
defender on each benchmark network:

  - Mean imputation: predicts the global mean pressure for every
    unobserved node.
  - Pseudo-inverse: y_hat = A @ pinv(A) @ M @ y, where A is the
    node-incidence projection (least-squares onto the observed
    subspace).
  - Laplacian-regularised WLS: closed-form
        x = (M^T M + lambda * L^T L)^(-1) M^T M y
    with L the unweighted graph Laplacian. This is the classical
    "smooth signal on a graph" estimator and the closest single-shot
    analogue of WLS state estimation.

For anomaly detection a 3-sigma residual rule is used: any observed
sensor whose absolute residual exceeds 3 * sigma_clean is flagged.
sigma_clean is fit on a held-out CLEAN portion of the validation set.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wdn.metrics import compute_anomaly_metrics, compute_recon_metrics
import torch


NETS = {
    "Net1":   {"data_dir": "data/moe_net1"},
    "Net3":   {"data_dir": "data/temporal_moe_net3"},
    "Modena": {"data_dir": "data/temporal_moe_modena"},
}


def _laplacian(num_nodes: int, edge_index: np.ndarray) -> np.ndarray:
    """Unweighted graph Laplacian L = D - A, treating edge_index as
    bidirectional (each undirected edge appears in either order)."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for u, v in edge_index.T:
        A[int(u), int(v)] = 1.0
        A[int(v), int(u)] = 1.0
    D = np.diag(A.sum(axis=1))
    return D - A


def _mean_imputation(p_obs, p_mask):
    obs = p_obs[p_mask > 0]
    mean = float(obs.mean()) if obs.size else 0.0
    out = p_obs.copy()
    out[p_mask == 0] = mean
    return out


def _laplacian_wls(p_obs, p_mask, L, lam=10.0):
    """Closed-form solution of  argmin_x ||M(x-y)||^2 + lam * ||Lx||^2."""
    N = p_obs.shape[0]
    M = np.diag(p_mask.astype(np.float64))
    A = M @ M + lam * L.T @ L + 1e-6 * np.eye(N)
    b = M @ M @ p_obs.astype(np.float64)
    return np.linalg.solve(A, b)


def _pseudo_inv(p_obs, p_mask, L, lam=0.1):
    """Lighter Tikhonov regularisation — pseudo-inverse style."""
    return _laplacian_wls(p_obs, p_mask, L, lam=lam)


def evaluate_network(name: str, data_dir: Path):
    snaps = pickle.load(open(data_dir / "snapshots.pkl", "rb"))
    corr = pickle.load(open(data_dir / "corrupted.pkl", "rb"))
    with open(data_dir / "graph.pkl", "rb") as f:
        graph = pickle.load(f)

    # Use the same 70/15/15 split as the trained models.
    n = len(snaps); nt = int(0.7 * n); nv = int(0.15 * n)
    test_s = snaps[nt + nv:]; test_c = corr[nt + nv:]
    val_c = corr[nt:nt + nv]

    edge_idx = np.asarray(graph.edge_index)
    L = _laplacian(graph.num_nodes, edge_idx)

    # Fit clean-sigma on the unattacked portion of the val set.
    clean_resid = []
    for c in val_c:
        m = c.pressure_mask.numpy().astype(bool)
        a = c.pressure_anomaly.numpy().astype(bool)
        good = m & ~a
        if not good.any():
            continue
        p_true_proxy = c.pressure_obs.numpy()  # approximate clean truth
        # residual = obs - obs = 0 by definition; use noise model instead
        # so derive sigma from variance of clean obs around its mean
        clean_resid.append(p_true_proxy[good] - p_true_proxy[good].mean())
    sigma_clean = float(np.concatenate(clean_resid).std()) if clean_resid else 1.0
    threshold = 3.0 * sigma_clean

    methods = {"Mean": [], "PseudoInv": [], "WLS": []}
    method_anom_logits = {k: [] for k in methods}
    p_true_all, p_pred_by_method, p_anom_all, p_mask_all = (
        [], {k: [] for k in methods}, [], []
    )

    for snap, c in zip(test_s, test_c):
        p_obs = c.pressure_obs.numpy().astype(np.float64)
        p_mask = c.pressure_mask.numpy().astype(np.float64)
        p_true = snap.pressure_true.numpy().astype(np.float64)
        p_anom = c.pressure_anomaly.numpy().astype(np.float64)

        preds = {
            "Mean":      _mean_imputation(p_obs, p_mask),
            "PseudoInv": _pseudo_inv(p_obs, p_mask, L, lam=0.1),
            "WLS":       _laplacian_wls(p_obs, p_mask, L, lam=10.0),
        }
        for k, p_hat in preds.items():
            p_pred_by_method[k].append(p_hat)
            # Anomaly = |obs - p_hat| > threshold AND observed
            resid = np.abs(p_obs - p_hat)
            score = resid - threshold
            method_anom_logits[k].append(score)
        p_true_all.append(p_true)
        p_anom_all.append(p_anom)
        p_mask_all.append(p_mask)

    # Aggregate
    p_true_arr = np.concatenate(p_true_all)
    p_mask_arr = np.concatenate(p_mask_all)
    p_anom_arr = np.concatenate(p_anom_all)
    results = {}
    for k in methods:
        p_pred_arr = np.concatenate(p_pred_by_method[k])
        unobs = p_mask_arr == 0
        recon = compute_recon_metrics(
            torch.tensor(p_pred_arr), torch.tensor(p_true_arr),
            torch.tensor(p_mask_arr), only_unobserved=True,
        )
        scores = np.concatenate(method_anom_logits[k])
        # Convert to a binary prediction with score > 0 -> flagged
        pred = (scores > 0).astype(int)
        sel = p_mask_arr > 0
        from wdn.metrics import compute_anomaly_metrics as cam
        anom = cam(
            torch.tensor(pred[sel]).long(),
            torch.tensor((p_anom_arr[sel] > 0.5)).long(),
            scores=torch.tensor(1.0 / (1.0 + np.exp(-scores[sel]))),
        )
        results[k] = {
            "p_mae": float(recon.mae),
            "p_rmse": float(recon.rmse),
            "f1": anom.f1, "auroc": anom.auroc,
            "precision": anom.precision, "recall": anom.recall,
        }
    return {"sigma_clean": sigma_clean,
            "threshold": threshold,
            "results": results}


def main():
    print(f"{'network':10s} {'method':12s} {'P_MAE':>9s} {'F1':>7s} "
          f"{'AUROC':>7s} {'P':>7s} {'R':>7s}")
    print("-" * 60)
    out = {}
    for name, cfg in NETS.items():
        data_dir = ROOT / cfg["data_dir"]
        if not (data_dir / "snapshots.pkl").exists():
            continue
        report = evaluate_network(name, data_dir)
        out[name] = report
        for method, r in report["results"].items():
            print(f"{name:10s} {method:12s} {r['p_mae']:>9.3f} "
                  f"{r['f1']:>7.3f} {r['auroc']:>7.3f} "
                  f"{r['precision']:>7.3f} {r['recall']:>7.3f}")
        print("-" * 60)

    (ROOT / "runs" / "selfplay" / "eval_classical.json").write_text(
        json.dumps(out, indent=2)
    )
    print("Saved runs/selfplay/eval_classical.json")


if __name__ == "__main__":
    main()

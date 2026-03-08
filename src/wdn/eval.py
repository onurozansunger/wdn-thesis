from __future__ import annotations

import argparse
import json
import shutil
import logging
from pathlib import Path

import numpy as np
import torch

from wdn.baselines import (
    BaselineConfig,
    analytical_baseline,
    residual_anomaly_scores,
    wls_baseline,
)
from wdn.config import load_eval_config
from wdn.dataset import create_dataloaders, load_npz, split_indices
from wdn.metrics import classification_metrics, recon_metrics
from wdn.models.multitask import MultiTaskGNN
from wdn.models.recon import ReconGNN
from wdn.normalization import load_normalizer
from wdn.plotting import plot_recon_errors
from wdn.train_utils import batch_to_device
from wdn.utils import get_device, make_run_dir, set_seed, setup_logging


logger = logging.getLogger("wdn.eval")


def _load_normalizer_from_model(model_path: str):
    model_dir = Path(model_path).parent.parent
    norm_path = model_dir / "artifacts" / "normalizer.yaml"
    if norm_path.exists():
        return load_normalizer(norm_path)
    return None


def _stratified_recon_metrics(p_true_list, p_pred_list, p_mask_list, bins):
    missing_rates = [1.0 - float(np.mean(mask)) for mask in p_mask_list]
    results = []
    for low, high in bins:
        idx = [i for i, r in enumerate(missing_rates) if low <= r < high]
        if not idx:
            results.append({"bin": [low, high], "mae": None, "mse": None})
            continue
        p_true = np.concatenate([p_true_list[i] for i in idx])
        p_pred = np.concatenate([p_pred_list[i] for i in idx])
        metrics = recon_metrics(p_true, p_pred)
        results.append({"bin": [low, high], **metrics})
    return results


def eval_models(config_path: str) -> Path:
    cfg = load_eval_config(config_path)
    set_seed(cfg.seed)

    run_info = make_run_dir(Path(cfg.run_dir))
    setup_logging(run_info.run_dir / "eval.log")

    shutil.copy(config_path, run_info.run_dir / "config_eval.yaml")
    data = load_npz(cfg.data.dataset_path)
    splits = split_indices(data["P_true"].shape[0], cfg.data.split, cfg.seed)
    _, _, test_loader = create_dataloaders(
        data,
        splits,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        normalizer=None,
    )

    metrics_out = {}
    errors_for_plot = {}

    if cfg.baseline_enabled:
        baseline_cfg = BaselineConfig(
            pinv_rcond=cfg.baseline.get("pinv_rcond", 1e-4),
            wls_alpha=cfg.baseline.get("wls_alpha", 1e-2),
            wls_beta=cfg.baseline.get("wls_beta", 1e-2),
            diag_eps=cfg.baseline.get("diag_eps", 1e-6),
        )
        for name, recon_fn in {
            "baseline_analytical": analytical_baseline,
            "baseline_wls": wls_baseline,
        }.items():
            p_true_all = []
            p_pred_all = []
            q_true_all = []
            q_pred_all = []
            p_mask_all = []
            q_mask_all = []
            p_anom_all = []
            q_anom_all = []
            p_score_all = []
            q_score_all = []

            for batch in test_loader:
                for sample in batch:
                    p_hat, q_hat = recon_fn(
                        sample["P_obs"].numpy(),
                        sample["Q_obs"].numpy(),
                        sample["P_mask"].numpy(),
                        sample["Q_mask"].numpy(),
                        sample["edge_index"].numpy(),
                        baseline_cfg,
                    )
                    p_true_all.append(sample["P_true"].numpy())
                    q_true_all.append(sample["Q_true"].numpy())
                    p_pred_all.append(p_hat)
                    q_pred_all.append(q_hat)
                    p_mask_all.append(sample["P_mask"].numpy())
                    q_mask_all.append(sample["Q_mask"].numpy())

                    p_scores, _ = residual_anomaly_scores(
                        sample["P_obs"].numpy(),
                        p_hat,
                        sample["P_mask"].numpy(),
                        mad_scale=cfg.baseline.get("mad_scale", 3.5),
                    )
                    q_scores, _ = residual_anomaly_scores(
                        sample["Q_obs"].numpy(),
                        q_hat,
                        sample["Q_mask"].numpy(),
                        mad_scale=cfg.baseline.get("mad_scale", 3.5),
                    )
                    p_score_all.append(p_scores)
                    q_score_all.append(q_scores)
                    p_anom_all.append(sample["P_anom"].numpy())
                    q_anom_all.append(sample["Q_anom"].numpy())

            p_true = np.concatenate(p_true_all)
            q_true = np.concatenate(q_true_all)
            p_pred = np.concatenate(p_pred_all)
            q_pred = np.concatenate(q_pred_all)
            metrics_out[name] = {
                "pressure": recon_metrics(p_true, p_pred),
                "flow": recon_metrics(q_true, q_pred),
                "pressure_by_missing": _stratified_recon_metrics(
                    p_true_all, p_pred_all, p_mask_all, bins=[(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
                ),
                "flow_by_missing": _stratified_recon_metrics(
                    q_true_all, q_pred_all, q_mask_all, bins=[(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
                ),
                "pressure_anom": classification_metrics(np.concatenate(p_anom_all), np.concatenate(p_score_all)),
                "flow_anom": classification_metrics(np.concatenate(q_anom_all), np.concatenate(q_score_all)),
            }
            errors_for_plot[f\"{name}_pressure\"] = p_true - p_pred
            errors_for_plot[f\"{name}_flow\"] = q_true - q_pred

    if cfg.recon_model_path:
        normalizer = _load_normalizer_from_model(cfg.recon_model_path)
        if cfg.device == "cpu":
            device = torch.device("cpu")
        elif cfg.device == "mps":
            device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            device = get_device(prefer_mps=True)
        node_in_dim = data["node_static"].shape[1] + 2
        edge_in_dim = data["edge_static"].shape[1] + 2
        model = ReconGNN(
            node_in_dim,
            edge_in_dim,
            cfg.model.hidden_dim,
            cfg.model.num_layers,
            cfg.model.dropout,
        )
        model.load_state_dict(torch.load(cfg.recon_model_path, map_location=device))
        model.to(device)
        model.eval()

        p_true_all = []
        q_true_all = []
        p_pred_all = []
        q_pred_all = []
        p_mask_all = []
        q_mask_all = []
        for batch in test_loader:
            batch = batch_to_device(batch, device)
            for sample in batch:
                p_obs = sample["P_obs"].cpu().numpy()
                q_obs = sample["Q_obs"].cpu().numpy()
                if normalizer is not None:
                    p_obs = normalizer.transform_p(p_obs)
                    q_obs = normalizer.transform_q(q_obs)
                    p_obs[~sample["P_mask"].cpu().numpy()] = 0.0
                    q_obs[~sample["Q_mask"].cpu().numpy()] = 0.0
                p_hat, q_hat = model(
                    sample["edge_index"],
                    sample["node_static"],
                    sample["edge_static"],
                    torch.as_tensor(p_obs, device=device),
                    torch.as_tensor(q_obs, device=device),
                    sample["P_mask"],
                    sample["Q_mask"],
                )
                p_hat_np = p_hat.detach().cpu().numpy()
                q_hat_np = q_hat.detach().cpu().numpy()
                if normalizer is not None:
                    p_hat_np = normalizer.inverse_p(p_hat_np)
                    q_hat_np = normalizer.inverse_q(q_hat_np)
                p_true_all.append(sample["P_true"].cpu().numpy())
                q_true_all.append(sample["Q_true"].cpu().numpy())
                p_pred_all.append(p_hat_np)
                q_pred_all.append(q_hat_np)
                p_mask_all.append(sample["P_mask"].cpu().numpy())
                q_mask_all.append(sample["Q_mask"].cpu().numpy())

        p_true = np.concatenate(p_true_all)
        q_true = np.concatenate(q_true_all)
        p_pred = np.concatenate(p_pred_all)
        q_pred = np.concatenate(q_pred_all)
        metrics_out["recon_model"] = {
            "pressure": recon_metrics(p_true, p_pred),
            "flow": recon_metrics(q_true, q_pred),
            "pressure_by_missing": _stratified_recon_metrics(
                p_true_all, p_pred_all, p_mask_all, bins=[(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
            ),
            "flow_by_missing": _stratified_recon_metrics(
                q_true_all, q_pred_all, q_mask_all, bins=[(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
            ),
        }
        errors_for_plot["recon_pressure"] = p_true - p_pred
        errors_for_plot["recon_flow"] = q_true - q_pred

    if cfg.multitask_model_path:
        normalizer = _load_normalizer_from_model(cfg.multitask_model_path)
        if cfg.device == "cpu":
            device = torch.device("cpu")
        elif cfg.device == "mps":
            device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            device = get_device(prefer_mps=True)
        node_in_dim = data["node_static"].shape[1] + 2
        edge_in_dim = data["edge_static"].shape[1] + 2
        model = MultiTaskGNN(
            node_in_dim,
            edge_in_dim,
            cfg.model.hidden_dim,
            cfg.model.num_layers,
            cfg.model.dropout,
        )
        model.load_state_dict(torch.load(cfg.multitask_model_path, map_location=device))
        model.to(device)
        model.eval()

        p_true_all = []
        q_true_all = []
        p_pred_all = []
        q_pred_all = []
        p_mask_all = []
        q_mask_all = []
        p_label_all = []
        q_label_all = []
        p_score_all = []
        q_score_all = []
        for batch in test_loader:
            batch = batch_to_device(batch, device)
            for sample in batch:
                p_obs = sample["P_obs"].cpu().numpy()
                q_obs = sample["Q_obs"].cpu().numpy()
                if normalizer is not None:
                    p_obs = normalizer.transform_p(p_obs)
                    q_obs = normalizer.transform_q(q_obs)
                    p_obs[~sample["P_mask"].cpu().numpy()] = 0.0
                    q_obs[~sample["Q_mask"].cpu().numpy()] = 0.0
                p_hat, q_hat, p_logits, q_logits = model(
                    sample["edge_index"],
                    sample["node_static"],
                    sample["edge_static"],
                    torch.as_tensor(p_obs, device=device),
                    torch.as_tensor(q_obs, device=device),
                    sample["P_mask"],
                    sample["Q_mask"],
                )
                p_hat_np = p_hat.detach().cpu().numpy()
                q_hat_np = q_hat.detach().cpu().numpy()
                if normalizer is not None:
                    p_hat_np = normalizer.inverse_p(p_hat_np)
                    q_hat_np = normalizer.inverse_q(q_hat_np)
                p_true_all.append(sample["P_true"].cpu().numpy())
                q_true_all.append(sample["Q_true"].cpu().numpy())
                p_pred_all.append(p_hat_np)
                q_pred_all.append(q_hat_np)
                p_mask_all.append(sample["P_mask"].cpu().numpy())
                q_mask_all.append(sample["Q_mask"].cpu().numpy())
                p_label_all.append(sample["P_anom"].cpu().numpy())
                q_label_all.append(sample["Q_anom"].cpu().numpy())
                p_score_all.append(torch.sigmoid(p_logits).detach().cpu().numpy())
                q_score_all.append(torch.sigmoid(q_logits).detach().cpu().numpy())

        p_true = np.concatenate(p_true_all)
        q_true = np.concatenate(q_true_all)
        p_pred = np.concatenate(p_pred_all)
        q_pred = np.concatenate(q_pred_all)
        metrics_out["multitask_model"] = {
            "pressure": recon_metrics(p_true, p_pred),
            "flow": recon_metrics(q_true, q_pred),
            "pressure_by_missing": _stratified_recon_metrics(
                p_true_all, p_pred_all, p_mask_all, bins=[(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
            ),
            "flow_by_missing": _stratified_recon_metrics(
                q_true_all, q_pred_all, q_mask_all, bins=[(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
            ),
            "pressure_anom": classification_metrics(np.concatenate(p_label_all), np.concatenate(p_score_all)),
            "flow_anom": classification_metrics(np.concatenate(q_label_all), np.concatenate(q_score_all)),
        }
        errors_for_plot["multitask_pressure"] = p_true - p_pred
        errors_for_plot["multitask_flow"] = q_true - q_pred

    plot_recon_errors(errors_for_plot, run_info.run_dir / "figures")

    with open(run_info.run_dir / "metrics" / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    return run_info.run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    eval_models(args.config)


if __name__ == "__main__":
    main()

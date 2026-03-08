from __future__ import annotations

import argparse
import json
import shutil
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wdn.baselines import BaselineConfig, analytical_baseline, residual_anomaly_scores, wls_baseline
from wdn.config import load_eval_config, load_generate_config
from wdn.data_generation import generate_dataset
from wdn.dataset import split_indices
from wdn.metrics import classification_metrics, recon_metrics
from wdn.utils import make_run_dir, set_seed, setup_logging


logger = logging.getLogger("wdn.sweep")


def _baseline_metrics(data, splits, baseline_cfg: BaselineConfig, mad_scale: float):
    results = {}
    for name, recon_fn in {
        "baseline_analytical": analytical_baseline,
        "baseline_wls": wls_baseline,
    }.items():
        p_true_all = []
        p_pred_all = []
        q_true_all = []
        q_pred_all = []
        p_anom_all = []
        q_anom_all = []
        p_score_all = []
        q_score_all = []
        for idx in splits.test_idx:
            p_hat, q_hat = recon_fn(
                data["P_obs"][idx],
                data["Q_obs"][idx],
                data["P_mask"][idx],
                data["Q_mask"][idx],
                data["edge_index"],
                baseline_cfg,
            )
            p_true_all.append(data["P_true"][idx])
            q_true_all.append(data["Q_true"][idx])
            p_pred_all.append(p_hat)
            q_pred_all.append(q_hat)

            p_scores, _ = residual_anomaly_scores(
                data["P_obs"][idx], p_hat, data["P_mask"][idx], mad_scale=mad_scale
            )
            q_scores, _ = residual_anomaly_scores(
                data["Q_obs"][idx], q_hat, data["Q_mask"][idx], mad_scale=mad_scale
            )
            p_score_all.append(p_scores)
            q_score_all.append(q_scores)
            p_anom_all.append(data["P_anom"][idx])
            q_anom_all.append(data["Q_anom"][idx])

        p_true = np.concatenate(p_true_all)
        q_true = np.concatenate(q_true_all)
        p_pred = np.concatenate(p_pred_all)
        q_pred = np.concatenate(q_pred_all)
        results[name] = {
            "pressure": recon_metrics(p_true, p_pred),
            "flow": recon_metrics(q_true, q_pred),
            "pressure_anom": classification_metrics(np.concatenate(p_anom_all), np.concatenate(p_score_all)),
            "flow_anom": classification_metrics(np.concatenate(q_anom_all), np.concatenate(q_score_all)),
        }
    return results


def run_sweep(config_path: str) -> Path:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        sweep_cfg = yaml.safe_load(f)

    gen_cfg = load_generate_config(sweep_cfg["base_generate_config"])
    eval_cfg = load_eval_config(sweep_cfg["base_eval_config"])

    set_seed(gen_cfg.seed)
    run_info = make_run_dir(Path(sweep_cfg.get("run_dir", "runs")))
    shutil.copy(config_path, run_info.run_dir / "config_sweep.yaml")
    setup_logging(run_info.run_dir / "sweep.log")

    baseline_cfg = BaselineConfig(
        pinv_rcond=eval_cfg.baseline.get("pinv_rcond", 1e-4),
        wls_alpha=eval_cfg.baseline.get("wls_alpha", 1e-2),
        wls_beta=eval_cfg.baseline.get("wls_beta", 1e-2),
        diag_eps=eval_cfg.baseline.get("diag_eps", 1e-6),
    )
    mad_scale = eval_cfg.baseline.get("mad_scale", 3.5)

    results_rows = []

    for sweep_name, values in sweep_cfg.get("sweeps", {}).items():
        for value in values:
            if sweep_name == "missing_rates":
                gen_cfg.corruption.missing_p = value
                gen_cfg.corruption.missing_q = value
            elif sweep_name == "noise_sigmas":
                gen_cfg.corruption.noise_sigma_p = value
                gen_cfg.corruption.noise_sigma_q = value
            elif sweep_name == "attack_fractions":
                gen_cfg.corruption.attack_enabled = value > 0.0
                gen_cfg.corruption.attack_fraction = value
            elif sweep_name == "attack_biases":
                gen_cfg.corruption.attack_enabled = value > 0.0
                gen_cfg.corruption.attack_bias_p = value
                gen_cfg.corruption.attack_bias_q = value * 0.2

            data = generate_dataset(gen_cfg)
            splits = split_indices(data["P_true"].shape[0], eval_cfg.data.split, eval_cfg.seed)
            metrics = _baseline_metrics(data, splits, baseline_cfg, mad_scale)

            for model_name, metrics_dict in metrics.items():
                row = {
                    "sweep": sweep_name,
                    "value": value,
                    "model": model_name,
                    "pressure_mae": metrics_dict["pressure"]["mae"],
                    "pressure_mse": metrics_dict["pressure"]["mse"],
                    "flow_mae": metrics_dict["flow"]["mae"],
                    "flow_mse": metrics_dict["flow"]["mse"],
                    "pressure_f1": metrics_dict["pressure_anom"]["f1"],
                    "flow_f1": metrics_dict["flow_anom"]["f1"],
                }
                results_rows.append(row)

    df = pd.DataFrame(results_rows)
    csv_path = run_info.run_dir / "metrics" / "sweep_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    with open(run_info.run_dir / "metrics" / "sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_rows, f, indent=2)

    logger.info("Saved sweep results to %s", csv_path)
    return run_info.run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_sweep(args.config)


if __name__ == "__main__":
    main()

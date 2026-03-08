from __future__ import annotations

import argparse
import json
import shutil
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from wdn.config import load_multitask_train_config
from wdn.dataset import create_dataloaders, load_npz, split_indices
from wdn.metrics import classification_metrics, recon_metrics
from wdn.models.multitask import MultiTaskGNN
from wdn.normalization import compute_normalizer, save_normalizer
from wdn.train_utils import anomaly_loss, batch_to_device, recon_loss
from wdn.utils import get_device, make_run_dir, set_seed, setup_logging


logger = logging.getLogger("wdn.train_multitask")


def _eval_loop(model: nn.Module, loader, loss_on_all: bool, device: torch.device):
    model.eval()
    losses = []
    p_true_all = []
    p_pred_all = []
    q_true_all = []
    q_pred_all = []
    p_label_all = []
    q_label_all = []
    p_score_all = []
    q_score_all = []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            batch_loss = 0.0
            for sample in batch:
                p_hat, q_hat, p_logits, q_logits = model(
                    sample["edge_index"],
                    sample["node_static"],
                    sample["edge_static"],
                    sample["P_obs"],
                    sample["Q_obs"],
                    sample["P_mask"],
                    sample["Q_mask"],
                )
                r_loss = recon_loss(
                    p_hat,
                    q_hat,
                    sample["P_true"],
                    sample["Q_true"],
                    sample["P_mask"],
                    sample["Q_mask"],
                    loss_on_all,
                )
                a_loss = anomaly_loss(p_logits, q_logits, sample["P_anom"], sample["Q_anom"])
                loss = r_loss + a_loss
                batch_loss += loss.item()

                p_true_all.append(sample["P_true"].cpu().numpy())
                p_pred_all.append(p_hat.cpu().numpy())
                q_true_all.append(sample["Q_true"].cpu().numpy())
                q_pred_all.append(q_hat.cpu().numpy())
                p_label_all.append(sample["P_anom"].cpu().numpy())
                q_label_all.append(sample["Q_anom"].cpu().numpy())
                p_score_all.append(torch.sigmoid(p_logits).cpu().numpy())
                q_score_all.append(torch.sigmoid(q_logits).cpu().numpy())
            losses.append(batch_loss / max(1, len(batch)))

    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "pressure": recon_metrics(np.concatenate(p_true_all), np.concatenate(p_pred_all)),
        "flow": recon_metrics(np.concatenate(q_true_all), np.concatenate(q_pred_all)),
        "pressure_anom": classification_metrics(np.concatenate(p_label_all), np.concatenate(p_score_all)),
        "flow_anom": classification_metrics(np.concatenate(q_label_all), np.concatenate(q_score_all)),
    }
    return metrics


def train(config_path: str) -> Path:
    cfg = load_multitask_train_config(config_path)
    set_seed(cfg.seed)

    run_info = make_run_dir(Path(cfg.run_dir))
    setup_logging(run_info.run_dir / "train_multitask.log")
    logger.info("Run dir: %s", run_info.run_dir)

    shutil.copy(config_path, run_info.run_dir / "config_multitask.yaml")
    with open(run_info.run_dir / "config_multitask.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, default=lambda o: o.__dict__, indent=2)

    data = load_npz(cfg.data.dataset_path)
    splits = split_indices(data["P_true"].shape[0], cfg.data.split, cfg.seed)

    normalizer = compute_normalizer(data["P_true"][splits.train_idx], data["Q_true"][splits.train_idx])
    save_normalizer(normalizer, run_info.run_dir / "artifacts" / "normalizer.yaml")

    train_loader, val_loader, _ = create_dataloaders(
        data,
        splits,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        normalizer=normalizer,
    )

    node_in_dim = data["node_static"].shape[1] + 2
    edge_in_dim = data["edge_static"].shape[1] + 2

    if cfg.device == "cpu":
        device = torch.device("cpu")
    elif cfg.device == "mps":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        device = get_device(prefer_mps=True)
    model = MultiTaskGNN(node_in_dim, edge_in_dim, cfg.model.hidden_dim, cfg.model.num_layers, cfg.model.dropout)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    best_val = float("inf")
    for epoch in range(cfg.optim.epochs):
        model.train()
        epoch_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.optim.epochs}"):
            batch = batch_to_device(batch, device)
            optimizer.zero_grad()
            batch_loss = 0.0
            for sample in batch:
                p_hat, q_hat, p_logits, q_logits = model(
                    sample["edge_index"],
                    sample["node_static"],
                    sample["edge_static"],
                    sample["P_obs"],
                    sample["Q_obs"],
                    sample["P_mask"],
                    sample["Q_mask"],
                )
                r_loss = recon_loss(
                    p_hat,
                    q_hat,
                    sample["P_true"],
                    sample["Q_true"],
                    sample["P_mask"],
                    sample["Q_mask"],
                    cfg.recon_loss_on_all,
                )
                a_loss = anomaly_loss(p_logits, q_logits, sample["P_anom"], sample["Q_anom"])
                loss = r_loss + cfg.lambda_anom * a_loss
                batch_loss += loss
            batch_loss = batch_loss / max(1, len(batch))
            batch_loss.backward()
            optimizer.step()
            epoch_losses.append(batch_loss.item())

        val_metrics = _eval_loop(model, val_loader, cfg.recon_loss_on_all, device)
        logger.info("Epoch %d train_loss=%.4f val_loss=%.4f", epoch + 1, np.mean(epoch_losses), val_metrics["loss"])

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(model.state_dict(), run_info.run_dir / "models" / "multitask.pt")
            with open(run_info.run_dir / "metrics" / "val_multitask.json", "w", encoding="utf-8") as f:
                json.dump(val_metrics, f, indent=2)

    return run_info.run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()

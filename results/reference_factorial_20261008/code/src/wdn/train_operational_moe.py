"""Development training with separate calibration, validation and locked test."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from wdn.dataset import Normalizer
from wdn.models.mechanism_moe import FAMILY_NAMES, MechanismMoE
from wdn.models.temporal_moe import TemporalMixtureOfExpertsGNN, temporal_moe_loss
from wdn.models.specialist_loss import direct_specialist_loss
from wdn.temporal_dataset import TemporalWDNDataset, temporal_collate_fn
from wdn.train_temporal_moe import _to_device


@dataclass
class TrainingConfig:
    data_dir: str = "data/thesis_v2/operational_modena_seed811"
    output_dir: str = "runs/operational/pilot"
    model: str = "mechanism"
    seed: int = 1
    split_seed: int = 417
    epochs: int = 12
    patience: int = 8
    batch_size: int = 8
    threads: int = 2
    device: str = "cpu"
    endpoint_window: int = 16
    short_window: int = 4
    medium_window: int = 8
    long_window: int = 16
    hidden_dim: int = 32
    replay_hidden_dim: int = 32
    drift_hidden_dim: int = 32
    router_hidden_dim: int = 24
    num_layers: int = 2
    dropout: float = .1
    router_temperature: float = 1.
    lr: float = .001
    weight_decay: float = .0001
    lambda_anomaly: float = 1.
    lambda_expert: float = .5
    lambda_router: float = .1
    anomaly_pos_weight: float = 10.
    replay_weight: float = 2.
    balanced_training: bool = True
    evaluate_test: bool = False
    feature_version: int = 1
    expert_objective: str = "legacy"
    expert_warmup_epochs: int = 0
    hard_negative_fraction: float = .25


def target_score(overall_f1, replay_f1):
    return float(overall_f1 - max(0., .5-replay_f1))


def _binary_counts(scores, labels, threshold):
    pred, pos = scores > threshold, labels > .5
    tp = int((pred & pos).sum()); fp = int((pred & ~pos).sum())
    fn = int((~pred & pos).sum()); tn = int((~pred & ~pos).sum())
    return {"f1": 2*tp/max(2*tp+fp+fn, 1), "precision": tp/max(tp+fp, 1),
            "recall": tp/max(tp+fn, 1), "fpr": fp/max(fp+tn, 1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(scores)}


def select_threshold(scores, labels, families):
    """One global threshold on calibration ONLY; replay is a selection target."""
    replay = families == FAMILY_NAMES.index("replay")
    if not np.any(labels[replay] > .5):
        raise ValueError("Calibration requires observed replay positives")
    grid = np.unique(np.concatenate([np.linspace(.01, .99, 99),
                                     np.quantile(scores, np.linspace(0, 1, 101))]))
    best = None
    for threshold in grid:
        overall = _binary_counts(scores, labels, threshold)["f1"]
        replay_f1 = _binary_counts(scores[replay], labels[replay], threshold)["f1"]
        candidate = (target_score(overall, replay_f1), overall, float(threshold))
        if best is None or candidate > best:
            best = candidate
    return best[2]


def summarise(scores, labels, families, threshold):
    overall = _binary_counts(scores, labels, threshold)
    overall["auprc"] = float(average_precision_score(labels, scores)) if labels.any() else None
    overall["auroc"] = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None
    per_family = {}
    for cls, name in enumerate(FAMILY_NAMES):
        selected = families == cls
        if selected.any():
            per_family[name] = _binary_counts(scores[selected], labels[selected], threshold)
    replay = per_family.get("replay", {}).get("f1", 0.)
    return {"overall": overall, "per_family": per_family, "threshold": float(threshold),
            "target_score": target_score(overall["f1"], replay),
            "replay_target_met": replay > .5,
            "overall_gap_to_0_90": max(0., .9-overall["f1"]),
            "replay_gap_to_0_50": max(0., .5-replay)}


def make_splits(snapshots, corrupted, seed):
    """Deterministic label-stratified scenario split, independent of scores.

    Require all classes and observed replay positives in every split.
    Search only the fixed random partition, never model scores or thresholds.
    """
    ids = sorted({s.scenario_id for s in snapshots})
    if len(ids) < 12:
        raise ValueError("At least 12 scenarios required for four disjoint splits")
    classes = {sid: set() for sid in ids}
    replay_ids = set()
    for s, c in zip(snapshots, corrupted):
        classes[s.scenario_id].add(c.attack_type_id)
        if c.attack_type_id == 2 and c.pressure_anomaly.any():
            replay_ids.add(s.scenario_id)
    sizes = [int(.6*len(ids)), max(2, int(.15*len(ids))), max(2, int(.15*len(ids)))]
    rng = np.random.default_rng(seed)
    for attempt in range(10000):
        shuffled = rng.permutation(ids)
        partitions = np.split(shuffled, np.cumsum(sizes))
        sets = [set(p.tolist()) for p in partitions]
        if any(not s or not (s & replay_ids) for s in sets):
            continue
        if any(set.union(*(classes[sid] for sid in group)) != set(range(6)) for group in sets):
            continue
        return {name: sorted(s) for name, s in zip(("train", "calibration", "validation", "test"), sets)}
    raise ValueError("Insufficient family/replay coverage: generate more scenarios before training")


def _forward(model, batch):
    return model(**{name: batch[name] for name in (
        "x_seq", "edge_index", "edge_attr", "is_original_edge", "batch_size",
        "pressure_obs", "flow_obs", "pressure_mask", "flow_mask")},
        num_nodes_per_graph=batch["num_nodes"])


def build_model(cfg, sample):
    """One architecture factory for training and saved-checkpoint audits."""
    model_args = dict(node_in_dim=sample["x_seq"][0].shape[1],
        edge_in_dim=sample["edge_attr"].shape[1], hidden_dim=cfg.hidden_dim,
        router_hidden_dim=cfg.router_hidden_dim, dropout=cfg.dropout, num_layers=cfg.num_layers)
    if cfg.model in ("mechanism", "uniform"):
        return MechanismMoE(**model_args, short_window=cfg.short_window,
            medium_window=cfg.medium_window, long_window=cfg.long_window,
            replay_hidden_dim=cfg.replay_hidden_dim, drift_hidden_dim=cfg.drift_hidden_dim,
            router_temperature=cfg.router_temperature, uniform=cfg.model == "uniform",
            feature_version=cfg.feature_version)
    return TemporalMixtureOfExpertsGNN(**model_args, num_experts=1 if cfg.model == "single" else 6,
                                       window_size=cfg.long_window, reroute_alpha=0.)


def operational_loss(out, batch, cfg, expert_only=False):
    direct = None
    if cfg.expert_objective == "balanced":
        direct = direct_specialist_loss(out, batch, cfg.replay_weight, cfg.hard_negative_fraction)
        if expert_only:
            return cfg.lambda_anomaly*direct["anomaly"] + .2*direct["reconstruction"]
    targets = batch["attack_type"]
    if cfg.model == "single":
        base_batch = dict(batch, attack_type=torch.zeros_like(targets))
    else:
        base_batch = batch
    base = temporal_moe_loss(out, base_batch, lambda_anomaly=cfg.lambda_anomaly,
        lambda_router=0. if cfg.model in ("single", "uniform") else cfg.lambda_router,
        lambda_balance=0., lambda_expert=0., anomaly_pos_weight=cfg.anomaly_pos_weight,
        replay_weight=1., replay_class_id=None)
    # Recompute pressure BCE with canonical labels so the single-expert control
    # receives exactly the same replay weighting as the mixtures.
    observed = batch["pressure_mask"] > 0
    families = targets.repeat_interleave(batch["num_nodes"])
    labels = batch["pressure_anomaly"]
    extra = out["pressure_pred"].new_zeros(())
    if observed.any():
        logits = out["pressure_anomaly_logits"][observed]
        y = labels[observed]
        weights = 1 + (cfg.replay_weight-1.) * ((families[observed] == 2) & (y > 0)).float()
        pos_weight = logits.new_tensor(cfg.anomaly_pos_weight)
        losses = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
        extra = cfg.lambda_anomaly * ((weights-1.)*losses).mean()
        if cfg.model != "single" and direct is None:
            expert_logits = out["expert_pressure_anomaly_logits"][observed]
            owners = F.one_hot(families[observed], 6).to(logits.dtype)
            owners[:, 0] = 1.  # General expert learns every family.
            owners[families[observed] == 0] = 1.  # All experts learn clean data.
            legacy_direct = F.binary_cross_entropy_with_logits(expert_logits,
                y[:, None].expand_as(expert_logits), pos_weight=pos_weight, reduction="none")
            legacy_direct = (legacy_direct * owners * weights[:, None]).sum() / owners.sum().clamp(min=1)
            extra = extra + cfg.lambda_expert*legacy_direct
    # No zero-demand conservation penalty on normalised flows: it is not a
    # physically valid continuity equation. Reconstruction remains supervised.
    if direct is not None:
        extra = extra + cfg.lambda_expert*(direct["anomaly"] + .2*direct["reconstruction"])
    return base["total_loss"] + extra


@torch.no_grad()
def collect(model, loader, device, normalizer, window):
    model.eval()
    scores, labels, families = [], [], []
    error_sum, missing_count = 0., 0
    for raw in loader:
        batch = _to_device(raw, device)
        batch["x_seq"] = batch["x_seq"][-window:]
        out = _forward(model, batch)
        observed = batch["pressure_mask"] > 0
        scores.append(out["pressure_anomaly_logits"].sigmoid()[observed].cpu().numpy())
        labels.append(batch["pressure_anomaly"][observed].cpu().numpy())
        family = batch["attack_type"].repeat_interleave(batch["num_nodes"])
        families.append(family[observed].cpu().numpy())
        pred = normalizer.denormalize_pressure(out["pressure_pred"].cpu())
        truth = normalizer.denormalize_pressure(batch["y_pressure"].cpu())
        missing = ~observed.cpu()
        error_sum += float((pred-truth).abs()[missing].sum())
        missing_count += int(missing.sum())
    return tuple(np.concatenate(x) for x in (scores, labels, families)), error_sum/max(missing_count, 1)


def train(cfg: TrainingConfig, trial=None):
    if cfg.model not in ("mechanism", "homogeneous", "single", "uniform"):
        raise ValueError("Unknown model")
    if not 1 <= cfg.short_window <= cfg.medium_window <= cfg.long_window <= cfg.endpoint_window:
        raise ValueError("Expert windows must be ordered and <= the common endpoint window")
    if cfg.epochs < 1:
        raise ValueError("epochs must be positive")
    if cfg.expert_objective not in ("legacy", "balanced"):
        raise ValueError("Unknown expert objective")
    if not 0 <= cfg.expert_warmup_epochs < cfg.epochs:
        raise ValueError("Warmup must leave at least one joint-training epoch")
    if cfg.expert_warmup_epochs and (cfg.expert_objective != "balanced" or cfg.model == "single"):
        raise ValueError("Expert warmup requires a multi-expert model and balanced objective")
    if trial is not None and cfg.evaluate_test:
        raise ValueError("Optuna trials must never evaluate test")
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    device = torch.device(cfg.device)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "args.json").write_text(json.dumps(asdict(cfg), indent=2))
    source_sha256 = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                    for name in ("train_operational_moe.py", "models/mechanism_moe.py", "models/specialist_loss.py",
                                 "models/temporal_moe.py", "models/temporal_multitask.py", "temporal_dataset.py")}
    (output / "source_sha256.json").write_text(json.dumps(source_sha256, indent=2))
    data = Path(cfg.data_dir)
    manifest = json.loads((data / "manifest.json").read_text())
    import yaml
    dataset_config = yaml.safe_load((data / "generate_config.yaml").read_text())
    if any(dataset_config.get(f"missing_rate_{c}") != .5 for c in ("pressure", "flow")):
        raise ValueError("This runner requires the fixed 50% missing protocol")
    with (data / "snapshots.pkl").open("rb") as f: snapshots = pickle.load(f)
    with (data / "corrupted.pkl").open("rb") as f: corrupted = pickle.load(f)
    splits = make_splits(snapshots, corrupted, cfg.split_seed)
    (output / "splits.json").write_text(json.dumps(splits, indent=2))
    normalizer = Normalizer(mode="per_node")
    normalizer.fit([s for s in snapshots if s.scenario_id in splits["train"]])
    datasets, loaders = {}, {}
    for name, ids in splits.items():
        # Construct no test loader until an explicit final evaluation request.
        if name == "test" and not cfg.evaluate_test:
            continue
        indices = [i for i, s in enumerate(snapshots) if s.scenario_id in ids]
        ds = TemporalWDNDataset([snapshots[i] for i in indices], [corrupted[i] for i in indices],
                                cfg.endpoint_window, normalizer)
        datasets[name] = ds
        endpoint_labels = np.array([ds.corrupted[w[-1]].attack_type_id for w in ds.windows])
        if name in ("calibration", "validation"):
            if not any(ds.corrupted[w[-1]].attack_type_id == 2 and
                       ds.corrupted[w[-1]].pressure_anomaly.any() for w in ds.windows):
                raise ValueError(f"{name} has no observed replay positives on common endpoints")
        sampler = None
        if name == "train" and cfg.balanced_training:
            counts = np.bincount(endpoint_labels, minlength=6)
            sampler = WeightedRandomSampler(1./np.maximum(counts[endpoint_labels], 1),
                                             len(ds), replacement=True,
                                             generator=torch.Generator().manual_seed(cfg.seed))
        loaders[name] = DataLoader(ds, batch_size=cfg.batch_size,
            sampler=sampler, shuffle=name == "train" and sampler is None,
            collate_fn=temporal_collate_fn, num_workers=0)
    sample = datasets["train"][0]
    model = build_model(cfg, sample)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, history, stale = None, [], 0
    started = time.monotonic()
    for epoch in range(1, cfg.epochs+1):
        epoch_start = time.monotonic()
        model.train()
        warmup = epoch <= cfg.expert_warmup_epochs
        loss_sum = 0.
        for raw in loaders["train"]:
            batch = _to_device(raw, device)
            # Same endpoints for all candidates; only visible history differs.
            batch["x_seq"] = batch["x_seq"][-cfg.long_window:]
            out = _forward(model, batch)
            loss = operational_loss(out, batch, cfg, expert_only=warmup)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            loss_sum += float(loss.detach())
        cal_arrays, cal_mae = collect(model, loaders["calibration"], device, normalizer, cfg.long_window)
        threshold = select_threshold(*cal_arrays)
        val_arrays, val_mae = collect(model, loaders["validation"], device, normalizer, cfg.long_window)
        validation = summarise(*val_arrays, threshold)
        entry = {"epoch": epoch, "stage": "expert_warmup" if warmup else "joint", "loss": loss_sum/len(loaders["train"]),
                 "seconds": time.monotonic()-epoch_start, "validation": validation,
                 "pressure_mae_missing_m": val_mae}
        history.append(entry)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(f"epoch={epoch} stage={entry['stage']} F1={validation['overall']['f1']:.4f} "
              f"replay={validation['per_family']['replay']['f1']:.4f} "
              f"score={validation['target_score']:.4f} seconds={entry['seconds']:.1f}", flush=True)
        if warmup:
            continue  # An untrained router is not a checkpoint-selection candidate.
        if best is None or validation["target_score"] > best["validation"]["target_score"]:
            best = entry
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "threshold": threshold, "config": asdict(cfg), "epoch": epoch}, output / "best.pt")
            stale = 0
        else:
            stale += 1
        if trial is not None:
            trial.report(best["validation"]["target_score"], epoch)
            trial.set_user_attr("overall_f1", best["validation"]["overall"]["f1"])
            trial.set_user_attr("replay_f1", best["validation"]["per_family"]["replay"]["f1"])
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        if stale >= cfg.patience:
            break
    with (output / "normalizer.pkl").open("wb") as f: pickle.dump(normalizer, f)
    summary = {"status": "development_only", "best_epoch": best["epoch"],
               "validation": best["validation"], "pressure_mae_missing_m": best["pressure_mae_missing_m"],
               "seconds": time.monotonic()-started,
               "n_params": sum(p.numel() for p in model.parameters()),
               "dataset_config_sha256": manifest["config_sha256"],
               "source_sha256": source_sha256,
               "test_evaluated": False}
    if cfg.evaluate_test:
        checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        arrays, mae = collect(model, loaders["test"], device, normalizer, cfg.long_window)
        summary["test"] = summarise(*arrays, checkpoint["threshold"])
        summary["test_pressure_mae_missing_m"] = mae
        summary["test_evaluated"] = True
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON TrainingConfig overrides")
    parser.add_argument("--data_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--model", choices=["mechanism", "homogeneous", "single", "uniform"])
    parser.add_argument("--evaluate_test", action="store_true")
    args = parser.parse_args()
    overrides = json.loads(Path(args.config).read_text()) if args.config else {}
    overrides.update({k: v for k, v in vars(args).items() if k != "config" and v is not None
                      and k != "evaluate_test"})
    overrides["evaluate_test"] = args.evaluate_test
    print(json.dumps(train(TrainingConfig(**overrides)), indent=2))


if __name__ == "__main__":
    main()

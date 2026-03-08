"""Configuration dataclasses for the WDN pipeline.

All configs are plain dataclasses that can be loaded from YAML files.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

@dataclass
class CorruptionConfig:
    """Controls how observations are corrupted (missing data, noise, attacks)."""

    # Missing data: probability of each sensor being unavailable
    missing_rate_pressure: float = 0.3
    missing_rate_flow: float = 0.3

    # Measurement noise (Gaussian, added to observed values only)
    noise_sigma_pressure: float = 0.5
    noise_sigma_flow: float = 0.2

    # Attacks (Phase 5 — disabled by default)
    attack_enabled: bool = False
    attack_fraction: float = 0.1       # fraction of sensors to attack
    attack_bias: float = 2.0           # additive bias on attacked sensors
    attack_scale: float = 1.0          # multiplicative scale on attacked sensors
    attack_type: str = "random"        # "random" or "targeted"


@dataclass
class GenerateConfig:
    """Top-level config for data generation."""

    # Network
    network_inp: str = "data/Net1.inp"

    # Simulation time range (hours)
    duration_hours: int = 24
    hydraulic_timestep_minutes: int = 60  # 1 snapshot per hour

    # How many independent simulation scenarios to generate
    # (each with different random demand multipliers for variety)
    num_scenarios: int = 50

    # Demand variation: multiply base demands by uniform[1-var, 1+var]
    demand_variation: float = 0.2

    # Corruption
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)

    # Output
    output_dir: str = "data/generated"

    # Reproducibility
    seed: int = 42


# ---------------------------------------------------------------------------
# Training (will be extended in Phase 4+)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """GNN architecture hyperparameters."""

    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    gnn_type: str = "GAT"              # "GAT", "GraphSAGE", "GCN"
    heads: int = 4                     # attention heads (GAT only)


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Data
    batch_size: int = 8
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    num_workers: int = 0

    # Model
    model: ModelConfig = field(default_factory=ModelConfig)

    # Optimization
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Physics-informed loss weight (0 = disabled)
    lambda_physics: float = 0.1

    # Anomaly detection loss weight (0 = reconstruction only)
    lambda_anomaly: float = 1.0

    # Loss: compute on all nodes or only unobserved?
    loss_on_all: bool = True

    # Run management
    output_dir: str = "runs"
    seed: int = 42


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------

def _merge_into_dataclass(dc_class, overrides: dict):
    """Recursively merge a dict of overrides into a dataclass."""
    kwargs = {}
    for f in fields(dc_class):
        if f.name in overrides:
            val = overrides[f.name]
            # If the field is itself a dataclass, recurse
            if hasattr(f.type, "__dataclass_fields__") or (
                isinstance(f.default_factory, type)
                and hasattr(f.default_factory, "__dataclass_fields__")
            ):
                # Resolve the actual dataclass type
                inner_cls = f.type if hasattr(f.type, "__dataclass_fields__") else f.default_factory
                kwargs[f.name] = _merge_into_dataclass(inner_cls, val)
            else:
                kwargs[f.name] = val
    return dc_class(**kwargs)


def load_config(path: str | Path, config_class=GenerateConfig):
    """Load a YAML config file into a dataclass, using defaults for missing fields."""
    path = Path(path)
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return _merge_into_dataclass(config_class, raw)
    return config_class()


def save_config(cfg, path: str | Path):
    """Save a dataclass config to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False, sort_keys=False)

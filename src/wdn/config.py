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

    # Adversarial attacks
    attack_enabled: bool = False
    attack_fraction: float = 0.1       # fraction of observed sensors to attack
    attack_bias: float = 2.0           # additive bias / max drift magnitude
    attack_scale: float = 1.0          # multiplicative scale (random falsification)
    # Strength of the noise-injection attack, as a multiplier on the
    # reading's own magnitude. Exposed because it is the one attack whose
    # size does not follow attack_bias/attack_scale, so placing a second
    # network at the same point on the regime axis needs it separately.
    # The default reproduces the previous hard-coded value exactly.
    attack_noise_multiplier: float = 5.0
    # Attack episodes. With the default of 0 an attack family, its target
    # sensors and its parameters are redrawn at every snapshot, which
    # means a detector window spans several unrelated attacks and no
    # attack persists on a sensor -- measured on data/v2_modena, no window
    # holds a single family and consecutive snapshots share 8% of their
    # attacked sensors. That makes any temporal claim untestable: a "slow
    # drift" never drifts anywhere.
    #
    # Set these to hold one family and one compromised sensor set for a
    # run of snapshots, so a window can sit inside a single episode. The
    # compromised set is fixed for the episode and intersected with
    # whichever sensors happen to report, which is what a real compromise
    # looks like.
    attack_episode_min: int = 0        # 0 disables episodes entirely
    attack_episode_max: int = 0
    # Draw episode families from shuffled complete cycles instead of sampling
    # independently. This keeps family coverage balanced and, because the
    # schedule has its own RNG stream, identical across different networks
    # generated with the same seed.
    balanced_episode_families: bool = False
    attack_type: str = "random"        # "random", "replay", "stealthy", "noise", "targeted", "mixed"

    # Which attacks "mixed" draws from. None = all five. Restricting the
    # pool generates a dataset that covers only part of the threat model
    # (e.g. dropping "replay" for a four-attack study).
    attack_pool: Optional[list] = None

    # What a replay attacker re-broadcasts. False (default) replays the
    # clean simulated value, which leaves the copy conspicuously free of
    # observation noise. True replays what the sensor actually *reported*
    # at that earlier step, noise included -- what an attacker who taps the
    # telemetry stream would really have on hand. The second is strictly
    # harder and is the honest threat model.
    replay_records_observation: bool = False

    # Replay lag is sampled once per coherent episode and then held fixed.
    # When episodes are disabled it may be redrawn per snapshot, preserving
    # the IID negative-control regime. A replay is not applied or labelled
    # until the requested amount of history exists.
    replay_lag_min: int = 3
    replay_lag_max: int = 6

    # Number of episode steps required for a stealthy drift to reach
    # ``attack_bias``. Its direction is sampled once per compromised sensor
    # when an episode starts and remains fixed throughout that episode.
    stealthy_ramp_steps: int = 20


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

    # Demand variation: multiply base demands by uniform[1-var, 1+var].
    # This varies demand *between* scenarios but is constant within one.
    demand_variation: float = 0.2

    # Amplitude of a 24 h diurnal demand cycle applied *within* a scenario.
    # 0 leaves demand constant over time (many .inp files, Modena included,
    # define no pattern at all, which makes the hydraulics steady-state);
    # 1 applies the full municipal cycle. Values in between scale it, which
    # is how we sweep signal variability while holding the network fixed.
    demand_pattern_amplitude: float = 0.0

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
    gnn_type: str = "GAT"              # "GAT", "GATv2", "Transformer", "GPS", "GraphSAGE", "GCN"
    heads: int = 4                     # attention heads (GAT/Transformer/GPS)

    # Uncertainty quantification (MC Dropout)
    mc_dropout_samples: int = 30       # number of forward passes for uncertainty


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

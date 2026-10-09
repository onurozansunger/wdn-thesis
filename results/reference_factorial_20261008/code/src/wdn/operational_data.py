"""Unit-explicit development scenarios; the legacy corruption path is unchanged."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle

import numpy as np
import torch
import yaml

from wdn.config import CorruptionConfig, GenerateConfig
from wdn.corruption import ATTACK_TYPE_TO_ID, corrupt_snapshot
from wdn.data_generation import generate_dataset


@dataclass
class OperationalConfig:
    profile: str = "operational_v1_development"
    network_inp: str = "data/modena.inp"
    output_dir: str = "data/thesis_v2/operational_modena_seed811"
    seed: int = 811
    num_scenarios: int = 24
    duration_hours: int = 168
    timestep_minutes: int = 60
    missing_rate_pressure: float = 0.5
    missing_rate_flow: float = 0.5
    pressure_noise_sigma_m: float = 0.1
    flow_noise_sigma_m3s: float = 0.0001
    attack_fraction: float = 0.05
    clean_gap_hours: tuple[float, float] = (48, 96)
    attack_duration_hours: tuple[float, float] = (6, 18)
    pressure_bias_m: tuple[float, float] = (0.5, 2.0)
    flow_bias_m3s: tuple[float, float] = (0.0005, 0.002)
    drift_ramp_hours: tuple[float, float] = (6, 18)
    replay_lag_hours: tuple[float, float] = (2, 6)
    injected_noise_factor: tuple[float, float] = (3, 8)
    demand_variation: float = 0.2
    demand_pattern_amplitude: float = 1.0

    def validate(self):
        if self.missing_rate_pressure != 0.5 or self.missing_rate_flow != 0.5:
            raise ValueError("Operational protocol fixes BOTH missing probabilities at 0.50")
        # The inherited 24-point demand cycle assumes hourly reporting.
        if self.timestep_minutes != 60:
            raise ValueError("This profile uses hourly sampling; other rates need a resampled demand cycle")
        if not 0 < self.attack_fraction <= 1:
            raise ValueError("attack_fraction must be in (0, 1]")
        if self.num_scenarios < 1 or self.duration_hours < 1:
            raise ValueError("Positive scenario count and duration required")
        for name in ("pressure_noise_sigma_m", "flow_noise_sigma_m3s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"Invalid {name}")
        for name in ("clean_gap_hours", "attack_duration_hours", "pressure_bias_m",
                     "flow_bias_m3s", "drift_ramp_hours", "replay_lag_hours",
                     "injected_noise_factor"):
            lo, hi = getattr(self, name)
            if not 0 < lo <= hi or not math.isfinite(hi):
                raise ValueError(f"Invalid positive range: {name}")


def _steps(hours, cfg, rng):
    lo, hi = hours
    lower = max(1, math.ceil(lo * 60 / cfg.timestep_minutes))
    upper = max(lower, math.floor(hi * 60 / cfg.timestep_minutes))
    return int(rng.integers(lower, upper + 1))


def corrupt_operational(snapshots, cfg: OperationalConfig, graph=None):
    """Return corrupted observations and an auditable event/source ledger.

    Missing masks and baseline noise have an independent random stream and
    are generated before any attack. Replay copies the exact pre-tampering
    reading available in that historical stream, and only when it was observed.
    It never reads clean hydraulic targets as replay payloads.
    """
    cfg.validate()
    observation_rng = np.random.default_rng(cfg.seed)
    schedule_rng = np.random.default_rng(cfg.seed + 1_000_003)
    attack_rng = np.random.default_rng(cfg.seed + 2_000_003)
    base_cfg = CorruptionConfig(
        missing_rate_pressure=0.5, missing_rate_flow=0.5,
        noise_sigma_pressure=cfg.pressure_noise_sigma_m,
        noise_sigma_flow=cfg.flow_noise_sigma_m3s, attack_enabled=False,
    )
    base = [corrupt_snapshot(s.pressure_true, s.flow_true, base_cfg, observation_rng)
            for s in snapshots]
    groups = {}
    for i, snap in enumerate(snapshots):
        groups.setdefault(snap.scenario_id, []).append(i)
    family_queue = []
    ledger = []
    result = list(base)
    degree = None
    if graph is not None:
        degree = np.bincount(np.asarray(graph.edge_index).reshape(-1), minlength=graph.num_nodes)

    for sid, indices in sorted(groups.items()):
        indices.sort(key=lambda i: snapshots[i].timestep)
        cursor = _steps(cfg.clean_gap_hours, cfg, schedule_rng)
        while cursor < len(indices):
            if not family_queue:
                family_queue = ["random", "replay", "stealthy", "noise", "targeted"]
                schedule_rng.shuffle(family_queue)
            family = family_queue.pop()
            length = _steps(cfg.attack_duration_hours, cfg, schedule_rng)
            lag = _steps(cfg.replay_lag_hours, cfg, schedule_rng)
            ramp = _steps(cfg.drift_ramp_hours, cfg, schedule_rng)
            first = base[indices[cursor]]
            targets, signs, magnitudes = {}, {}, {}
            for channel, obs, bounds in (
                ("pressure", first.pressure_obs, cfg.pressure_bias_m),
                ("flow", first.flow_obs, cfg.flow_bias_m3s),
            ):
                count = max(1, round(obs.numel() * cfg.attack_fraction))
                pool = np.arange(obs.numel())
                if family == "targeted" and degree is not None:
                    priority = degree if channel == "pressure" else degree[np.asarray(graph.edge_index)].sum(axis=0)
                    pool = np.argsort(priority)[-max(count, math.ceil(len(pool) * 0.3)):]
                selected = np.sort(attack_rng.choice(pool, size=min(count, len(pool)), replace=False))
                targets[channel] = selected
                signs[channel] = torch.tensor(attack_rng.choice([-1., 1.], len(selected)), dtype=torch.float32)
                magnitudes[channel] = torch.tensor(attack_rng.uniform(*bounds, len(selected)), dtype=torch.float32)
            noise_factor = float(attack_rng.uniform(*cfg.injected_noise_factor))
            event = {"scenario_id": int(sid), "family": family,
                     "start_timestep": int(snapshots[indices[cursor]].timestep),
                     "requested_steps": length, "actual_steps": min(length, len(indices)-cursor),
                     "lag_steps": lag, "ramp_steps": ramp,
                     "targets": {k: v.tolist() for k, v in targets.items()},
                     "magnitudes": {k: v.tolist() for k, v in magnitudes.items()},
                     "signs": {k: v.tolist() for k, v in signs.items()},
                     "noise_factor": noise_factor, "replay_sources": []}
            for age, local in enumerate(range(cursor, min(cursor + length, len(indices)))):
                i = indices[local]
                original = base[i]
                # Clone all fields so historical records remain immutable.
                c = type(original)(**{k: v.clone() if torch.is_tensor(v) else v
                                      for k, v in vars(original).items()})
                for channel, sigma in (("pressure", cfg.pressure_noise_sigma_m),
                                       ("flow", cfg.flow_noise_sigma_m3s)):
                    obs = getattr(c, f"{channel}_obs")
                    mask = getattr(c, f"{channel}_mask")
                    selected = torch.as_tensor(targets[channel], dtype=torch.long)
                    eligible = mask[selected] > 0
                    source = None
                    if family == "replay":
                        if local < lag:
                            continue
                        source = base[indices[local - lag]]
                        eligible &= getattr(source, f"{channel}_mask")[selected] > 0
                    active = selected[eligible]
                    if not active.numel():
                        continue
                    if family == "replay":
                        obs[active] = getattr(source, f"{channel}_obs")[active]
                        event["replay_sources"].append({
                            "channel": channel, "target_index": i,
                            "source_index": indices[local-lag], "sensors": active.tolist(),
                        })
                    elif family == "noise":
                        obs[active] += torch.tensor(attack_rng.normal(0, sigma * noise_factor, len(active)), dtype=obs.dtype)
                    else:
                        scale = min((age + 1) / ramp, 1.) if family == "stealthy" else 1.
                        obs[active] += signs[channel][eligible] * magnitudes[channel][eligible] * scale
                    # Numerical no-ops are never labelled positive.
                    changed = obs[active] != getattr(original, f"{channel}_obs")[active]
                    getattr(c, f"{channel}_anomaly")[active[changed]] = 1.
                if c.pressure_anomaly.any() or c.flow_anomaly.any():
                    c.attack_type_id = ATTACK_TYPE_TO_ID[family]
                result[i] = c
            ledger.append(event)
            cursor += length + _steps(cfg.clean_gap_hours, cfg, schedule_rng)
    return result, ledger


def dataset_manifest(snapshots, corrupted, cfg, ledger):
    result = {"profile": cfg.profile, "status": "provisional_development_simulation",
              "snapshots": len(snapshots), "events": len(ledger),
              "config_sha256": hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()}
    result["family_endpoints"] = {name: sum(c.attack_type_id == cls for c in corrupted)
                                  for name, cls in ATTACK_TYPE_TO_ID.items()}
    for channel, unit_scale in (("pressure", 1.), ("flow", 1000.)):
        masks = torch.cat([getattr(c, f"{channel}_mask") for c in corrupted]).bool()
        labels = torch.cat([getattr(c, f"{channel}_anomaly") for c in corrupted]).bool()
        if (labels & ~masks).any():
            raise ValueError("Attack label on missing observation")
        obs = torch.cat([getattr(c, f"{channel}_obs") for c in corrupted])
        truth = torch.cat([getattr(s, f"{channel}_true") for s in snapshots])
        if not torch.isfinite(obs).all() or not torch.isfinite(truth).all():
            raise ValueError("Nonfinite hydraulic or observed values")
        result[channel] = {
            "unit": "m" if channel == "pressure" else "L/s",
            "missing_rate": float((~masks).float().mean()),
            "attacked_observed_fraction": float(labels.sum() / masks.sum().clamp(min=1)),
            "displacement_quantiles": torch.quantile((obs-truth).abs()[labels]*unit_scale,
                torch.tensor([.1, .5, .9])).tolist() if labels.any() else [],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = OperationalConfig(**yaml.safe_load(Path(args.config).read_text()))
    cfg.validate()
    out = Path(cfg.output_dir)
    # Never overwrite a dataset used by earlier runs.
    out.mkdir(parents=True, exist_ok=False)
    simulation = GenerateConfig(network_inp=cfg.network_inp, duration_hours=cfg.duration_hours,
        hydraulic_timestep_minutes=cfg.timestep_minutes, num_scenarios=cfg.num_scenarios,
        seed=cfg.seed, demand_variation=cfg.demand_variation,
        demand_pattern_amplitude=cfg.demand_pattern_amplitude)
    graph, snapshots = generate_dataset(simulation)
    corrupted, ledger = corrupt_operational(snapshots, cfg, graph)
    manifest = dataset_manifest(snapshots, corrupted, cfg, ledger)
    for name, value in (("graph", graph), ("snapshots", snapshots), ("corrupted", corrupted)):
        with (out / f"{name}.pkl").open("wb") as handle:
            pickle.dump(value, handle)
    (out / "generate_config.yaml").write_text(yaml.safe_dump(asdict(cfg)))
    (out / "events.json").write_text(json.dumps(ledger, indent=2))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Derive a marginally matched IID control from a coherent-episode dataset.

For every scenario, the corruption residual/mask/label tuple is permuted over
time and re-attached to the clean hydraulic value at the receiving timestep.
The operation preserves the exact empirical marginal distribution of attack
families, attacked sensors, missingness and physical displacement while
destroying the chronological episode structure that a GRU could exploit.

This is a cleaner negative control than regenerating with episode length zero:
the latter changes the stealthy-ramp age distribution and therefore confounds
temporal coherence with attack severity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml


V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derangement(length: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform rejection sample with no corruption left at its original time."""
    original = np.arange(length)
    for _ in range(10_000):
        candidate = rng.permutation(length)
        if np.all(candidate != original):
            return candidate
    raise RuntimeError(f"failed to draw a derangement of length {length}")


def derive(source: Path, output: Path, permutation_seed: int) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)

    with (source / "snapshots.pkl").open("rb") as stream:
        snapshots = pickle.load(stream)
    with (source / "corrupted.pkl").open("rb") as stream:
        corrupted = pickle.load(stream)
    if len(snapshots) != len(corrupted):
        raise ValueError("snapshot/corruption length mismatch")

    by_scenario: dict[int, list[int]] = defaultdict(list)
    for index, snap in enumerate(snapshots):
        by_scenario[int(snap.scenario_id)].append(index)
    for indices in by_scenario.values():
        indices.sort(key=lambda index: int(snapshots[index].timestep))

    rng = np.random.default_rng(permutation_seed)
    iid = [None] * len(corrupted)
    permutations = {}
    source_adjacency = iid_adjacency = adjacent_pairs = 0
    for scenario, indices in sorted(by_scenario.items()):
        order = derangement(len(indices), rng)
        donors = [indices[int(position)] for position in order]
        permutations[str(scenario)] = [
            int(snapshots[index].timestep) for index in donors
        ]
        for recipient_index, donor_index in zip(indices, donors):
            recipient = snapshots[recipient_index]
            donor_snap = snapshots[donor_index]
            donor = corrupted[donor_index]

            p_mask = donor.pressure_mask.clone()
            q_mask = donor.flow_mask.clone()
            p_residual = donor.pressure_obs - donor_snap.pressure_true * p_mask
            q_residual = donor.flow_obs - donor_snap.flow_true * q_mask
            p_obs = (recipient.pressure_true + p_residual) * p_mask
            q_obs = (recipient.flow_true + q_residual) * q_mask
            iid[recipient_index] = type(donor)(
                pressure_obs=p_obs,
                flow_obs=q_obs,
                pressure_mask=p_mask,
                flow_mask=q_mask,
                pressure_anomaly=donor.pressure_anomaly.clone(),
                flow_anomaly=donor.flow_anomaly.clone(),
                attack_type_id=int(getattr(donor, "attack_type_id", 0)),
            )

        for left, right in zip(indices[:-1], indices[1:]):
            adjacent_pairs += 1
            source_adjacency += int(
                getattr(corrupted[left], "attack_type_id", 0)
                == getattr(corrupted[right], "attack_type_id", 0)
            )
            iid_adjacency += int(
                getattr(iid[left], "attack_type_id", 0)
                == getattr(iid[right], "attack_type_id", 0)
            )

    if any(item is None for item in iid):
        raise RuntimeError("not every corruption received a permuted donor")

    shutil.copy2(source / "graph.pkl", output / "graph.pkl")
    shutil.copy2(source / "snapshots.pkl", output / "snapshots.pkl")
    with (output / "corrupted.pkl").open("wb") as stream:
        pickle.dump(iid, stream)

    source_config = yaml.safe_load((source / "generate_config.yaml").read_text())
    output_config = dict(source_config)
    output_config["output_dir"] = str(output.relative_to(ROOT))
    output_config["derived_control"] = {
        "type": "within_scenario_temporal_permutation",
        "source_dataset": str(source.relative_to(ROOT)),
        "permutation_seed": permutation_seed,
        "preserves": [
            "clean hydraulics",
            "corruption residuals",
            "missingness masks",
            "sensor attack labels",
            "attack-family counts",
            "standardised displacement distribution",
        ],
    }
    (output / "generate_config.yaml").write_text(
        yaml.safe_dump(output_config, sort_keys=False)
    )

    audit = {
        "source_dataset": str(source.relative_to(ROOT)),
        "output_dataset": str(output.relative_to(ROOT)),
        "permutation_seed": permutation_seed,
        "scenarios": len(by_scenario),
        "snapshots": len(snapshots),
        "fixed_points": 0,
        "same_family_adjacent_fraction_source": source_adjacency / adjacent_pairs,
        "same_family_adjacent_fraction_iid": iid_adjacency / adjacent_pairs,
        "permutations_by_scenario": permutations,
    }
    (output / "iid_control_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    # Exact marginal-preservation assertions.
    source_families = sorted(int(getattr(item, "attack_type_id", 0)) for item in corrupted)
    iid_families = sorted(int(getattr(item, "attack_type_id", 0)) for item in iid)
    if source_families != iid_families:
        raise AssertionError("family marginals changed")
    source_labels = sum(int(item.pressure_anomaly.sum()) for item in corrupted)
    iid_labels = sum(int(item.pressure_anomaly.sum()) for item in iid)
    if source_labels != iid_labels:
        raise AssertionError("attack-label marginals changed")
    source_masks = sum(int(item.pressure_mask.sum()) for item in corrupted)
    iid_masks = sum(int(item.pressure_mask.sum()) for item in iid)
    if source_masks != iid_masks:
        raise AssertionError("missingness marginals changed")

    audit["files"] = {
        name: sha256(output / name)
        for name in ("graph.pkl", "snapshots.pkl", "corrupted.pkl", "generate_config.yaml")
    }
    (output / "iid_control_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path,
        default=ROOT / "data" / "thesis_v2" / "modena_episode_seed101",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data" / "thesis_v2" / "modena_iid_seed101",
    )
    parser.add_argument("--permutation-seed", type=int, default=91_011)
    args = parser.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    audit = derive(source, output, args.permutation_seed)
    print(json.dumps({key: value for key, value in audit.items() if key != "permutations_by_scenario"}, indent=2))


if __name__ == "__main__":
    main()

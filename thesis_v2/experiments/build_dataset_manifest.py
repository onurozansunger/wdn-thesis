"""Audit generated thesis datasets and write JSON/Markdown manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
OUT = V2 / "outputs" / "manifests"
FAMILY = {0: "clean", 1: "random", 2: "replay", 3: "stealthy", 4: "noise", 5: "targeted"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0, "median": None, "q25": None, "q75": None, "q90": None}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
    }


def robust_clean_scale(snapshots: list, corrupted: list, n_nodes: int):
    residuals: dict[int, list[np.ndarray]] = defaultdict(list)
    global_values: list[np.ndarray] = []
    for snap, corr in zip(snapshots, corrupted):
        observed = np.asarray(corr.pressure_mask).astype(bool)
        attacked = np.asarray(corr.pressure_anomaly).astype(bool)
        clean = observed & ~attacked
        if not clean.any():
            continue
        residual = np.asarray(corr.pressure_obs) - np.asarray(snap.pressure_true)
        global_values.append(residual[clean])
        for node in np.flatnonzero(clean):
            residuals[int(node)].append(np.asarray([residual[node]]))

    global_residual = np.concatenate(global_values) if global_values else np.asarray([0.0])
    global_median = float(np.median(global_residual))
    global_scale = max(
        float(1.4826 * np.median(np.abs(global_residual - global_median))), 1e-6
    )
    medians = np.full(n_nodes, global_median, dtype=np.float64)
    scales = np.full(n_nodes, global_scale, dtype=np.float64)
    for node, chunks in residuals.items():
        values = np.concatenate(chunks)
        if values.size < 10:
            continue
        median = float(np.median(values))
        scale = float(1.4826 * np.median(np.abs(values - median)))
        medians[node] = median
        scales[node] = max(scale, 1e-6)
    return medians, scales, global_scale


def episode_runs(snapshots: list, corrupted: list) -> dict[str, list[int]]:
    runs: dict[str, list[int]] = defaultdict(list)
    current_scenario = None
    current_family = None
    length = 0
    for snap, corr in zip(snapshots, corrupted):
        scenario = int(snap.scenario_id)
        family = FAMILY.get(int(getattr(corr, "attack_type_id", 0)), "unknown")
        if scenario != current_scenario or family != current_family:
            if current_family is not None:
                runs[current_family].append(length)
            current_scenario, current_family, length = scenario, family, 1
        else:
            length += 1
    if current_family is not None:
        runs[current_family].append(length)
    return runs


def build(data_dir: Path) -> dict:
    required = ["generate_config.yaml", "graph.pkl", "snapshots.pkl", "corrupted.pkl"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{data_dir}: missing {', '.join(missing)}")

    cfg = yaml.safe_load((data_dir / "generate_config.yaml").read_text())
    with (data_dir / "graph.pkl").open("rb") as stream:
        graph = pickle.load(stream)
    with (data_dir / "snapshots.pkl").open("rb") as stream:
        snapshots = pickle.load(stream)
    with (data_dir / "corrupted.pkl").open("rb") as stream:
        corrupted = pickle.load(stream)

    if len(snapshots) != len(corrupted):
        raise ValueError("snapshot/corruption length mismatch")

    n_nodes = int(graph.num_nodes)
    medians, scales, global_scale = robust_clean_scale(snapshots, corrupted, n_nodes)
    family_snapshots = Counter()
    family_attacked = Counter()
    displacements: dict[str, list[np.ndarray]] = defaultdict(list)
    observed_pressure = attacked_observed = labels_on_missing = 0

    for snap, corr in zip(snapshots, corrupted):
        family = FAMILY.get(int(getattr(corr, "attack_type_id", 0)), "unknown")
        family_snapshots[family] += 1
        mask = np.asarray(corr.pressure_mask).astype(bool)
        attacked = np.asarray(corr.pressure_anomaly).astype(bool)
        observed_pressure += int(mask.sum())
        attacked_observed += int((attacked & mask).sum())
        labels_on_missing += int((attacked & ~mask).sum())
        selected = attacked & mask
        family_attacked[family] += int(selected.sum())
        if selected.any():
            nodes = np.flatnonzero(selected)
            residual = np.asarray(corr.pressure_obs) - np.asarray(snap.pressure_true)
            z = np.abs((residual[nodes] - medians[nodes]) / scales[nodes])
            displacements[family].append(z)

    runs = episode_runs(snapshots, corrupted)
    run_summary = {
        family: {
            "n_runs": len(lengths),
            "median": float(np.median(lengths)),
            "min": int(min(lengths)),
            "max": int(max(lengths)),
        }
        for family, lengths in sorted(runs.items())
    }
    displacement_summary = {
        family: quantiles(np.concatenate(chunks) if chunks else np.asarray([]))
        for family, chunks in sorted(displacements.items())
    }

    scenarios = sorted({int(s.scenario_id) for s in snapshots})
    steps = Counter(int(s.scenario_id) for s in snapshots)
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_head = None

    return {
        "dataset": str(data_dir.relative_to(ROOT)),
        "config": cfg,
        "git_head": git_head,
        "files": {name: {"sha256": sha256(data_dir / name), "bytes": (data_dir / name).stat().st_size} for name in required},
        "network": {"nodes": n_nodes, "edges": int(graph.num_edges)},
        "sampling": {
            "scenarios": len(scenarios),
            "snapshots": len(snapshots),
            "timesteps_per_scenario": sorted(set(steps.values())),
        },
        "pressure_observations": {
            "observed": observed_pressure,
            "attacked_observed": attacked_observed,
            "labels_on_missing": labels_on_missing,
            "empirical_missing_rate": 1.0 - observed_pressure / (len(snapshots) * n_nodes),
        },
        "family_snapshot_counts": dict(sorted(family_snapshots.items())),
        "family_attacked_sensor_counts": dict(sorted(family_attacked.items())),
        "family_run_lengths": run_summary,
        "clean_residual_global_robust_scale": global_scale,
        "standardised_displacement": displacement_summary,
        "integrity": {
            "lengths_match": len(snapshots) == len(corrupted),
            "no_attack_labels_on_missing": labels_on_missing == 0,
            "all_values_finite": all(
                np.isfinite(np.asarray(c.pressure_obs)).all() for c in corrupted
            ),
        },
    }


def markdown(report: dict) -> str:
    lines = [
        f"# Dataset manifest: `{report['dataset']}`",
        "",
        f"- Network: {report['network']['nodes']} nodes, {report['network']['edges']} edges",
        f"- Sampling: {report['sampling']['scenarios']} scenarios, {report['sampling']['snapshots']} snapshots",
        f"- Empirical pressure missing rate: {report['pressure_observations']['empirical_missing_rate']:.3f}",
        f"- Attack labels on missing sensors: {report['pressure_observations']['labels_on_missing']}",
        "",
        "| Family | Snapshots | Attacked sensor-readings | Median displacement | IQR displacement | Median run |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    families = sorted(report["family_snapshot_counts"])
    for family in families:
        displacement = report["standardised_displacement"].get(family, {})
        run = report["family_run_lengths"].get(family, {})
        median = displacement.get("median")
        q25, q75 = displacement.get("q25"), displacement.get("q75")
        lines.append(
            f"| {family} | {report['family_snapshot_counts'].get(family, 0)} | "
            f"{report['family_attacked_sensor_counts'].get(family, 0)} | "
            f"{('--' if median is None else f'{median:.3f}')} | "
            f"{('--' if q25 is None else f'{q25:.3f}--{q75:.3f}')} | "
            f"{run.get('median', '--')} |"
        )
    lines.extend(["", f"Integrity checks: `{json.dumps(report['integrity'], sort_keys=True)}`", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for data_dir in args.data_dirs:
        resolved = data_dir if data_dir.is_absolute() else ROOT / data_dir
        report = build(resolved)
        safe_name = report["dataset"].replace("/", "__")
        json_path = OUT / f"{safe_name}.json"
        md_path = OUT / f"{safe_name}.md"
        json_path.write_text(json.dumps(report, indent=2) + "\n")
        md_path.write_text(markdown(report))
        print(markdown(report))
        print(f"wrote {json_path.relative_to(ROOT)} and {md_path.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()


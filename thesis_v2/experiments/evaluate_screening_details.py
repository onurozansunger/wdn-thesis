"""Add severity- and event-level diagnostics to completed screening runs.

The training script reports sensor-level aggregate metrics.  This evaluator
replays a saved checkpoint on the identical scenario split and adds the two
operational views pre-registered in ``EXPERIMENT_PROTOCOL.md``:

* performance by true attack displacement, measured in robust clean-residual
  standard deviations; and
* coherent-episode detection rate and delay.

Severity-bin metrics compare the attacked observations in one bin against the
same pool of clean test observations.  This avoids the meaningless all-positive
"F1 within a bin" calculation while keeping the bins directly comparable.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_workshop_routing import load_split, to_device  # noqa: E402
from wdn.corruption import ID_TO_ATTACK_TYPE  # noqa: E402
from wdn.models.temporal_moe import TemporalMixtureOfExpertsGNN  # noqa: E402


SEVERITY_BINS = (
    ("<0.5", 0.0, 0.5),
    ("0.5-1", 0.5, 1.0),
    ("1-2", 1.0, 2.0),
    ("2-4", 2.0, 4.0),
    (">=4", 4.0, float("inf")),
)


def robust_clean_scale(snapshots: list, corrupted: list, n_nodes: int):
    """Per-node median and MAD of observed, unattacked training residuals."""
    rows: list[list[float]] = [[] for _ in range(n_nodes)]
    global_values: list[float] = []
    for snap, corr in zip(snapshots, corrupted):
        observed = np.asarray(corr.pressure_mask).astype(bool)
        attacked = np.asarray(corr.pressure_anomaly).astype(bool)
        residual = (
            np.asarray(corr.pressure_obs, dtype=np.float64)
            - np.asarray(snap.pressure_true, dtype=np.float64)
        )
        for node in np.flatnonzero(observed & ~attacked):
            value = float(residual[node])
            rows[int(node)].append(value)
            global_values.append(value)

    values = np.asarray(global_values, dtype=np.float64)
    global_median = float(np.median(values)) if values.size else 0.0
    global_scale = (
        float(1.4826 * np.median(np.abs(values - global_median)))
        if values.size else 1.0
    )
    global_scale = max(global_scale, 1e-6)
    location = np.full(n_nodes, global_median, dtype=np.float64)
    scale = np.full(n_nodes, global_scale, dtype=np.float64)
    for node, row in enumerate(rows):
        if len(row) < 10:
            continue
        arr = np.asarray(row, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(1.4826 * np.median(np.abs(arr - median)))
        location[node] = median
        scale[node] = max(mad, 1e-6)
    return location, scale


def episode_lookup(snapshots: list, corrupted: list) -> dict[tuple[int, int], dict]:
    """Map each endpoint to its scenario-local clean or attack interval."""
    lookup: dict[tuple[int, int], dict] = {}
    current_scenario = current_family = None
    episode_number = 0
    start = 0
    for snap, corr in zip(snapshots, corrupted):
        scenario = int(snap.scenario_id)
        timestep = int(snap.timestep)
        family = int(getattr(corr, "attack_type_id", 0))
        if scenario != current_scenario:
            current_scenario = scenario
            current_family = None
            episode_number = 0
        if family != current_family:
            episode_number += 1
            current_family = family
            start = timestep
        lookup[(scenario, timestep)] = {
            "key": f"{scenario}:{episode_number}",
            "scenario": scenario,
            "family_id": family,
            "family": ID_TO_ATTACK_TYPE.get(family, str(family)),
            "is_attack": family != 0,
            "start_timestep": start,
        }
    return lookup


def binary_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predictions = scores > threshold
    both_classes = np.unique(labels).size == 2
    return {
        "n": int(labels.size),
        "positives": int(labels.sum()),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)) if both_classes else None,
    }


def severity_report(
    scores: np.ndarray,
    labels: np.ndarray,
    severity: np.ndarray,
    threshold: float,
) -> dict:
    clean = labels == 0
    report = {}
    for name, lower, upper in SEVERITY_BINS:
        positive = (labels == 1) & (severity >= lower) & (severity < upper)
        selected = clean | positive
        if not positive.any():
            report[name] = {"positives": 0}
            continue
        item = binary_metrics(scores[selected], labels[selected], threshold)
        item["severity_lower"] = lower
        item["severity_upper"] = None if np.isinf(upper) else upper
        report[name] = item
    return report


def event_report(
    events: dict[str, dict],
    operational_fraction_threshold: float,
    calibration_clean_endpoints: int,
) -> dict:
    by_family: dict[str, list[dict]] = defaultdict(list)
    attacked = [event for event in events.values() if event["is_attack"]]
    clean = [event for event in events.values() if not event["is_attack"]]
    for event in attacked:
        by_family[event["family"]].append(event)

    def summarise(
        rows: list[dict],
        detection_field: str,
        delay_field: str,
        evaluable_delay_field: str,
    ) -> dict:
        detected = [row for row in rows if row[detection_field]]
        delay = [row[delay_field] for row in detected]
        evaluable_delay = [row[evaluable_delay_field] for row in detected]
        return {
            "events": len(rows),
            "detected": len(detected),
            "detection_rate": len(detected) / len(rows) if rows else 0.0,
            "median_delay_from_episode_start": (
                float(np.median(delay)) if delay else None
            ),
            "median_delay_from_first_evaluable": (
                float(np.median(evaluable_delay)) if evaluable_delay else None
            ),
        }

    localisation_false_alarms = sum(bool(row["false_alarm"]) for row in clean)
    operational_false_alarms = sum(bool(row["operational_false_alarm"]) for row in clean)
    clean_endpoints = sum(int(row["evaluable_endpoints"]) for row in clean)
    clean_alarm_endpoints = sum(int(row["operational_alarm_endpoints"]) for row in clean)
    operational_overall = summarise(
        attacked,
        "operational_detected",
        "operational_delay_from_episode_start",
        "operational_delay_from_first_evaluable",
    )
    return {
        "definition": (
            "The operational alarm uses the fraction of observed sensors above "
            "the sensor threshold. Its fraction threshold is the 95th percentile "
            "of clean validation endpoints (higher interpolation). Delays are in "
            "dataset timesteps and exclude missed events. The localisation-any "
            "diagnostic reports whether any truly attacked sensor was flagged."
        ),
        "operational_fraction_threshold": operational_fraction_threshold,
        "calibration_clean_endpoints": calibration_clean_endpoints,
        "overall": operational_overall,
        "clean_intervals": {
            "intervals": len(clean),
            "false_alarms": operational_false_alarms,
            "false_alarm_rate": operational_false_alarms / len(clean) if clean else None,
        },
        "clean_endpoints": {
            "endpoints": clean_endpoints,
            "false_alarms": clean_alarm_endpoints,
            "false_alarm_rate": (
                clean_alarm_endpoints / clean_endpoints if clean_endpoints else None
            ),
        },
        "localisation_any": {
            "attack_intervals": summarise(
                attacked,
                "detected",
                "delay_from_episode_start",
                "delay_from_first_evaluable",
            ),
            "clean_intervals": {
                "intervals": len(clean),
                "false_alarms": localisation_false_alarms,
                "false_alarm_rate": (
                    localisation_false_alarms / len(clean) if clean else None
                ),
            },
        },
        "per_family": {
            family: summarise(
                rows,
                "operational_detected",
                "operational_delay_from_episode_start",
                "operational_delay_from_first_evaluable",
            )
            for family, rows in sorted(by_family.items())
        },
        "events": [events[key] for key in sorted(events)],
    }


def make_model(args: dict, sample: dict, device: torch.device):
    return TemporalMixtureOfExpertsGNN(
        node_in_dim=sample["x_seq"][0].shape[1],
        edge_in_dim=sample["edge_attr"].shape[1],
        hidden_dim=int(args["hidden_dim"]),
        num_experts=int(args["num_experts"]),
        router_hidden_dim=int(args["router_hidden_dim"]),
        num_layers=2,
        num_temporal_layers=int(args.get("num_temporal_layers", 1)),
        window_size=int(args["window_size"]),
        dropout=0.1,
        gnn_type=args["gnn_type"],
        heads=4,
        use_pattern_features=not bool(args.get("no_pattern_features", False)),
        use_topology=not bool(args.get("no_topology", False)),
    ).to(device)


def forward_model(model, batch: dict) -> dict:
    return model(
        x_seq=batch["x_seq"],
        edge_index=batch["edge_index"],
        edge_attr=batch["edge_attr"],
        is_original_edge=batch["is_original_edge"],
        batch_size=batch["batch_size"],
        num_nodes_per_graph=batch["num_nodes"],
        pressure_obs=batch["pressure_obs"],
        flow_obs=batch["flow_obs"],
        pressure_mask=batch["pressure_mask"],
        flow_mask=batch["flow_mask"],
    )


@torch.no_grad()
def calibrate_sensor_threshold(model, loader, device, min_timestep: int):
    """F1 threshold on validation endpoints shared by every architecture."""
    scores_all, labels_all = [], []
    cursor = 0
    for raw in loader:
        batch = to_device(raw, device)
        scores = torch.sigmoid(forward_model(model, batch)["pressure_anomaly_logits"])
        n_nodes = int(batch["num_nodes"])
        for local in range(int(batch["batch_size"])):
            window = loader.dataset.windows[cursor + local]
            snap = loader.dataset.snapshots[window[-1]]
            if int(snap.timestep) < min_timestep:
                continue
            start, stop = local * n_nodes, (local + 1) * n_nodes
            observed = batch["pressure_mask"][start:stop] > 0
            scores_all.append(scores[start:stop][observed].cpu().numpy())
            labels_all.append(
                batch["pressure_anomaly"][start:stop][observed].cpu().numpy().astype(int)
            )
        cursor += int(batch["batch_size"])
    scores = np.concatenate(scores_all)
    labels = np.concatenate(labels_all)
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.02, 0.98, 97):
        value = f1_score(labels, scores > threshold, zero_division=0)
        if value > best_f1:
            best_threshold, best_f1 = float(threshold), float(value)
    return best_threshold, best_f1, int(labels.size)


@torch.no_grad()
def calibrate_event_fraction(
    model,
    loader,
    device,
    sensor_threshold: float,
    min_timestep: int,
):
    """95th percentile of the flagged-sensor fraction on clean validation endpoints."""
    fractions = []
    cursor = 0
    for raw in loader:
        batch = to_device(raw, device)
        scores = torch.sigmoid(forward_model(model, batch)["pressure_anomaly_logits"])
        n_nodes = int(batch["num_nodes"])
        for local in range(int(batch["batch_size"])):
            window = loader.dataset.windows[cursor + local]
            snap = loader.dataset.snapshots[window[-1]]
            if int(snap.timestep) < min_timestep:
                continue
            if int(batch["attack_report"][local]) != 0:
                continue
            start, stop = local * n_nodes, (local + 1) * n_nodes
            observed = batch["pressure_mask"][start:stop] > 0
            if not bool(observed.any()):
                continue
            fraction = ((scores[start:stop] > sensor_threshold) & observed).sum() / observed.sum()
            fractions.append(float(fraction.cpu()))
        cursor += int(batch["batch_size"])
    if not fractions:
        raise RuntimeError("no clean validation endpoints for event-alarm calibration")
    threshold = float(np.quantile(fractions, 0.95, method="higher"))
    return threshold, len(fractions)


@torch.no_grad()
def evaluate_run(run_dir: Path, device: torch.device) -> dict:
    args = json.loads((run_dir / "args.json").read_text())
    saved = json.loads((run_dir / "test_results.json").read_text())
    data_dir = ROOT / args["data_dir"]
    (train_loader, val_loader, test_loader, normalizer), graph, _ = load_split(
        data_dir,
        int(args["window_size"]),
        int(args["batch_size"]),
        args["norm_mode"],
    )
    model = make_model(args, test_loader.dataset[0], device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()

    location, scale = robust_clean_scale(
        train_loader.dataset.snapshots,
        train_loader.dataset.corrupted,
        graph.num_nodes,
    )
    episodes = episode_lookup(
        test_loader.dataset.snapshots,
        test_loader.dataset.corrupted,
    )
    event_rows: dict[str, dict] = {}
    raw_scores_all, raw_labels_all = [], []
    scores_all, labels_all, severity_all = [], [], []
    family_all = []
    cursor = 0
    threshold = float(saved["threshold"])
    aligned_min_timestep = 5
    aligned_threshold, aligned_val_f1, aligned_val_n = calibrate_sensor_threshold(
        model, val_loader, device, aligned_min_timestep
    )
    event_fraction_threshold, event_calibration_n = calibrate_event_fraction(
        model,
        val_loader,
        device,
        aligned_threshold,
        aligned_min_timestep,
    )

    for raw in test_loader:
        batch = to_device(raw, device)
        out = forward_model(model, batch)
        batch_scores = torch.sigmoid(out["pressure_anomaly_logits"]).cpu().numpy()
        batch_labels = batch["pressure_anomaly"].cpu().numpy().astype(int)
        batch_mask = batch["pressure_mask"].cpu().numpy().astype(bool)
        n_nodes = int(batch["num_nodes"])

        # Dataset values are normalised.  The physical residual equals the
        # normalised residual times the fitted per-node normaliser scale.
        p_obs = batch["pressure_obs"].cpu()
        p_true = batch["y_pressure"].cpu()
        residual_physical = (
            normalizer.denormalize_pressure(p_obs)
            - normalizer.denormalize_pressure(p_true)
        ).numpy()

        for local in range(int(batch["batch_size"])):
            start, stop = local * n_nodes, (local + 1) * n_nodes
            window = test_loader.dataset.windows[cursor + local]
            snap_idx = window[-1]
            snap = test_loader.dataset.snapshots[snap_idx]
            corr = test_loader.dataset.corrupted[snap_idx]
            family_id = int(getattr(corr, "attack_type_id", 0))
            observed = batch_mask[start:stop]
            labels = batch_labels[start:stop]
            scores = batch_scores[start:stop]
            raw_scores_all.append(scores[observed])
            raw_labels_all.append(labels[observed])
            if int(snap.timestep) < aligned_min_timestep:
                continue
            nodes = np.arange(n_nodes)
            z = np.full(n_nodes, np.nan, dtype=np.float64)
            attacked_observed = observed & (labels == 1)
            z[attacked_observed] = np.abs(
                (residual_physical[start:stop][attacked_observed]
                 - location[nodes[attacked_observed]])
                / scale[nodes[attacked_observed]]
            )

            scores_all.append(scores[observed])
            labels_all.append(labels[observed])
            severity_all.append(z[observed])
            family_all.append(np.full(int(observed.sum()), family_id, dtype=int))

            info = episodes.get((int(snap.scenario_id), int(snap.timestep)))
            if info is not None and (not info["is_attack"] or attacked_observed.any()):
                row = event_rows.setdefault(
                    info["key"],
                    {
                        **info,
                        "first_evaluable_timestep": int(snap.timestep),
                        "last_evaluable_timestep": int(snap.timestep),
                        "evaluable_endpoints": 0,
                        "attacked_observations": 0,
                        "detected": False,
                        "false_alarm": False,
                        "first_detection_timestep": None,
                        "operational_detected": False,
                        "operational_false_alarm": False,
                        "operational_alarm_endpoints": 0,
                        "first_operational_detection_timestep": None,
                    },
                )
                row["last_evaluable_timestep"] = int(snap.timestep)
                row["evaluable_endpoints"] += 1
                row["attacked_observations"] += int(attacked_observed.sum())
                flagged_fraction = float(
                    ((scores > aligned_threshold) & observed).sum()
                    / max(int(observed.sum()), 1)
                )
                operational_alarm = flagged_fraction > event_fraction_threshold
                if info["is_attack"]:
                    hit = bool(((scores > aligned_threshold) & attacked_observed).any())
                    if hit and not row["detected"]:
                        row["detected"] = True
                        row["first_detection_timestep"] = int(snap.timestep)
                    if operational_alarm and not row["operational_detected"]:
                        row["operational_detected"] = True
                        row["first_operational_detection_timestep"] = int(snap.timestep)
                else:
                    row["false_alarm"] = bool(
                        row["false_alarm"]
                        or ((scores > aligned_threshold) & observed).any()
                    )
                    row["operational_false_alarm"] = bool(
                        row["operational_false_alarm"] or operational_alarm
                    )
                    row["operational_alarm_endpoints"] += int(operational_alarm)

        cursor += int(batch["batch_size"])

    for row in event_rows.values():
        detected_at = row["first_detection_timestep"]
        row["delay_from_episode_start"] = (
            detected_at - row["start_timestep"] if detected_at is not None else None
        )
        row["delay_from_first_evaluable"] = (
            detected_at - row["first_evaluable_timestep"]
            if detected_at is not None else None
        )
        operational_at = row["first_operational_detection_timestep"]
        row["operational_delay_from_episode_start"] = (
            operational_at - row["start_timestep"] if operational_at is not None else None
        )
        row["operational_delay_from_first_evaluable"] = (
            operational_at - row["first_evaluable_timestep"]
            if operational_at is not None else None
        )

    raw_scores = np.concatenate(raw_scores_all)
    raw_labels = np.concatenate(raw_labels_all)
    scores = np.concatenate(scores_all)
    labels = np.concatenate(labels_all)
    severity = np.concatenate(severity_all)
    families = np.concatenate(family_all)
    recomputed = binary_metrics(raw_scores, raw_labels, threshold)
    saved_pressure = saved["anomaly_detection"]["pressure"]
    agreement = {
        metric: abs(recomputed[metric] - float(saved_pressure[metric]))
        for metric in ("f1", "precision", "recall", "auprc", "auroc")
    }
    if max(agreement.values()) > 1e-6:
        raise RuntimeError(f"recomputed metrics disagree with saved result: {agreement}")

    return {
        "run_id": run_dir.name,
        "data_dir": args["data_dir"],
        "model_seed": int(args["seed"]),
        "window_size": int(args["window_size"]),
        "num_experts": int(args["num_experts"]),
        "threshold": threshold,
        "aggregate_recomputed": recomputed,
        "aggregate_agreement_absolute_error": agreement,
        "aligned_sensor_metrics": {
            "minimum_timestep": aligned_min_timestep,
            "threshold": aligned_threshold,
            "validation_f1": aligned_val_f1,
            "validation_observations": aligned_val_n,
            "test": binary_metrics(scores, labels, aligned_threshold),
        },
        "severity": {
            "definition": (
                "Absolute reported-minus-clean displacement divided by the "
                "training sensor's robust clean residual scale. Each bin is "
                "evaluated against the shared clean-test control pool."
            ),
            "overall": severity_report(scores, labels, severity, aligned_threshold),
            "per_family": {
                ID_TO_ATTACK_TYPE.get(int(family), str(family)): severity_report(
                    scores[(families == family) | (labels == 0)],
                    labels[(families == family) | (labels == 0)],
                    severity[(families == family) | (labels == 0)],
                    aligned_threshold,
                )
                for family in sorted(set(families[labels == 1]))
            },
        },
        "events": event_report(
            event_rows,
            event_fraction_threshold,
            event_calibration_n,
        ),
    }


def discover(run_root: Path) -> list[Path]:
    return [
        directory for directory in sorted(run_root.iterdir())
        if directory.is_dir()
        and (directory / "args.json").exists()
        and (directory / "test_results.json").exists()
        and (directory / "best_model.pt").exists()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=V2 / "outputs" / "runs" / "modena_screening",
    )
    parser.add_argument("--run-dir", action="append", type=Path, default=[])
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
        help="Use CPU while training occupies the accelerator, if required.",
    )
    args = parser.parse_args()
    run_dirs = args.run_dir or discover(args.run_root)
    if not run_dirs:
        raise SystemExit("no completed runs found")
    selected = args.device
    if selected == "auto":
        selected = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    device = torch.device(selected)
    print(f"device={device}; runs={len(run_dirs)}")
    for run_dir in run_dirs:
        run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
        report = evaluate_run(run_dir, device)
        path = run_dir / "detailed_analysis.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        event = report["events"]["overall"]
        print(
            f"{run_dir.name}: event detection={event['detection_rate']:.3f} "
            f"({event['detected']}/{event['events']}); wrote {path}"
        )


if __name__ == "__main__":
    main()

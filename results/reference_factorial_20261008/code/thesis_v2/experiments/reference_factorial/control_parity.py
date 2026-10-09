"""Validate E1R1 against frozen references and TRAIN caches before fitting.

Only one manifest-allowlisted TRAIN scenario per anchor is extracted. Pickle
containers are loaded in the same way as the original pipeline; reserved
Modena scenarios are never extracted or evaluated. No models are fitted and
no existing artifacts are overwritten. The approval marker is written only
after every reference, prediction, feature and metadata comparison passes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
ORIGINAL = ROOT / "runs/operational/early_warning_multiseed_v1"
DEFAULT_RUN = ROOT / "runs/operational/reference_factorial_20261008"
RESERVED_MODENA_811 = {0, 1, 8, 20}
CODE_FILES = (
    "thesis_v2/experiments/reference_factorial/control_parity.py",
    "src/wdn/models/reference_factorial.py",
    "src/wdn/models/robust_reference.py",
    "src/wdn/models/blind_reference.py",
    "src/wdn/run_expert_redesign.py",
    "src/wdn/latency_deployment.py",
    "src/wdn/probe_blind_reference.py",
    "src/wdn/dynamic_residual_features.py",
    "src/wdn/sequential_evidence.py",
    "src/wdn/seasonal_pressure_features.py",
    "src/wdn/observed_history.py",
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return str(Path(path).relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    marker = args.run_dir / "control_parity_approved.json"
    if marker.exists():
        raise RuntimeError(f"Approval marker already exists; refusing to overwrite {marker}")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / "src"))

    import lightgbm  # noqa: F401: load before sklearn's OpenMP runtime
    import joblib
    import numpy as np
    import threadpoolctl
    from wdn.latency_deployment import specialist_bank
    from wdn.models.reference_factorial import ReferenceFactorial
    from wdn.observed_history import observed_history_features
    from wdn.run_expert_redesign import CampaignData
    from wdn.seasonal_pressure_features import endpoint_seasonal_features

    inputs = {}

    def fingerprint(path):
        name = relative(path)
        if name not in inputs:
            inputs[name] = sha(path)
        return inputs[name]

    def array_digest(value):
        return hashlib.sha256(np.asarray(value).tobytes(order="C")).hexdigest()

    def bitwise(actual, expected, context):
        a, b = np.asarray(actual), np.asarray(expected)
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(f"{context}: shape/dtype differ: {a.shape}/{a.dtype} versus {b.shape}/{b.dtype}")
        if a.tobytes(order="C") != b.tobytes(order="C"):
            finite = np.isfinite(a) & np.isfinite(b)
            difference = float(np.max(np.abs(a[finite].astype(float) - b[finite].astype(float)))) if finite.any() else None
            raise AssertionError(f"{context}: not bitwise equal; max finite difference={difference}; "
                                 f"allclose={np.allclose(a, b, equal_nan=True)}")

    class OneTrainCase(CampaignData):
        def __init__(self, directory, scenario, allowed, network, source):
            if scenario not in allowed or (network == "modena" and source == 811 and scenario in RESERVED_MODENA_811):
                raise ValueError("The parity scenario is outside the permitted TRAIN allowlist")
            self.permitted_scenario = scenario
            self.scenario_accesses = []
            super().__init__(directory)

        def scenario(self, sid):
            if int(sid) != self.permitted_scenario:
                raise RuntimeError("Parity attempted to extract another scenario")
            self.scenario_accesses.append(int(sid))
            return super().scenario(sid)

    cases = []
    for network in ("modena", "ltown"):
        manifest_path = ORIGINAL / "features" / f"{network}_train/manifest.json"
        manifest = read(manifest_path)
        fingerprint(manifest_path)
        pieces = {p["seed"]: p for p in manifest["pieces"]}
        signature_path = ORIGINAL / f"stage_e_{network}/folds/signature.json"
        signature = read(signature_path)
        fingerprint(signature_path)
        if signature["sources"] != list(pieces):
            raise ValueError("Fold sources and TRAIN manifest ordering differ")
        full = (ROOT / "runs/operational/seasonal_family_deployment_v2/full/reference.joblib"
                if network == "modena" else ORIGINAL / "stage_e_ltown/reference.joblib")
        if fingerprint(full) != manifest["reference_sha256"]:
            raise ValueError("Full reference differs from the frozen TRAIN feature manifest")
        first = manifest["pieces"][0]
        cases.append((network, "full", full, first, ROOT / first["cache"], manifest))
        for fold, source in enumerate(signature["sources"]):
            folder = ORIGINAL / f"stage_e_{network}/folds/fold_{fold}"
            cases.append((network, f"fold_{fold}", folder / "reference.joblib", pieces[source],
                          folder / "features_held_out.npz", manifest))
    if len(cases) != 13:
        raise ValueError(f"Expected two full and eleven fold cases, got {len(cases)}")

    records = []
    started = time.monotonic()
    with threadpoolctl.threadpool_limits(limits=args.threads):
        for network, role, anchor_path, piece, cache_path, manifest in cases:
            begin = time.monotonic()
            source, sid = int(piece["seed"]), int(piece["scenarios"][0])
            directory = ROOT / "data/thesis_v2" / piece["directory"]
            for name in ("snapshots.pkl", "corrupted.pkl", "events.json", "generate_config.yaml"):
                fingerprint(directory / name)
            anchor_hash = fingerprint(anchor_path)
            cache_hash = fingerprint(cache_path)
            if role == "full" and cache_hash != piece["cache_sha256"]:
                raise ValueError("Full TRAIN feature cache fingerprint differs")
            data = OneTrainCase(directory, sid, piece["scenarios"], network, source)
            raw = data.scenario(sid)
            anchor = joblib.load(anchor_path)
            candidate = ReferenceFactorial(anchor, exclude_target_group=True, robust_reweighting=True)
            for name, a, b in zip(("prediction", "support", "spread"),
                                  candidate.predict_details(raw["values"], raw["mask"]),
                                  anchor.predict_details(raw["values"], raw["mask"])):
                bitwise(a, b, f"{network}/{role}/{name}")
            bitwise(candidate.noise_scale_, anchor.noise_scale_, "outer TRAIN normalization scale")
            bitwise(candidate.reference.noise_scale_, anchor.reference.noise_scale_, "inner TRAIN whitening scale")
            arrays, names = specialist_bank(data, [sid], candidate)
            if names != manifest["feature_names"]:
                raise AssertionError("Feature names/order differ from the frozen bank")
            global_sid = source * 1000 + sid
            with np.load(cache_path, allow_pickle=False) as frozen:
                keep = frozen["scenario"] == global_sid
                if not keep.any():
                    raise AssertionError("The chosen global TRAIN scenario is absent from its frozen cache")
                matched = {key: frozen[key][keep] for key in
                           ("X", "labels", "families", "event", "scenario", "source", "timestep", "node")}
            converted = {"X": np.asarray(arrays["X"], np.float32),
                         "labels": np.asarray(arrays["labels"], np.int8),
                         "families": np.asarray(arrays["families"], np.int8),
                         "event": np.asarray(arrays["event"], np.int32),
                         "scenario": np.asarray(arrays["scenario"], np.int64) + source * 1000,
                         "source": np.full(len(arrays["labels"]), source, np.int64),
                         "timestep": np.asarray(arrays["timestep"], np.int32),
                         "node": np.asarray(arrays["node"], np.int32)}
            for key in converted:
                bitwise(converted[key], matched[key], f"{network}/{role}/frozen cache/{key}")

            seasonal, seasonal_names = endpoint_seasonal_features(arrays, data.scenario)
            bitwise(converted["X"][:, -len(seasonal_names):], seasonal, "reference-independent seasonal columns")
            history_checks = None
            if network == "ltown":
                scale_column = names.index("normal_error_scale")
                expected_history, _ = observed_history_features(
                    raw["values"], raw["mask"], raw["timestep"], arrays["timestep"], arrays["node"],
                    matched["X"][:, scale_column])
                for exclusion, robust in ((True, True), (True, False), (False, True), (False, False)):
                    arm = ReferenceFactorial(anchor, exclude_target_group=exclusion, robust_reweighting=robust)
                    scale = arm.noise_scale_[arrays["node"]].astype(np.float32)
                    history, _ = observed_history_features(raw["values"], raw["mask"], raw["timestep"],
                                                           arrays["timestep"], arrays["node"], scale)
                    bitwise(history, expected_history, "fixed-scale received-history columns across factors")
                history_checks = "All four arms bitwise match the history computed with the frozen cache scale"

            record = {"network": network, "anchor_role": role, "anchor_path": relative(anchor_path),
                      "anchor_sha256": anchor_hash, "source_seed": source, "local_scenario": sid,
                      "global_scenario": global_sid, "case_scope": "allowlisted TRAIN scenario",
                      "cache_path": relative(cache_path), "cache_sha256": cache_hash,
                      "rows": len(arrays["labels"]), "feature_columns": len(names),
                      "prediction_support_spread_bitwise_equal": True,
                      "float32_features_and_metadata_bitwise_equal": True,
                      "inner_and_outer_scales_bitwise_equal": True,
                      "seasonal_columns_bitwise_equal": True, "received_history_preservation": history_checks,
                      "raw_case_array_sha256": {k: array_digest(v) for k, v in raw.items()},
                      "feature_case_sha256": array_digest(converted["X"]),
                      "scenario_accesses": data.scenario_accesses,
                      "elapsed_seconds": time.monotonic() - begin}
            records.append(record)
            print(json.dumps({"case": len(records), "network": network, "anchor_role": role,
                              "source": source, "scenario": sid, "status": "bitwise PASS",
                              "seconds": record["elapsed_seconds"]}), flush=True)
            del data, raw, anchor, candidate, arrays, matched, converted, seasonal

        pools = threadpoolctl.threadpool_info()
    record = {"status": "approved", "comparison": "bitwise", "expected_case_count": 13,
              "completed_case_count": len(records), "cases": records,
              "code_sha256": {p: sha(ROOT / p) for p in CODE_FILES}, "input_sha256": inputs,
              "reserved_scenarios_evaluated": False, "fitting_called": False,
              "data_scope": "Existing TRAIN cases only; pickle containers loaded wholesale, only the allowlisted scenario extracted",
              "calibration_or_evaluation_cases_read": False,
              "kernel_threads": args.threads, "threadpools": pools,
              "floating_point_note": "The historical L-Town fold_0 cache requires BLAS threads=2 for exact reproduction. A threads=1 diagnostic differed in two float32 drift-strength entries by one ULP (maximum 2.9103830456733704e-11); no approval was issued for that run.",
              "score_parity_scope": "This gate checks references and TRAIN feature generation only. Existing E10 extraction records separately establish exact original alarm-count reproduction across 120 evaluation cells. The new factorial pipeline must also verify all six retained control models' calibration threshold anchors; this gate does not claim to rerun full model scoring.",
              "existing_e10_provenance": {
                  "summary_path": "output/supervisor_revision_20261008/analysis/operating_points/complete_summary.json",
                  "summary_sha256": sha(ROOT / "output/supervisor_revision_20261008/analysis/operating_points/complete_summary.json"),
                  "extraction_script_path": "output/supervisor_revision_20261008/analysis/extract_hybrid_scores.py",
                  "extraction_script_sha256": sha(ROOT / "output/supervisor_revision_20261008/analysis/extract_hybrid_scores.py")},
              "python_version": platform.python_version(), "numpy_version": np.__version__,
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": time.monotonic() - started}
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        # Linking rather than replacing also rejects a concurrently created marker.
        os.link(temporary, marker)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"status": "approved", "cases": len(records), "marker": relative(marker)}), flush=True)


if __name__ == "__main__":
    main()

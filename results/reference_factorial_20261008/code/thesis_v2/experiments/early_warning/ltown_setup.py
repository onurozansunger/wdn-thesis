"""Prepare the L-Town arm: corpora, a network-specific reference, feature caches.

L-Town is treated as *the same method on a second network*, never as zero-shot
transfer from Modena. Concretely, that means:

* new operational corpora generated under the frozen benchmark, not the old
  ``v2_ltown`` / ``hard_ltown`` / ``rec_hard_ltown`` / episode datasets, which
  are not interchangeable with it;
* a normal reference fitted on **L-Town TRAIN only** — Modena's reference file
  is never opened here;
* whole corpora assigned entirely to TRAIN or to calibration, the same rule
  Modena uses, so no partial split is fabricated to hit an exact count.

Budget: five 24-scenario TRAIN corpora (120) and four 24-scenario calibration
corpora (96), against Modena's 110 and 99. The difference is reported, not
engineered away.

    /opt/miniconda3/bin/python thesis_v2/experiments/early_warning/ltown_setup.py --stage generate
    /opt/miniconda3/bin/python thesis_v2/experiments/early_warning/ltown_setup.py --stage reference
    /opt/miniconda3/bin/python thesis_v2/experiments/early_warning/ltown_setup.py --stage features
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np

from wdn.models.blind_reference import BlindPressureReference
from wdn.models.robust_reference import RobustBlindReference
from wdn.run_expert_redesign import CampaignData

from build_feature_cache import CORPORA, ROOT, build, write_json

DATA = ROOT / "data/thesis_v2"
CAMPAIGN = ROOT / "runs/operational/early_warning_multiseed_v1"
OUTPUT = CAMPAIGN / "stage_e_ltown"
REFERENCE = OUTPUT / "reference.joblib"

TRAIN_SEEDS = (60811, 61811, 62811, 63811, 64811)
CALIBRATION_SEEDS = (70811, 71811, 72811, 73811)
SCENARIOS = 24


def corpus_name(purpose, seed):
    return f"ew_ltown_{purpose}_seed{seed}"


def peak_rss_gb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024**3 if sys.platform == "darwin" else 1024**2)


def register():
    """Declare the L-Town corpora to the shared feature-cache builder."""
    CORPORA["ltown_train"] = {
        "reference": REFERENCE, "network": "ltown",
        "pieces": [(corpus_name("ltown_train", seed), seed, f"all:{SCENARIOS}")
                   for seed in TRAIN_SEEDS]}
    CORPORA["ltown_calibration"] = {
        "reference": REFERENCE, "network": "ltown",
        "pieces": [(corpus_name("ltown_calibration", seed), seed, f"all:{SCENARIOS}")
                   for seed in CALIBRATION_SEEDS]}
    return ("ltown_train", "ltown_calibration")


def stage_generate():
    generator = Path(__file__).with_name("generate_corpus.py")
    made = []
    for purpose, seeds in (("ltown_train", TRAIN_SEEDS),
                           ("ltown_calibration", CALIBRATION_SEEDS)):
        for seed in seeds:
            if (DATA / corpus_name(purpose, seed)).exists():
                continue
            subprocess.run([sys.executable, str(generator), "--network", "ltown",
                            "--purpose", purpose, "--seed", str(seed),
                            "--scenarios", str(SCENARIOS)], cwd=ROOT, check=True)
            made.append(corpus_name(purpose, seed))
    print(json.dumps({"generated": made,
                      "train_scenarios": SCENARIOS * len(TRAIN_SEEDS),
                      "calibration_scenarios": SCENARIOS * len(CALIBRATION_SEEDS)},
                     indent=2))


def stage_reference():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if REFERENCE.exists():
        print("L-Town reference already fitted; no refit", flush=True)
        return
    started = time.monotonic()
    values, masks, rows = [], [], 0
    for seed in TRAIN_SEEDS:
        directory = DATA / corpus_name("ltown_train", seed)
        if not directory.exists():
            raise SystemExit(f"{directory.name} is missing; run --stage generate first")
        data = CampaignData(directory)
        for sid in range(SCENARIOS):
            arrays = data.scenario(sid)
            normal = arrays["families"] == 0
            values.append(arrays["values"][normal].astype(np.float32))
            masks.append(arrays["mask"][normal])
            rows += int(normal.sum())
        del data
        gc.collect()
        print(f"  collected normal rows through seed {seed}: {rows}", flush=True)
    values = np.concatenate(values).astype(np.float64)
    masks = np.concatenate(masks)
    print(f"fitting the L-Town normal reference on {len(values)} normal rows, "
          f"{values.shape[1]} sensors", flush=True)
    base = BlindPressureReference(rank=16).fit(values, masks)
    reference = RobustBlindReference(base).calibrate_scale(values, masks)
    joblib.dump(reference, REFERENCE)
    summary = {
        "network": "ltown",
        "fitted_on": "L-Town TRAIN only; Modena's reference was never opened",
        "train_seeds": list(TRAIN_SEEDS), "train_scenarios": SCENARIOS * len(TRAIN_SEEDS),
        "normal_rows": int(len(values)), "sensors": int(values.shape[1]), "rank": 16,
        "median_noise_scale_m": float(np.median(reference.noise_scale_)),
        "group_sizes": np.bincount(base.group_).tolist(),
        "seconds": round(time.monotonic() - started, 1),
        "peak_rss_gb": round(peak_rss_gb(), 2),
        "calibration_evaluated": False, "eval_evaluated": False, "test_evaluated": False,
    }
    write_json(OUTPUT / "reference_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def stage_features():
    if not REFERENCE.exists():
        raise SystemExit("Fit the L-Town reference first (--stage reference)")
    for name in register():
        build(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("generate", "reference", "features", "all"))
    args = parser.parse_args()
    if args.stage in ("generate", "all"):
        stage_generate()
    if args.stage in ("reference", "all"):
        stage_reference()
    if args.stage in ("features", "all"):
        stage_features()


if __name__ == "__main__":
    main()

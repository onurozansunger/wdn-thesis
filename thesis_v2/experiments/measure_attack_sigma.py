"""Report attack magnitude in units of the sensor noise for a dataset.

The paper's whole argument is that a detection number means nothing
without the regime it was measured in, and the regime is two numbers:
signal movement over the replay lag, and attack size relative to the
noise floor. That second number was previously quoted from the generator
config by hand, which works only as long as the attack is additive. It is
not: `attack_scale` is multiplicative, so the same config lands at a
different multiple of sigma on a network with different pressures --- the
reason L-Town sits at 52 sigma where Modena sits at 31.

This measures it from the generated data instead, which is what has to
happen before a second network can be placed at the same point on the
regime axis.

    python3 scripts/measure_attack_sigma.py data/hard_modena [...]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def measure(d: Path):
    cfg = yaml.safe_load(open(d / "generate_config.yaml"))
    sigma = float(cfg["corruption"]["noise_sigma_pressure"])
    corrupted = pickle.load(open(d / "corrupted.pkl", "rb"))
    snapshots = pickle.load(open(d / "snapshots.pkl", "rb"))

    per_class: dict[int, list] = {}
    allmag = []
    for c, s in zip(corrupted, snapshots):
        atk = np.asarray(c.pressure_anomaly).astype(bool)
        obs = np.asarray(c.pressure_mask).astype(bool)
        sel = atk & obs
        if not sel.any():
            continue
        # Deviation from the true state, in noise units. The observation
        # noise is part of the reported value, so this is the total
        # displacement a detector actually sees, not the injected bias.
        mag = np.abs(np.asarray(c.pressure_obs)[sel]
                     - np.asarray(s.pressure_true)[sel]) / sigma
        allmag.append(mag)
        k = int(getattr(c, "attack_type_id", -1))
        per_class.setdefault(k, []).append(mag)

    if not allmag:
        print(f"  {d.name}: no attacked+observed sensors")
        return
    a = np.concatenate(allmag)
    amp = cfg.get("demand_pattern_amplitude", 0.0)
    print(f"\n  {d.name}   (pattern amplitude {amp}, sigma {sigma})")
    print(f"    attacked sensor-readings: {a.size:,}")
    print(f"    magnitude in sigma:  median {np.median(a):6.2f}   "
          f"mean {a.mean():6.2f}   p10 {np.percentile(a,10):5.2f}   "
          f"p90 {np.percentile(a,90):6.2f}")
    for k in sorted(per_class):
        v = np.concatenate(per_class[k])
        print(f"      class {k}: median {np.median(v):6.2f}  n={v.size:,}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        measure(Path(arg))

"""Guard against source leakage and promotion on an incomplete or regressing run."""
import copy
import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "thesis_v2/experiments/early_warning"))
    return importlib.import_module("replicate_observed_history")


def reports(runner):
    result = []
    for fold in range(1, 5):
        for seed in runner.SEEDS:
            deltas = {f: .001 for f in runner.FAMILY_NAMES.values()}
            deltas.update(replay=.02, pooled=.001, clean_fpr=-.00001)
            baseline = {f: {"f1": .8} for f in runner.FAMILY_NAMES.values()}
            baseline["_overall"] = {"f1": .9, "clean_fpr": .0003}
            candidate = {f: {"f1": .8 + deltas[f]} for f in runner.FAMILY_NAMES.values()}
            candidate["_overall"] = {"f1": .901, "clean_fpr": .00029}
            result.append({"fold": fold, "seed": seed, "deltas": deltas,
                           "results": {"baseline": {"mixture": baseline},
                                       "observed_history": {"mixture": candidate}}})
    return result


def test_gate_needs_all_primary_pairs_and_protects_every_source(runner):
    rows = reports(runner)
    assert runner.assess(rows)["passes"]
    with pytest.raises(RuntimeError, match="All 20"):
        runner.assess(rows[:-1])
    altered = copy.deepcopy(rows)
    for r in altered[:5]:
        r["deltas"]["drift"] = -.011
    assert not runner.assess(altered)["passes"]
    assert not runner.assess(altered)["checks"]["drift_source_guard"]
    altered = copy.deepcopy(rows)
    for r in altered:
        r["deltas"]["clean_fpr"] = .000001
    assert not runner.assess(altered)["passes"]


def test_supplemental_pilot_cannot_rescue_primary_failure(runner):
    rows = reports(runner)
    for r in rows[:5]:
        r["deltas"]["replay"] = -.001
    extra = copy.deepcopy(rows[0])
    extra.update(fold=0, seed=702)
    extra["deltas"]["replay"] = 100
    assert not runner.assess(rows + [extra])["passes"]


def test_partition_rejects_source_leakage(runner):
    manifest = {"pieces": [{"seed": s, "scenarios": [0, 1]} for s in (1, 2)]}
    held = {"source": np.full(10, 2), "scenario": np.repeat([2000, 2001], 5),
            "families": np.tile(np.arange(1, 6), 2), "labels": np.ones(10)}
    train = {"source": np.array([1]), "scenario": np.array([1000])}
    runner.validate_partition(train, held, manifest, 1)
    train["source"][0] = 2
    with pytest.raises(RuntimeError, match="Source-held"):
        runner.validate_partition(train, held, manifest, 1)

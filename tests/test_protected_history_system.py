import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "thesis_v2/experiments/early_warning"))
    return importlib.import_module("protected_history_system")


def test_exact_threshold_sweep_matches_boolean_or_with_ties(runner):
    rng = np.random.default_rng(9)
    families = np.tile(np.arange(6), 200)
    labels = (families > 0) & (rng.random(len(families)) < .15)
    arrays = {"labels": labels, "families": families, "specialist": rng.random(len(families)) < .1}
    score = np.round(rng.random(len(families)), 1)
    thresholds = np.array([-.1, 0., .2, .5, 1., 1.1])
    reports = runner.threshold_reports(score, thresholds, arrays)
    for threshold, report in zip(thresholds, reports):
        expected = runner.rc.fast_scorer(arrays)(arrays["specialist"] | (score > threshold))
        for family in runner.FAMILY_NAMES.values():
            for k in ("f1", "tp", "fp", "fn"):
                assert report[family][k] == expected[family][k]
        for k in ("f1", "tp", "fp", "fn", "clean_fpr", "all_negative_fpr", "clean_period_fp", "worst_family_f1"):
            assert report["_overall"][k] == expected["_overall"][k]


def test_selective_variant_keeps_exactly_five_slots_and_baseline_router(runner):
    arrays = {"baseline_experts": np.tile(np.arange(5.), (4, 1)),
              "history_experts": np.full((4, 5), 7.),
              "baseline_routing": np.full((4, 5), .2),
              "history_routing": np.tile([1., 0., 0., 0., 0.], (4, 1))}
    before = arrays["baseline_experts"].copy()
    experts, routing = runner.variant_scores(arrays, "selective_history")
    np.testing.assert_equal(experts[0], [7, 1, 7, 3, 4])
    assert experts.shape == (4, 5)
    assert routing is arrays["baseline_routing"]
    np.testing.assert_equal(arrays["baseline_experts"], before)


def test_protection_does_not_trade_noise_or_false_alarms_for_replay(runner):
    import copy
    base = {f: {"f1": .9} for f in runner.FAMILY_NAMES.values()}
    base["replay"]["f1"] = .65
    base["_overall"] = {"f1": .88, "clean_fpr": .0003}
    candidate = copy.deepcopy(base)
    candidate["replay"]["f1"] = .8
    assert runner.protected(candidate, base)
    candidate["noise"]["f1"] = .894
    assert not runner.protected(candidate, base)
    candidate["noise"]["f1"] = .9
    candidate["_overall"]["clean_fpr"] = .000301
    assert not runner.protected(candidate, base)

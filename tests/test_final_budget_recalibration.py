import sys
from pathlib import Path

import numpy as np


EXPERIMENT = Path(__file__).parents[1] / "thesis_v2/experiments/early_warning"
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))

from recalibrate_final_budget import gated_branch_scores, search_final_budget
from screen_router_temperature import transformed_mixture


def test_gating_precedes_threshold_calibration():
    corpus = {
        "mixture": np.array([.1, .2, .7]),
        "router": np.array([[.7, .1, .1, .05, .05]] * 3),
        "feedback": np.ones((3, 2)),
    }
    scores = {"final_drift": np.array([.2, .8, .9]),
              "final_noise": np.array([.3, .7, .8])}
    keep = {"drift": np.array([.9, .2, .9]),
            "noise": np.array([.9, .9, .1])}
    got = gated_branch_scores(corpus, scores, keep, {"drift": .5, "noise": .5})
    assert np.isneginf(got["drift"][1])
    assert np.isneginf(got["noise"][2])
    np.testing.assert_array_equal(got["mixture"], corpus["mixture"])


def test_search_is_family_agnostic_at_inference_and_respects_guards():
    branch = {"mixture": np.array([.05, .10, .90, .85, .80, .75]),
              "drift": np.zeros(6), "noise": np.zeros(6)}
    clean = np.array([True, True, False, False, False, False])
    labels = np.array([0, 0, 1, 1, 1, 1], dtype=bool)
    families = np.array([0, 0, 1, 2, 3, 4])

    def score(decision):
        names = {0: "clean", 1: "random", 2: "replay", 3: "drift", 4: "noise"}
        out = {}
        worst = 1.0
        for code, name in names.items():
            scope = families == code
            tp = int(np.sum(labels & decision & scope))
            fp = int(np.sum(~labels & decision & scope))
            fn = int(np.sum(labels & ~decision & scope))
            f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            out[name] = {"f1": f1}
            worst = min(worst, f1) if code else worst
        out["targeted"] = {"f1": 1.0}
        tp, fp, fn = (int(np.sum(labels & decision)), int(np.sum(~labels & decision)),
                      int(np.sum(labels & ~decision)))
        out["_overall"] = {"f1": 2 * tp / max(1, 2 * tp + fp + fn),
                           "clean_fpr": float(decision[clean].mean()),
                           "worst_family_f1": worst}
        return out

    reference = {"pooled_f1": .5, "random": .9, "replay": .5,
                 "drift": .9, "noise": .9, "targeted": .9}
    found, count, frontier = search_final_budget(
        branch, clean, score, reference, shares=(0., .5, 1.), budget_scales=(1.,))
    assert count > 0
    assert found is not None
    assert frontier["best_pooled"] is not None
    assert found[2]["_overall"]["clean_fpr"] <= .005


def test_router_transform_is_symmetric_and_normalised():
    experts = np.array([[.1, .2, .3, .4, .5]])
    routing = np.array([[.05, .10, .15, .20, .50]])
    assert np.isclose(transformed_mixture(experts, routing, 1., 0.),
                      np.sum(experts * routing))
    assert np.isclose(transformed_mixture(experts, routing, 1., 1.),
                      experts.mean())

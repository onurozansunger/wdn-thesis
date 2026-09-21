"""Verification requirements for the early-warning / verifier campaign.

These are the checks listed in EARLY_WARNING_MULTISEED_PROTOCOL.md section 9.
They are deliberately about the properties that would invalidate a claim —
causality, leakage, alignment, one-directionality — rather than about numbers,
which live in the campaign's own reports.
"""
from __future__ import annotations

import numpy as np
import pytest

from wdn.alarm_verifier import (BRANCHES, EARLY_CHANNELS, SCORE_CHANNELS,
                                AlarmVerifier, candidate_rows, verifier_features)
from wdn.early_warning import (CAUSAL_SCORE_NAMES, CLASS_TO_REPORT, EARLY_LOCAL_NAMES,
                               HISTORY_LAGS, early_group_features, early_local_evidence,
                               group_family_state, past_lag_index)
from wdn.early_warning_metrics import (clean_hour_false_alarms, event_warnings,
                                       summarise_events, warning_episodes)
from wdn.evidence_feedback import (RAW_EVIDENCE_NAMES, ROUTER_CLASSES,
                                   SCORE_EVIDENCE_NAMES, contiguous_groups)
from wdn.latency_deployment import forward_max, logit_blend


BANK = list(RAW_EVIDENCE_NAMES) + ["padding_a", "padding_b"]


def synthetic(sensors=6, hours=12, scenarios=2, seed=0, drop=0.4):
    """A small observed-endpoint corpus with realistic gaps and stable ordering."""
    rng = np.random.default_rng(seed)
    rows = {key: [] for key in ("scenario", "timestep", "node", "source",
                                "labels", "families")}
    X = []
    for scenario in range(scenarios):
        for hour in range(hours):
            observed = np.flatnonzero(rng.random(sensors) > drop)
            if not len(observed):
                observed = np.array([0])
            for node in observed:
                rows["scenario"].append(100 + scenario)
                rows["timestep"].append(hour)
                rows["node"].append(int(node))
                rows["source"].append(7)
                rows["labels"].append(0)
                rows["families"].append(0)
                X.append(rng.normal(size=len(BANK)))
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    arrays["X"] = np.asarray(X, dtype=np.float32)
    return arrays


def causal_scores_for(arrays, seed=1):
    rng = np.random.default_rng(seed)
    return rng.random((len(arrays["labels"]), 2))


# --------------------------------------------------------------------------
# 1. The early warning at t cannot see anything after t.
# --------------------------------------------------------------------------

def test_early_warning_ignores_every_observation_after_t():
    arrays = synthetic()
    scores = causal_scores_for(arrays)
    local = early_local_evidence(arrays["X"], BANK, scores)
    features, names, _, starts = early_group_features(local, arrays)
    cut = 6

    disturbed = local.copy()
    future = arrays["timestep"] > cut
    assert future.any()
    disturbed[future] += 1000.0
    changed, changed_names, _, changed_starts = early_group_features(disturbed, arrays)

    assert names == changed_names
    np.testing.assert_array_equal(starts, changed_starts)
    group_time = arrays["timestep"][starts]
    past = group_time <= cut
    np.testing.assert_allclose(features[past], changed[past], equal_nan=True)
    # And the future groups really did move, so the test is not vacuous.
    assert not np.allclose(features[~past], changed[~past], equal_nan=True)


def test_early_head_never_sees_a_delayed_or_pooled_score():
    forbidden = {name for name in SCORE_EVIDENCE_NAMES
                 if name.startswith(("maxpool", "delayed", "final"))}
    assert forbidden
    assert not forbidden & set(EARLY_LOCAL_NAMES)
    assert set(EARLY_LOCAL_NAMES) == set(RAW_EVIDENCE_NAMES) | set(CAUSAL_SCORE_NAMES)


def test_past_lag_index_matches_by_hour_not_by_row_position():
    keys = np.array([[7, 100, 0], [7, 100, 1], [7, 100, 5], [7, 101, 1]], dtype=np.int64)
    index = past_lag_index(keys, (1, 2))
    assert index[0, 0] == -1                    # nothing before the first hour
    assert index[1, 0] == 0                     # hour 1 finds hour 0
    assert index[2, 0] == -1                    # hour 5 must not adopt hour 1
    assert index[2, 1] == -1
    assert index[3, 0] == -1                    # a different scenario is not a lag


def test_early_group_features_are_not_shifted_by_a_missing_hour():
    arrays = synthetic(drop=0.0)
    keep = arrays["timestep"] != 4
    gapped = {key: value[keep] for key, value in arrays.items()}
    scores = causal_scores_for(arrays)
    local = early_local_evidence(arrays["X"], BANK, scores)

    full, _, _, full_starts = early_group_features(local, arrays)
    thin, _, _, thin_starts = early_group_features(local[keep], gapped)

    full_time = arrays["timestep"][full_starts]
    thin_time = gapped["timestep"][thin_starts]
    # Hour 5 lost its immediate predecessor, so its lag-1 flag must go to zero
    # rather than silently pointing at hour 3.
    has_lag1 = [i for i, name in enumerate(
        early_group_features(local, arrays)[1]) if name == "has_lag1"][0]
    assert full[full_time == 5, has_lag1].min() == 1.0
    assert thin[thin_time == 5, has_lag1].max() == 0.0


# --------------------------------------------------------------------------
# 2. The confirmed decision for t cannot see anything after t + delta.
# --------------------------------------------------------------------------

def test_confirmed_decision_ignores_observations_after_t_plus_delta():
    arrays = synthetic(hours=16, seed=3)
    delta = 3
    rng = np.random.default_rng(5)
    causal = rng.random(len(arrays["labels"]))
    head = rng.random(len(arrays["labels"]))

    def final(scores):
        return logit_blend(head, forward_max(scores, arrays, delta), .5)

    cut = 8
    disturbed = causal.copy()
    disturbed[arrays["timestep"] > cut + delta] = 1.0
    before, after = final(causal), final(disturbed)
    protected = arrays["timestep"] <= cut
    np.testing.assert_allclose(before[protected], after[protected])


def test_forward_max_does_not_reach_across_a_scenario_boundary():
    arrays = {"source": np.array([7, 7]), "scenario": np.array([100, 101]),
              "node": np.array([0, 0]), "timestep": np.array([5, 6]),
              "labels": np.zeros(2)}
    pooled = forward_max(np.array([0.1, 0.9]), arrays, 3)
    assert pooled[0] == pytest.approx(0.1)


# --------------------------------------------------------------------------
# 3. No label, onset, family or target identity reaches an inference feature.
# --------------------------------------------------------------------------

def test_no_inference_feature_name_encodes_a_label_or_an_event():
    forbidden = ("label", "anomaly", "attack", "onset", "event", "family",
                 "target", "truth", "clean_value")
    for name in list(EARLY_LOCAL_NAMES) + list(SCORE_CHANNELS) + list(EARLY_CHANNELS):
        assert not any(word in name.lower() for word in forbidden), name


def test_early_features_are_invariant_to_the_labels_and_families():
    arrays = synthetic(seed=11)
    scores = causal_scores_for(arrays)
    local = early_local_evidence(arrays["X"], BANK, scores)
    reference, _, _, _ = early_group_features(local, arrays)
    relabelled = dict(arrays)
    rng = np.random.default_rng(12)
    relabelled["labels"] = rng.integers(0, 2, len(arrays["labels"]))
    relabelled["families"] = rng.integers(0, 5, len(arrays["labels"]))
    changed, _, _, _ = early_group_features(local, relabelled)
    np.testing.assert_allclose(reference, changed, equal_nan=True)


def test_group_family_state_folds_targeted_into_abrupt():
    families = np.array([0, 0, 5, 5, 3, 3])
    starts = np.array([0, 2, 4])
    state = group_family_state(families, starts)
    assert state.tolist() == [0, 1, 3]
    assert ROUTER_CLASSES[1] == "abrupt"


# --------------------------------------------------------------------------
# 4. The verifier can only remove an alarm, and never touches the mixture.
# --------------------------------------------------------------------------

class _AlwaysReject:
    def predict_proba(self, X):
        return np.column_stack((np.ones(len(X)), np.zeros(len(X))))


class _AlwaysAccept:
    def predict_proba(self, X):
        return np.column_stack((np.zeros(len(X)), np.ones(len(X))))


def _verifier(model):
    return AlarmVerifier(("a",), {branch: model for branch in BRANCHES},
                         {branch: 0.5 for branch in BRANCHES}, uses_early=False)


def test_verifier_leaves_rows_outside_the_candidate_region_untouched():
    features = np.zeros((4, 1), dtype=np.float32)
    scores = {"final_drift": np.array([0.1, 0.9, 0.1, 0.9]),
              "final_noise": np.array([0.1, 0.1, 0.9, 0.9])}
    probability = _verifier(_AlwaysReject()).verify(features, scores)
    # Below the pre-threshold the verifier abstains by returning 1.0.
    assert probability["drift"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert probability["noise"].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_verifier_is_one_directional_and_spares_the_general_branch():
    rng = np.random.default_rng(2)
    n = 200
    mixture = rng.random(n)
    scores = {"final_drift": rng.random(n), "final_noise": rng.random(n)}
    features = np.zeros((n, 1), dtype=np.float32)
    veto = np.zeros((n, 2), dtype=bool)
    thresholds = {"mixture": .5, "drift": .5, "noise": .5}

    def decide(keep):
        decision = mixture > thresholds["mixture"]
        for column, branch in enumerate(BRANCHES):
            alarm = (scores[f"final_{branch}"] > thresholds[branch]) & ~veto[:, column]
            if keep is not None:
                alarm = alarm & keep[branch]
            decision |= alarm
        return decision

    baseline = decide(None)
    rejected = decide(_verifier(_AlwaysReject()).keep(features, scores,
                                                      {b: .5 for b in BRANCHES}))
    accepted = decide(_verifier(_AlwaysAccept()).keep(features, scores,
                                                      {b: .5 for b in BRANCHES}))
    assert not np.any(rejected & ~baseline), "the verifier created an alarm"
    assert np.array_equal(accepted, baseline), "an accepting verifier changed the decision"
    # Every general-branch alarm survives the strictest possible verifier.
    assert np.all(rejected[mixture > thresholds["mixture"]])


def test_absent_early_warning_cannot_disable_a_specialist():
    """C3's early channels are inputs to a score, never a gate on the decision."""
    rng = np.random.default_rng(4)
    arrays = synthetic(seed=8)
    n = len(arrays["labels"])
    scores = {name: rng.random(n) for name in SCORE_CHANNELS}
    silent = {
        "probabilities": np.tile(np.array([1., 0., 0., 0., 0.]), (5, 1)),
        "prediction": np.zeros(5, dtype=int), "confidence": np.ones(5),
        "abstain": np.ones(5, dtype=bool),
        "group": np.zeros(n, dtype=int),
        "keys": np.column_stack((np.full(5, 7), np.full(5, 100), np.arange(5))),
    }
    features, columns = verifier_features(arrays["X"], BANK, scores, silent, arrays)
    assert len(columns) == len(RAW_EVIDENCE_NAMES) + len(SCORE_CHANNELS) + len(EARLY_CHANNELS)
    assert features.shape == (n, len(columns))
    # A silent, abstaining early head is just another feature value: the decision
    # path is unchanged because `keep` still comes from the verifier's own cutoff.
    keep = _verifier(_AlwaysAccept()).keep(np.zeros((n, 1), np.float32), scores,
                                           {b: .5 for b in BRANCHES})
    assert all(mask.all() for mask in keep.values())


def test_candidate_region_is_defined_on_negatives_only():
    score = np.r_[np.linspace(0, 1, 100), np.full(100, 5.0)]
    clean = np.r_[np.ones(100, bool), np.zeros(100, bool)]
    inside, threshold = candidate_rows(score, clean, .10)
    assert threshold < 1.0
    assert inside[100:].all(), "large positive scores must be inside the region"
    assert inside[:100].sum() <= 12


# --------------------------------------------------------------------------
# 5. Warning metrics use their declared denominators.
# --------------------------------------------------------------------------

def _result(keys, prediction, abstain=None):
    prediction = np.asarray(prediction)
    n = len(prediction)
    probabilities = np.zeros((n, len(ROUTER_CLASSES)))
    probabilities[np.arange(n), prediction] = 1.
    return {"keys": np.asarray(keys, dtype=np.int64), "prediction": prediction,
            "abstain": np.zeros(n, bool) if abstain is None else np.asarray(abstain),
            "confidence": np.ones(n), "probabilities": probabilities,
            "group": np.arange(n), "starts": np.arange(n)}


def test_a_missed_event_stays_in_the_denominator_as_a_censored_delay():
    keys = np.column_stack((np.full(8, 7), np.full(8, 100), np.arange(8)))
    result = _result(keys, np.zeros(8, dtype=int))
    events = {100: [{"scenario": 100, "source": 7, "family": "drift",
                     "start": 2, "steps": 4}]}
    rows = event_warnings(result, events)
    summary = summarise_events(rows)["drift"]
    assert summary["events"] == 1
    assert summary["warned_by_plus1"] == 0.0
    assert summary["ever_warned"] == 0.0
    assert summary["first_warning_delay_hours"]["censored_missed_events"] == 1
    assert summary["first_warning_delay_hours"]["median"] is None


def test_a_wrong_family_warning_is_not_counted_as_a_correct_one():
    keys = np.column_stack((np.full(8, 7), np.full(8, 100), np.arange(8)))
    prediction = np.zeros(8, dtype=int)
    prediction[3] = 4                       # noise warning during a drift event
    result = _result(keys, prediction)
    events = {100: [{"scenario": 100, "source": 7, "family": "drift",
                     "start": 2, "steps": 4}]}
    summary = summarise_events(event_warnings(result, events))["drift"]
    assert summary["warned_by_plus1"] == 1.0
    assert summary["correct_family_by_plus1"] == 0.0


def test_an_abstention_is_not_a_warning():
    keys = np.column_stack((np.full(4, 7), np.full(4, 100), np.arange(4)))
    result = _result(keys, np.array([3, 3, 3, 3]), abstain=np.ones(4, bool))
    events = {100: [{"scenario": 100, "source": 7, "family": "drift",
                     "start": 0, "steps": 4}]}
    summary = summarise_events(event_warnings(result, events))["drift"]
    assert summary["ever_warned"] == 0.0
    assert warning_episodes(result)["episodes"] == 0


def test_clean_hour_denominator_excludes_open_events():
    keys = np.column_stack((np.full(6, 7), np.full(6, 100), np.arange(6)))
    result = _result(keys, np.array([0, 0, 3, 3, 0, 4]))
    events = {100: [{"scenario": 100, "source": 7, "family": "drift",
                     "start": 2, "steps": 2}]}
    report = clean_hour_false_alarms(result, np.zeros(6), np.arange(6), events)
    assert report["clean_graph_hours"] == 4
    assert report["false_warning_hours"] == 1          # only hour 5
    assert report["noise_false_warning_hours"] == 1
    tail = clean_hour_false_alarms(result, np.zeros(6), np.arange(6), events, span_hours=6)
    assert tail["clean_graph_hours"] == 2              # hours 0 and 1 only
    assert tail["false_warning_hours"] == 0


def test_warning_episodes_group_consecutive_hours_only():
    keys = np.column_stack((np.full(6, 7), np.full(6, 100), np.array([0, 1, 2, 5, 6, 9])))
    result = _result(keys, np.array([3, 3, 0, 4, 4, 3]))
    assert warning_episodes(result)["episodes"] == 3
    assert warning_episodes(result)["warned_graph_hours"] == 5


def test_report_class_map_keeps_replay_and_abrupt_out_of_drift_and_noise():
    assert CLASS_TO_REPORT["replay"] == "other_mechanism"
    assert CLASS_TO_REPORT["abrupt"] == "other_mechanism"
    assert CLASS_TO_REPORT["drift"] == "likely_drift"
    assert CLASS_TO_REPORT["noise"] == "likely_noise"


# --------------------------------------------------------------------------
# 6. Alignment and caching invariants.
# --------------------------------------------------------------------------

def test_contiguous_groups_rejects_a_split_graph_time_block():
    arrays = {"source": np.array([7, 7, 7]), "scenario": np.array([1, 2, 1]),
              "timestep": np.array([0, 0, 0])}
    with pytest.raises(ValueError):
        contiguous_groups(arrays)


def test_early_local_evidence_rejects_a_row_count_mismatch():
    arrays = synthetic()
    with pytest.raises(ValueError):
        early_local_evidence(arrays["X"], BANK, np.zeros((3, 2)))


def test_early_feature_names_are_unique_and_stable():
    arrays = synthetic()
    _, names, _, _ = early_group_features(
        early_local_evidence(arrays["X"], BANK, causal_scores_for(arrays)), arrays)
    assert len(names) == len(set(names))
    assert sum(name.startswith("delta") for name in names) == \
        len(HISTORY_LAGS) * len([n for n in names if n.startswith("delta1_")])

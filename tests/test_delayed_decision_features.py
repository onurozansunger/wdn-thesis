"""The forward window must read the same sensor's future and nothing else."""
import numpy as np
import pytest

from wdn.delayed_decision_features import (FORWARD_COLUMNS, SUMMARY_NAMES,
                                           delayed_decision_features, forward_index)

NAMES = list(FORWARD_COLUMNS)


def arrays_from(rows):
    """rows: (source, scenario, node, timestep, residual)."""
    rows = np.asarray(rows, dtype=float)
    n = len(rows)
    X = np.zeros((n, len(NAMES)), dtype=np.float32)
    X[:, NAMES.index("residual")] = rows[:, 4]
    X[:, NAMES.index("abs_residual")] = np.abs(rows[:, 4])
    return {"X": X, "source": rows[:, 0].astype(int), "scenario": rows[:, 1].astype(int),
            "node": rows[:, 2].astype(int), "timestep": rows[:, 3].astype(int)}


def test_forward_index_stays_inside_its_own_series():
    arrays = arrays_from([(811, 0, 5, 0, 1.), (811, 0, 5, 1, 2.),
                          (811, 0, 6, 2, 9.), (811, 1, 5, 1, 9.), (1811, 0, 5, 1, 9.)])
    index = forward_index(arrays, 1)
    assert index[0, 0] == 1                     # same series, next hour
    assert index[1, 0] == -1                    # no hour 2 for that sensor
    assert (index[2:, 0] == -1).all()           # other sensors are never reachable


def test_gap_in_the_series_is_missing_not_borrowed():
    arrays = arrays_from([(811, 0, 5, 0, 1.), (811, 0, 5, 2, 7.)])
    index = forward_index(arrays, 2)
    assert index[0, 0] == -1                    # hour 1 absent
    assert index[0, 1] == 1                     # hour 2 present
    features, names = delayed_decision_features(arrays, NAMES, 2)
    assert np.isnan(features[0, names.index("residual_t+1")])
    assert features[0, names.index("residual_t+2")] == pytest.approx(7.)


def test_window_summaries_use_only_the_declared_horizon():
    arrays = arrays_from([(811, 0, 5, t, float(t)) for t in range(6)])
    features, names = delayed_decision_features(arrays, NAMES, 3)
    row = features[0]
    assert row[names.index("future_residual_max")] == pytest.approx(3.)
    assert row[names.index("future_residual_rise")] == pytest.approx(3.)
    assert row[names.index("future_residual_slope")] == pytest.approx(1.)
    assert row[names.index("future_support")] == pytest.approx(4.)
    # the last row has no future at all
    tail = features[-1]
    assert tail[names.index("future_support")] == pytest.approx(1.)
    assert np.isnan(tail[names.index("residual_t+1")])


def test_shape_and_names_are_declared_up_front():
    arrays = arrays_from([(811, 0, 5, t, float(t)) for t in range(4)])
    features, names = delayed_decision_features(arrays, NAMES, 3)
    assert features.shape == (4, len(FORWARD_COLUMNS) * 3 + len(SUMMARY_NAMES))
    assert names[-len(SUMMARY_NAMES):] == list(SUMMARY_NAMES)


def test_duplicate_endpoints_are_rejected():
    arrays = arrays_from([(811, 0, 5, 1, 1.), (811, 0, 5, 1, 2.)])
    with pytest.raises(ValueError, match="Duplicate"):
        forward_index(arrays, 1)


def test_zero_latency_is_not_a_delayed_decision():
    arrays = arrays_from([(811, 0, 5, 0, 1.)])
    with pytest.raises(ValueError, match="at least one hour"):
        delayed_decision_features(arrays, NAMES, 0)


def test_shared_history_is_causal_gap_aware_and_series_isolated():
    from wdn.shared_history import HISTORY_COLUMNS, causal_history_features
    rows = [(60811, 60811000, 5, 15, 2.), (60811, 60811000, 5, 17, 8.),
            (60811, 60811000, 5, 18, 99.), (60811, 60811000, 6, 16, 90.),
            (61811, 60811000, 5, 16, 80.), (60811, 60811001, 5, 16, 70.)]
    arrays = arrays_from(rows)
    names = list(HISTORY_COLUMNS)
    arrays["X"] = np.zeros((len(rows), len(names)), np.float32)
    arrays["X"][:, 0] = [r[-1] for r in rows]
    features, bank = causal_history_features(arrays, names)
    assert np.isnan(features[1, bank.index("history_residual_lag_1")])
    assert features[1, bank.index("history_available_1")] == 0
    assert features[1, bank.index("history_residual_lag_2")] == 2
    assert features[1, bank.index("history_residual_change_2")] == 6
    # Alter future evidence and evaluator metadata: past features must not move.
    arrays["X"][2] = -12345
    arrays["labels"] = np.ones(len(rows))
    arrays["families"] = np.full(len(rows), 4)
    partial, _ = causal_history_features(arrays, names, rows=[1])
    np.testing.assert_equal(partial[0], features[1])
    # Query order and input order do not define elapsed hours.
    reversed_arrays = {k: v[::-1] for k, v in arrays.items()}
    reversed_features, _ = causal_history_features(reversed_arrays, names)
    np.testing.assert_equal(reversed_features[-2], features[1])


def test_presampling_keeps_original_expert_and_router_objectives():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from wdn.probe_residual_experts import ResidualExpertMixture
    from wdn.shared_history import fit_presampled

    class TinyMixture(ResidualExpertMixture):
        def _model(self):
            return HistGradientBoostingClassifier(max_iter=2, min_samples_leaf=2,
                                                   random_state=self.seed)

    rng = np.random.default_rng(7)
    names = ["residual", "abs_residual"]
    X = rng.normal(size=(120, 2))
    y = np.tile([0, 1], 60)
    families = np.tile([1, 2, 3, 4, 5], 24)
    original = TinyMixture(names, seed=701).fit(X, y, families)
    negatives = np.flatnonzero(y == 0)
    chosen = np.random.default_rng(701).choice(negatives, len(negatives), replace=False)
    subset = np.r_[np.flatnonzero(y), chosen]
    adapted = fit_presampled(TinyMixture(names, seed=701), X[subset], y[subset],
                             families[subset], np.ones(len(subset)))
    for key, expected in original.predict(X).items():
        np.testing.assert_allclose(adapted.predict(X)[key], expected)


def test_shared_history_is_available_to_every_existing_expert():
    from wdn.probe_residual_experts import feature_profiles
    from wdn.shared_history import SharedHistoryExpertMixture
    original = ["residual", "mean_4", "lag_advantage", "slope_8", "rms_16"]
    names = original + ["history_residual_lag_1", "history_available_1"]
    model = SharedHistoryExpertMixture(names)
    assert len(model.profiles) == 5
    for old, augmented in zip(feature_profiles(original), model.profiles):
        np.testing.assert_equal(augmented, np.r_[old, 5, 6])


def test_shared_history_rejects_ambiguous_duplicate_endpoints():
    from wdn.shared_history import HISTORY_COLUMNS, causal_history_features
    arrays = arrays_from([(811, 0, 5, 15, 1.), (811, 0, 5, 15, 2.)])
    arrays["X"] = np.zeros((2, len(HISTORY_COLUMNS)), np.float32)
    with pytest.raises(ValueError, match="Duplicate"):
        causal_history_features(arrays, list(HISTORY_COLUMNS))

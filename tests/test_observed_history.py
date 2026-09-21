import numpy as np
import pytest

from wdn.observed_history import observed_history_features


def test_exact_hours_missing_and_future_independence():
    times = np.array([0, 2, 3, 5])
    values = np.array([[2., 90.], [8., 80.], [99., 70.], [999., 60.]])
    mask = np.ones_like(values, bool)
    features, names = observed_history_features(values, mask, times, np.array([2]),
                                                np.array([0]), np.array([2.]))
    assert np.isnan(features[0, names.index("history_received_change_1")])
    assert features[0, names.index("history_received_available_1")] == 0
    assert features[0, names.index("history_received_change_2")] == 3
    values[2:] = -12345
    values[:, 1] = 99999
    changed, _ = observed_history_features(values, mask, times, np.array([2]),
                                           np.array([0]), np.array([2.]))
    np.testing.assert_equal(changed, features)
    reordered, _ = observed_history_features(values[::-1], mask[::-1], times[::-1],
                                              np.array([2]), np.array([0]), np.array([2.]))
    np.testing.assert_equal(reordered, features)
    mask[0, 0] = False
    missing, _ = observed_history_features(values, mask, times, np.array([2]),
                                           np.array([0]), np.array([2.]))
    assert missing[0, names.index("history_received_available_2")] == 0
    assert np.isnan(missing[0, names.index("history_received_change_2")])


def test_near_equal_values_share_a_bin_and_rounding_is_explicit():
    mask = np.ones((2, 1), bool)
    times = np.array([0, 1])
    def features(values, rounding=0.):
        return observed_history_features(np.array(values).reshape(2, 1), mask, times,
                                         np.array([1]), np.array([0]), np.array([1.]), rounding)[0]
    np.testing.assert_equal(features([10., 10.]), features([10., 10.03]))
    assert features([10., 10.21])[0, 0] == pytest.approx(.2)
    assert features([10.01, 10.24], .5)[0, 0] == 0


def test_invalid_or_ambiguous_inputs_rejected():
    values, mask = np.ones((2, 1)), np.ones((2, 1), bool)
    with pytest.raises(ValueError, match="duplicate"):
        observed_history_features(values, mask, np.array([1, 1]), np.array([1]),
                                   np.array([0]), np.array([1.]))
    with pytest.raises(ValueError, match="Positive"):
        observed_history_features(values, mask, np.array([0, 1]), np.array([1]),
                                   np.array([0]), np.array([0.]))


def test_corpus_adapter_isolates_sources_and_scenarios(monkeypatch):
    import sys
    from pathlib import Path
    experiment = str(Path(__file__).parents[1] / "thesis_v2/experiments/early_warning")
    monkeypatch.syspath_prepend(experiment)
    import screen_observed_history as runner

    class ReceivedData:
        def __init__(self, directory):
            self.source = int(directory.name)

        def scenario(self, sid):
            base = 10 * sid
            return {"values": np.array([[base], [base + self.source + sid]], float),
                    "mask": np.ones((2, 1), bool), "timestep": np.array([0, 1]),
                    "labels": "must never be used", "families": "must never be used"}

    monkeypatch.setattr(runner, "CampaignData", ReceivedData)
    arrays = {"X": np.ones((3, 1)), "source": np.array([1, 2, 1]),
              "scenario": np.array([1000, 2000, 1001]), "timestep": np.ones(3, int),
              "node": np.zeros(3, int), "labels": np.array([0, 1, 0])}
    manifest = {"pieces": [{"seed": s, "directory": str(s), "scenarios": [0, 1]}
                            for s in (1, 2)]}
    banks, _ = runner.build_bank(arrays, ["normal_error_scale"], manifest, (0.,))
    np.testing.assert_equal(banks[0.][:, 0], [1., 2., 2.])
    arrays["labels"] = 1 - arrays["labels"]
    unchanged, _ = runner.build_bank(arrays, ["normal_error_scale"], manifest, (0.,))
    np.testing.assert_equal(unchanged[0.], banks[0.])
    arrays["scenario"][0] = 1002
    with pytest.raises(ValueError, match="frozen TRAIN"):
        runner.build_bank(arrays, ["normal_error_scale"], manifest, (0.,))

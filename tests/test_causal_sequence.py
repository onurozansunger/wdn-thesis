import numpy as np

from wdn.models.causal_sequence import causal_row_ids, causal_score_memory


def test_causal_windows_never_include_future_and_mark_missing_hours():
    arrays = {"scenario": np.array([1, 1, 1]), "node": np.array([2, 2, 2]),
        "timestep": np.array([3, 5, 6])}
    rows = causal_row_ids(arrays, 4)
    np.testing.assert_array_equal(rows[1], [-1, 0, -1, 1])
    before = rows[:2].copy()
    arrays["timestep"][2] = 100
    np.testing.assert_array_equal(causal_row_ids(arrays, 4)[:2], before)


def test_causal_windows_can_materialise_only_selected_targets():
    arrays = {"scenario": np.array([0, 0, 0, 1]),
              "node": np.array([2, 2, 2, 2]),
              "timestep": np.array([0, 1, 3, 0])}
    rows = causal_row_ids(arrays, 3, np.array([2, 3]))
    assert rows.tolist() == [[1, -1, 2], [-1, -1, 3]]


def test_causal_score_memory_only_propagates_from_past_with_gap_decay():
    score = np.array([.8, .1, .2])
    result = causal_score_memory(score, np.ones(3), np.array([1, 2, 5]),
                                 np.ones(3), half_life=2)
    np.testing.assert_allclose(result, [.8, .8/np.sqrt(2), .2])
    changed = score.copy(); changed[-1] = .99
    np.testing.assert_allclose(causal_score_memory(changed, np.ones(3),
        np.array([1, 2, 5]), np.ones(3), 2)[:2], result[:2])

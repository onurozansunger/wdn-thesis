import numpy as np
import pytest
from wdn.received_trajectory import received_trajectory_features
from wdn.observed_history import observed_history_features


def extract(values, mask=None, times=None, query=6, node=0, rounding=0.):
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    mask = np.ones_like(values, bool) if mask is None else mask
    times = np.arange(len(values)) if times is None else times
    return received_trajectory_features(values, mask, times, np.array([query]),
                                        np.array([node]), np.array([1.]), rounding)


def test_exact_expected_sequence_statistics_and_point_compatibility():
    v = np.arange(9.)
    x, names = extract(v)
    assert x.shape == (1, 182) and len(set(names)) == 182
    base, _ = observed_history_features(v[:, None], np.ones((9, 1), bool),
        np.arange(9), np.array([6]), np.array([0]), np.ones(1))
    np.testing.assert_equal(x[:, :42], base)
    for stat, expected in [('median_signed', 2), ('median_abs', 2), ('mad', 0), ('pair_count', 3), ('newest_age', 0)]:
        assert x[0, names.index(f'history_trajectory_{stat}_lag2_w3')] == expected


def test_missing_exact_timestamps_and_single_pair_support():
    x, names = extract([0., 2., 8.], times=np.array([0, 2, 6]))
    idx = names.index('history_trajectory_pair_count_lag4_w3')
    assert x[0, idx] == 1
    assert np.isnan(x[0, idx - 1]) and np.isnan(x[0, idx - 2])
    assert x[0, idx + 1] == 0
    idx = names.index('history_trajectory_pair_count_lag1_w3')
    assert x[0, idx] == 0 and np.isnan(x[0, idx + 1])


def test_masked_values_other_sensors_future_and_order_do_not_change_features():
    v = np.column_stack((np.arange(10.), np.arange(10.) + 20))
    mask = np.ones_like(v, bool)
    mask[2, 0] = False
    x, _ = extract(v, mask)
    changed = v.copy()
    changed[2, 0] = np.nan
    changed[7:] = 1e10
    changed[:, 1] = -1e10
    y, _ = extract(changed, mask)
    np.testing.assert_equal(x, y)
    z, _ = extract(changed[::-1], mask[::-1], np.arange(10)[::-1])
    np.testing.assert_equal(x, z)


def test_near_equal_pairs_are_quantized_before_statistics():
    x, _ = extract(np.ones(9) * 10)
    v = np.ones(9) * 10
    v[6] += .02
    y, _ = extract(v)
    np.testing.assert_equal(x, y)
    v[6] += .2
    z, _ = extract(v)
    assert not np.array_equal(y, z, equal_nan=True)


def test_missing_current_endpoint_is_rejected():
    mask = np.ones((9, 1), bool)
    mask[6] = False
    with pytest.raises(ValueError, match='observed'):
        extract(np.arange(9.), mask)

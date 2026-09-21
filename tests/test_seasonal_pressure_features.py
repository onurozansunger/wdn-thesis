import numpy as np

from wdn.seasonal_pressure_features import seasonal_pressure_features


def test_seasonal_features_are_past_only_and_respect_missing_history():
    values = np.tile(np.arange(60, dtype=float)[:, None], (1, 2))
    observed = np.ones_like(values, dtype=bool)
    features, names = seasonal_pressure_features(values, observed, noise_sigma=1.)
    assert len(names) == features.shape[-1] == 7
    np.testing.assert_allclose(features[24, :, 0], 24 / np.sqrt(2))
    before = features[:40].copy()
    changed = values.copy(); changed[50:] += 1000
    np.testing.assert_array_equal(seasonal_pressure_features(changed, observed, 1.)[0][:40], before)
    missing = observed.copy(); missing[0, 0] = False
    result, _ = seasonal_pressure_features(values, missing, 1.)
    assert result[24, 0, 2] == 0 and result[24, 0, 0] == 0

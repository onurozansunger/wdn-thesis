import importlib
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'thesis_v2/experiments/early_warning'))
    # Other research drivers customize this shared module on import.
    # Restore its own definitions before testing trajectory calibration.
    importlib.reload(importlib.import_module('calibrate_received_trajectory'))
    return importlib.import_module('experiment_received_trajectory')


def test_negative_weighting_preserves_population_and_positive_weights(runner):
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    family = np.array([2, 3, 0, 1, 2, 3, 4, 5])
    w = np.array([1., 2., 10., 10., 10., 10., 10., 10.])
    result = runner.hard_negative_weights(y, family, w)
    np.testing.assert_equal(result[:2], w[:2])
    assert result[2:].sum() == pytest.approx(w[2:].sum())
    np.testing.assert_allclose(result[3:], result[2] * 3)
    np.testing.assert_equal(w, [1, 2, 10, 10, 10, 10, 10, 10])


def test_bank_adapter_uses_only_received_inputs_and_isolates_scenarios(runner, monkeypatch):
    class Received:
        def __init__(self, directory): self.seed = int(directory.name)
        def scenario(self, sid):
            return {'values': (np.arange(9.) * (self.seed + sid))[:, None],
                    'mask': np.ones((9, 1), bool), 'timestep': np.arange(9),
                    'labels': 'forbidden', 'clean_values': 'forbidden'}
    monkeypatch.setattr(runner, 'CampaignData', Received)
    a = {'labels': np.array([0, 1, 0]), 'source': np.array([1, 2, 1]),
         'scenario': np.array([1000, 2000, 1001]), 'timestep': np.array([6, 6, 6]),
         'node': np.zeros(3, int), 'X': np.ones((3, 1))}
    manifest = {'pieces': [{'seed': s, 'directory': str(s), 'scenarios': [0, 1]} for s in [1, 2]]}
    x, names = runner.build_bank(a, ['normal_error_scale'], manifest)
    np.testing.assert_equal(x[:, 0], [1, 2, 2])
    a['labels'] = 1 - a['labels']
    y, _ = runner.build_bank(a, ['normal_error_scale'], manifest)
    np.testing.assert_equal(x, y)
    a['scenario'][0] = 1002
    with pytest.raises(ValueError, match='outside'):
        runner.build_bank(a, ['normal_error_scale'], manifest)


def test_failed_train_rounding_blocks_calibration(runner, monkeypatch, tmp_path):
    import json
    cal = importlib.import_module('calibrate_received_trajectory')
    monkeypatch.setattr(cal, 'OUT', tmp_path)
    (tmp_path / 'train_selection.json').write_text(json.dumps({'selected': 'trajectory'}))
    (tmp_path / 'rounding_summary.json').write_text(json.dumps({'passes': False}))
    with pytest.raises(RuntimeError, match='prohibited'):
        cal.prerequisite()


def test_high_mean_cannot_hide_a_weak_calibration_source(runner, monkeypatch, tmp_path):
    import json
    cal = importlib.import_module('calibrate_received_trajectory')
    monkeypatch.setattr(cal, 'TARGET', tmp_path)
    monkeypatch.setattr(cal, 'prerequisite', lambda: 'trajectory')
    families = ['random', 'replay', 'drift', 'noise', 'targeted']
    baseline = {f: {'f1': .70} for f in families}
    mean = {f: {'f1': .81} for f in families}
    mean['_overall'] = {'f1': .85, 'clean_fpr': .0002}
    source = {str(s): {f: {'f1': .83} for f in families} for s in range(4)}
    source['0']['replay']['f1'] = .77
    for seed in cal.SEEDS:
        folder = tmp_path / str(seed)
        folder.mkdir()
        (folder / 'selection.json').write_text(json.dumps({'status': 'selected', 'baseline': baseline,
            'report': mean, 'source_reports': source, 'baseline_sources': {str(s): baseline for s in range(4)}}))
    cal.aggregate()
    decision = json.loads((tmp_path / 'decision.json').read_text())
    assert decision['mean_target_reached']
    assert not decision['confirmation_ready']


def test_rounding_report_serializes_and_requires_every_source(runner, monkeypatch, tmp_path):
    import json
    rounding = importlib.import_module('check_trajectory_rounding')
    monkeypatch.setattr(rounding, 'OUT', tmp_path)
    for fold in range(5):
        record = {'candidate': 'trajectory', 'reference': {'replay': {'f1': .7}},
                  'rounded': {r: {'replay': {'f1': .72}} for r in ('0.01', '0.05')}}
        (tmp_path / f'rounding_fold_{fold}.json').write_text(json.dumps(record))
    rounding.aggregate()
    assert json.loads((tmp_path / 'rounding_summary.json').read_text())['passes'] is True
    record['rounded']['0.05']['replay']['f1'] = .69
    (tmp_path / 'rounding_fold_4.json').write_text(json.dumps(record))
    rounding.aggregate()
    assert json.loads((tmp_path / 'rounding_summary.json').read_text())['passes'] is False

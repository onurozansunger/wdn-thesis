"""Calibration and information-boundary checks for the thesis supplement."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1]/'thesis_v2/experiments/early_warning'
sys.path.insert(0,str(EXPERIMENTS))
spec = importlib.util.spec_from_file_location('single_model_baseline',EXPERIMENTS/'single_model_baseline.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def test_exact_sweep_matches_brute_force_with_ties_and_budgets():
    rng = np.random.default_rng(492)
    for _ in range(8):
        families = np.tile(np.arange(6),30)
        labels = rng.integers(0,2,len(families)); labels[families==0] = 0
        arrays = {'labels':labels,'families':families}
        scores = rng.integers(0,11,len(families)).astype(float)/10
        thresholds = [float(scores.max()), *np.nextafter(np.unique(scores),-np.inf)]
        for cap in (0.,.05,.5,1.):
            actual = baseline.choose_thresholds(scores,arrays,cap)
            reports = [baseline.report(scores,t,arrays) for t in thresholds]
            for objective in baseline.OBJECTIVES:
                def key(r):
                    o=r['_overall']; p=o['f1']; w=o['worst_family_f1']
                    return (p,w,-o['clean_fpr']) if objective=='pooled' else (w,p,-o['clean_fpr'])
                feasible = [r for r in reports if r['_overall']['clean_fpr']<=cap]
                assert key(actual[objective]['calibration']) == max(map(key,feasible))


def test_forward_features_ignore_labels_and_do_not_cross_series():
    from wdn.delayed_decision_features import FORWARD_COLUMNS, delayed_decision_features
    names = list(FORWARD_COLUMNS)
    a = {'X':np.tile(np.arange(5,dtype=np.float32)[:,None],(1,len(names))),
         'source':np.array([1,1,1,1,2]),'scenario':np.array([1,1,1,2,1]),
         'node':np.array([7,7,7,7,7]),'timestep':np.array([0,2,4,2,2]),
         'labels':np.array([0,1,0,1,0]),'families':np.array([0,3,0,4,0])}
    first, columns = delayed_decision_features(a,names,3)
    altered = dict(a,labels=1-a['labels'],families=np.full(5,5))
    second,_=delayed_decision_features(altered,names,3)
    np.testing.assert_equal(first,second)
    assert np.isnan(first[0,columns.index('residual_t+1')])
    assert first[0,columns.index('residual_t+2')] == 1.
    assert np.isnan(first[0,columns.index('residual_t+3')])
    changed = {**a,'X':a['X'].copy()}; changed['X'][2:, :] = 999.
    third,_=delayed_decision_features(changed,names,3)
    np.testing.assert_equal(first[0],third[0])

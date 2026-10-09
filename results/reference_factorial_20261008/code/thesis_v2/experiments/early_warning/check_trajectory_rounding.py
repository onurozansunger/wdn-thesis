"""Predeclared post-selection sensitivity checks; no model/rule reselection."""
import argparse
import json
import joblib
import numpy as np

from experiment_received_trajectory import OUT, FOLDS, MANIFEST, build_bank
from build_feature_cache import write_json, sha
from screen_shared_history import load, predict_chunks
from wdn.latency_deployment import family_scores


def run(fold):
    selection = json.loads((OUT / 'train_selection.json').read_text())
    arm = selection['selected']
    if arm is None:
        raise RuntimeError('No candidate qualified; sensitivity is not a new search')
    a = load(FOLDS / f'fold_{fold}/features_held_out.npz')
    manifest = json.loads(MANIFEST.read_text())
    names = manifest['feature_names']
    target = OUT / f'fold_{fold}/seed_701'
    summary = json.loads((target / 'summary.json').read_text())
    threshold = summary['thresholds'][arm]['mixture']['threshold']
    model = joblib.load(target / f'{arm}.joblib')
    rows = a['scenario'] % 2 == 1
    result = {'fold': fold, 'seed': 701, 'candidate': arm, 'fixed_threshold': threshold,
              'reference': summary['results']['reference']['mixture'],
              'unrounded': summary['results'][arm]['mixture'], 'rounded': {},
              'scope': 'Only additional received-feature path rounded; canonical data and original features unchanged',
              'selection_sha256': sha(OUT / 'train_selection.json')}
    for resolution in (.01, .05):
        bank, _ = build_bank(a, names, manifest, resolution)
        if arm == 'hard_negative': bank = bank[:, :42]
        prediction = predict_chunks(model, a['X'], bank)['mixture']
        result['rounded'][str(resolution)] = family_scores(prediction[rows] > threshold,
                                                         {k: a[k][rows] for k in ('labels', 'families')})
    write_json(OUT / f'rounding_fold_{fold}.json', result)


def aggregate():
    rows = [json.loads((OUT / f'rounding_fold_{f}.json').read_text()) for f in range(5)]
    checks = {}
    for resolution in ('.01', '.05'):
        key = str(float(resolution))
        delta = [r['rounded'][key]['replay']['f1'] - r['reference']['replay']['f1'] for r in rows]
        checks[key] = {'source_deltas': delta, 'mean_delta': float(np.mean(delta)),
                       'passes': bool(min(delta) >= 0 and np.mean(delta) > 0)}
    report = {'passes': all(r['passes'] for r in checks.values()), 'checks': checks,
              'candidate': rows[0]['candidate'], 'model_seeds': [701], 'source_count': 5,
              'no_reselection': True}
    write_json(OUT / 'rounding_summary.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=range(5))
    args = parser.parse_args()
    aggregate() if args.fold is None else run(args.fold)

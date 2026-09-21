"""Audit existing TRAIN errors only; does not define or fit a detector."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_replay_train import FOLDS, reference_dir
from build_feature_cache import CAMPAIGN, ROOT, sha, write_json
from replicate_observed_history import frozen_write
from wdn.latency_deployment import family_scores, quantile_threshold

OUT = CAMPAIGN / 'replay_target_075_v1'
ARMS = ('reference', 'trajectory', 'flow')


def run():
    OUT.mkdir(exist_ok=True)
    frozen_write(OUT / 'audit_protocol.json', {
        'scope': 'Previously used source-held TRAIN only; descriptive error overlap, not independent evidence',
        'arms': ARMS, 'folds': list(range(5)), 'seeds': list(range(701, 706)),
        'operating_points': ['unchanged saved TRAIN-selected thresholds', 'fixed clean FPR budget 0.0005'],
        'partition': 'Existing even TRAIN scenarios select thresholds; odd scenarios supply diagnostics',
        'interpretation': 'Cross-model recovery counts are not an admissible combined detector or attainable F1. No score combination or model selection is performed.',
        'script_sha256': sha(Path(__file__)),
    })
    records, hashes = [], {}
    for fold in range(5):
        data_path = FOLDS / f'fold_{fold}/features_held_out.npz'
        hashes[str(data_path.relative_to(ROOT))] = sha(data_path)
        with np.load(data_path) as z:
            a = {k: z[k] for k in ('labels', 'families', 'scenario', 'source')}
        selection = a['scenario'] % 2 == 0
        diagnostic = ~selection
        scope = {k: a[k][diagnostic] for k in ('labels', 'families')}
        positive = scope['labels'] > 0
        replay = scope['families'] == 2
        clean = scope['families'] == 0
        for seed in range(701, 706):
            folders = {
                'reference': reference_dir(fold, seed),
                'trajectory': CAMPAIGN / f'received_trajectory_v1/fold_{fold}/seed_{seed}',
                'flow': CAMPAIGN / f'flow_fusion_v1/fold_{fold}/seed_{seed}',
            }
            reports = {}
            for arm in ('trajectory', 'flow'):
                path = folders[arm] / 'summary.json'
                hashes[str(path.relative_to(ROOT))] = sha(path)
                reports[arm] = json.loads(path.read_text())
            decisions = {'saved': {}, 'matched_budget': {}}
            for arm in ARMS:
                name = 'observed_history' if arm == 'reference' else arm
                path = folders[arm] / f'{name}_predictions.npz'
                digest = sha(path)
                hashes[str(path.relative_to(ROOT))] = digest
                if arm != 'reference':
                    expected = json.loads((folders[arm] / 'artifacts_sha256.json').read_text())
                    if digest != expected[path.name]:
                        raise RuntimeError(f'Prediction integrity failed: {path}')
                with np.load(path) as z:
                    score = z['mixture']
                if len(score) != len(selection) or not np.isfinite(score).all():
                    raise RuntimeError('Invalid or mis-sized cached predictions')
                report = reports['trajectory' if arm == 'reference' else arm]
                threshold = report['thresholds'][arm]['mixture']['threshold']
                decisions['saved'][arm] = score[diagnostic] > threshold
                got = family_scores(decisions['saved'][arm], scope)
                expected = report['results'][arm]['mixture']
                for family in ('replay', 'random', 'drift', 'noise', 'targeted'):
                    for key in ('tp', 'fp', 'fn', 'f1'):
                        if got[family][key] != expected[family][key]:
                            raise RuntimeError('Saved TRAIN result did not reproduce exactly')
                threshold = quantile_threshold(score, selection & (a['families'] == 0), .0005)
                decisions['matched_budget'][arm] = score[diagnostic] > threshold
            record = {'fold': fold, 'source': int(np.unique(a['source']).item()), 'seed': seed, 'operating_points': {}}
            for mode, d in decisions.items():
                counts = {'replay_positives': int((replay & positive).sum()),
                          'replay_negatives': int((replay & ~positive).sum()),
                          'clean_rows': int(clean.sum())}
                for arm in ARMS:
                    counts[f'{arm}_tp'] = int((replay & positive & d[arm]).sum())
                    counts[f'{arm}_fn'] = int((replay & positive & ~d[arm]).sum())
                    counts[f'{arm}_replay_fp'] = int((replay & ~positive & d[arm]).sum())
                    counts[f'{arm}_clean_fp'] = int((clean & d[arm]).sum())
                for left, right in (('trajectory', 'flow'), ('flow', 'trajectory')):
                    extra = d[right] & ~d[left]
                    prefix = f'{right}_beyond_{left}'
                    counts[f'{prefix}_replay_tp'] = int((extra & replay & positive).sum())
                    counts[f'{prefix}_replay_fp'] = int((extra & replay & ~positive).sum())
                    counts[f'{prefix}_clean_fp'] = int((extra & clean).sum())
                    counts[f'{prefix}_all_negative_fp'] = int((extra & ~positive).sum())
                counts['both_miss_replay'] = int((replay & positive & ~d['trajectory'] & ~d['flow']).sum())
                record['operating_points'][mode] = {'counts': counts}
            records.append(record)
        print(f'Completed TRAIN overlap fold {fold}', flush=True)
    summaries = {}
    for mode in ('saved', 'matched_budget'):
        by_source = []
        for fold in range(5):
            rows = [r['operating_points'][mode]['counts'] for r in records if r['fold'] == fold]
            by_source.append({'fold': fold, 'mean_counts_across_model_seeds': {
                key: float(np.mean([r[key] for r in rows])) for key in rows[0]}})
        counts = {key: float(np.mean([r['operating_points'][mode]['counts'][key] for r in records]))
                  for key in records[0]['operating_points'][mode]['counts']}
        summaries[mode] = {'mean_counts_per_run': counts, 'by_source': by_source,
            'flow_recovers_fraction_of_trajectory_misses': counts['flow_beyond_trajectory_replay_tp'] / max(1., counts['trajectory_fn']),
            'trajectory_recovers_fraction_of_flow_misses': counts['trajectory_beyond_flow_replay_tp'] / max(1., counts['flow_fn'])}
    write_json(OUT / 'train_complementarity.json', {
        'scope': 'Descriptive TRAIN General-only overlap. Seeds reuse sources. No combined detector, full-system, or independent performance claim.',
        'summaries': summaries, 'records': records, 'input_sha256': hashes,
        'audit_protocol_sha256': sha(OUT / 'audit_protocol.json'),
        'saved_metrics_exactly_reproduced': True,
    })
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    run()

"""Read-only replay error audit on existing held-source TRAIN predictions."""
from __future__ import annotations

import json
import numpy as np
from build_feature_cache import CAMPAIGN, ROOT, sha, write_json
from screen_shared_history import load, streams
from screen_observed_history import build_bank
from wdn.latency_deployment import family_scores, quantile_threshold

OUT = CAMPAIGN / 'received_trajectory_v1'
FOLDS = CAMPAIGN / 'stage_e_ltown/folds'


def reference_dir(fold, seed):
    if (fold, seed) == (0, 701):
        return CAMPAIGN / 'observed_history_pilot_v1/ltown/fold_0_seed_701'
    return CAMPAIGN / f'observed_history_replication_v1/fold_{fold}_seed_{seed}'


def run():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((CAMPAIGN / 'features/ltown_train/manifest.json').read_text())
    records = []
    inputs = {}
    for fold, piece in enumerate(manifest['pieces']):
        a = load(FOLDS / f'fold_{fold}/features_held_out.npz')
        assert set(np.unique(a['source'])) == {piece['seed']}
        inputs[str(FOLDS / f'fold_{fold}/features_held_out.npz')] = sha(FOLDS / f'fold_{fold}/features_held_out.npz')
        select = a['scenario'] % 2 == 0
        diag = ~select
        positive = a['labels'] > 0
        replay = (a['families'] == 2) & diag
        bank, _ = build_bank(a, manifest['feature_names'], manifest, (0.,))
        h = bank[0.]
        support = h[:, 2::3].sum(1)
        events = json.loads((ROOT / 'data/thesis_v2' / piece['directory'] / 'events.json').read_text())
        age = np.full(len(positive), -1)
        selected = np.zeros(len(positive), bool)
        archive_match = np.zeros(len(positive), bool)
        archive_available = np.zeros(len(positive), bool)
        for event_id, event in enumerate(events):
            rows = a['event'] == event_id
            age[rows] = a['timestep'][rows] - event['start_timestep']
            selected[rows] = np.isin(a['node'][rows], event['targets']['pressure'])
            if event['family'] == 'replay':
                lag = event['lag_steps']
                # Event lag is used only to annotate this offline audit.
                col = list(range(1, 13)).index(lag)
                archive_available[rows] = h[rows, 3 * col + 2] > 0
                archive_match[rows] = h[rows, 3 * col + 1] == 0
        groups = {
            'replay_positive': replay & positive,
            'replay_positive_age_0_2': replay & positive & (age < 3),
            'replay_positive_age_3_5': replay & positive & (age >= 3) & (age < 6),
            'replay_positive_age_6_plus': replay & positive & (age >= 6),
            'replay_positive_archive_bin_match': replay & positive & archive_match,
            'replay_positive_archive_no_match': replay & positive & ~archive_match,
            'replay_positive_support_0_3': replay & positive & (support <= 3),
            'replay_positive_support_4_plus': replay & positive & (support >= 4),
            'replay_positive_abs_residual_le_1': replay & positive & (a['X'][:, 1] <= 1),
            'replay_positive_abs_residual_gt_1': replay & positive & (a['X'][:, 1] > 1),
            'replay_selected_unchanged': replay & ~positive & selected,
            'replay_unselected_negative': replay & ~positive & ~selected,
            'clean': diag & (a['families'] == 0),
        }
        for seed in range(701, 706):
            path = reference_dir(fold, seed) / 'observed_history_predictions.npz'
            inputs[str(path)] = sha(path)
            p = load(path)
            decisions = {}
            thresholds = {}
            for name, score in streams(p).items():
                thresholds[name] = quantile_threshold(score, select & (a['families'] == 0), .0005)
                decisions[name] = score > thresholds[name]
            mixture = decisions['mixture']
            any_component = np.logical_or.reduce([decisions[n] for n in decisions if n.startswith('expert_')])
            miss = replay & positive & ~mixture
            record = {'fold': fold, 'seed': seed, 'source': piece['seed'],
                      'thresholds': thresholds,
                      'metrics': {n: family_scores(d[diag], {k: a[k][diag] for k in ('labels', 'families')}) for n, d in decisions.items()},
                      'groups': {n: {'rows': int(g.sum()), 'alarms': int((g & mixture).sum()),
                                     'any_component_alarms': int((g & any_component).sum())} for n, g in groups.items()},
                      'replay_misses': int(miss.sum()),
                      'misses_with_any_component_alarm': int((miss & any_component).sum()),
                      'misses_with_replay_component_alarm': int((miss & decisions['expert_replay']).sum()),
                      'positive_archive_available': int((replay & positive & archive_available).sum()),
                      'events': [], 'sensor_error_counts': []}
            for event_id in np.unique(a['event'][replay]):
                g = replay & (a['event'] == event_id)
                record['events'].append({'event': int(event_id), 'positive': int((g & positive).sum()),
                    'tp': int((g & positive & mixture).sum()), 'fp': int((g & ~positive & mixture).sum())})
            for sensor in np.unique(a['node'][replay & selected]):
                g = replay & (a['node'] == sensor)
                record['sensor_error_counts'].append({'sensor': int(sensor),
                    'fn': int((g & positive & ~mixture).sum()), 'fp': int((g & ~positive & mixture).sum())})
            records.append(record)
        print('audit completed TRAIN fold', fold, flush=True)
    means = {n: {metric: float(np.mean([r['metrics'][n]['replay'][metric] for r in records]))
                 for metric in ('f1', 'precision', 'recall')} for n in ('mixture', 'expert_replay')}
    groups = {n: {key: int(sum(r['groups'][n][key] for r in records))
                  for key in ('rows', 'alarms', 'any_component_alarms')} for n in records[0]['groups']}
    misses = sum(r['replay_misses'] for r in records)
    recoverable = sum(r['misses_with_any_component_alarm'] for r in records)
    replay_recoverable = sum(r['misses_with_replay_component_alarm'] for r in records)
    report = {'scope': 'Previously used TRAIN only; General diagnostic, not full-system or independent evidence',
              'fixed_clean_budget': .0005, 'selection': 'even scenarios', 'diagnostic': 'odd scenarios',
              'dependence': 'Five model seeds repeat endpoints; aggregated counts are repeated-decision counts',
              'means': means, 'groups': groups, 'misses': misses,
              'misses_with_any_component_alarm': recoverable,
              'misses_with_replay_component_alarm': replay_recoverable,
              'records': records, 'input_sha256': inputs}
    write_json(OUT / 'train_audit.json', report)
    lines = ['# Received-history replay TRAIN error audit', '', report['scope'], '',
        'All 25 fold/seed comparisons use the same 0.0005 clean-FPR budget selected on even held-source scenarios; odd scenarios are diagnostic. Counts below repeat endpoints across model seeds.', '',
        '| Stream | Mean replay F1 | Precision | Recall |', '|---|---:|---:|---:|']
    for n, m in means.items():
        lines.append(f"| {n} | {m['f1']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} |")
    lines += ['', f'Of {misses} missed replay decisions, {recoverable} ({recoverable/max(1, misses):.1%}) have an alarm from at least one component at its matched budget; {replay_recoverable} ({replay_recoverable/max(1, misses):.1%}) have an internal replay-component alarm. This is an optimistic diagnostic, not an admissible oracle detector or guaranteed router gain.', '',
              '| Group | Repeated rows | Mixture alarm rate |', '|---|---:|---:|']
    for n, g in groups.items():
        lines.append(f"| {n} | {g['rows']} | {g['alarms']/max(1,g['rows']):.6f} |")
    (OUT / 'TRAIN_AUDIT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('records', 'input_sha256')}, indent=2), flush=True)


if __name__ == '__main__':
    run()

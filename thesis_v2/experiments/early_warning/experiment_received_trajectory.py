"""Frozen TRAIN campaign: shared received trajectories and hard negatives.

This runner cannot load calibration or confirmation corpora. General-only
development findings require a separate protected full-system calibration.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from build_feature_cache import CAMPAIGN, ROOT, atomic_npz, sha, write_json, check_distribution, config_of
from audit_replay_train import reference_dir
from replicate_observed_history import frozen_write, validate_partition
from screen_shared_history import load, predict_chunks, streams, BUDGETS, select
from wdn.latency_deployment import FAMILY_NAMES, family_scores
from wdn.received_trajectory import received_trajectory_features, TRAJECTORY_WINDOWS, TRAJECTORY_STATS
from wdn.observed_history import OBSERVED_LAGS, DELTA_BIN_SIGMA
from wdn.run_expert_redesign import CampaignData
from wdn.shared_history import SharedHistoryExpertMixture, fit_presampled

OUT = CAMPAIGN / 'received_trajectory_v1'
FOLDS = CAMPAIGN / 'stage_e_ltown/folds'
MANIFEST = CAMPAIGN / 'features/ltown_train/manifest.json'
SEEDS = tuple(range(701, 706))
ARMS = ('reference', 'trajectory', 'hard_negative', 'combined')
HARD_NEGATIVE_FACTOR = 3.


def hard_negative_weights(labels, families, weights):
    """TRAIN-only weighting; all attack families treated symmetrically.

Keep sampled rows and total negative weight identical. No target sensor,
event identity, true lag, or attack severity enters the weighting rule.
"""
    w = np.array(weights, dtype=float, copy=True)
    negative = np.asarray(labels) <= .5
    w[negative & (np.asarray(families) > 0)] *= HARD_NEGATIVE_FACTOR
    w[negative] *= np.sum(np.asarray(weights)[negative]) / np.sum(w[negative])
    return w


def build_bank(a, names, manifest, rounding=0.):
    result = np.empty((len(a['labels']), 182), np.float32)
    filled = np.zeros(len(result), bool)
    col = names.index('normal_error_scale')
    found = None
    for piece in manifest['pieces']:
        source = np.flatnonzero(a['source'] == piece['seed'])
        if not len(source):
            continue
        data = CampaignData(ROOT / 'data/thesis_v2' / piece['directory'])
        for scenario in np.unique(a['scenario'][source]):
            sid = int(scenario - piece['seed'] * 1000)
            if sid not in piece['scenarios']:
                raise ValueError('Scenario outside frozen TRAIN manifest')
            rows = source[a['scenario'][source] == scenario]
            observed = data.scenario(sid)
            result[rows], found = received_trajectory_features(observed['values'], observed['mask'],
                observed['timestep'], a['timestep'][rows], a['node'][rows], a['X'][rows, col], rounding)
            filled[rows] = True
        del data
        gc.collect()
    if not filled.all():
        raise ValueError('Source absent from frozen TRAIN manifest')
    return result, found


def freeze():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    previous = json.loads((CAMPAIGN / 'observed_history_replication_v1/protocol_frozen.json').read_text())
    files = [Path(__file__), Path(__file__).with_name('audit_replay_train.py'), MANIFEST,
             OUT / 'train_audit.json', Path(__file__).with_name('screen_shared_history.py'),
             Path(__file__).with_name('build_feature_cache.py')]
    files += sorted((ROOT / 'src/wdn').rglob('*.py'))
    benchmark = config_of(ROOT / 'data/thesis_v2' / manifest['pieces'][0]['directory'])
    for piece in manifest['pieces']:
        d = ROOT / 'data/thesis_v2' / piece['directory']
        check_distribution(d, benchmark)
        files.extend(d / name for name in ('snapshots.pkl', 'corrupted.pkl', 'generate_config.yaml', 'events.json'))
    for fold in range(5):
        for name in ('features_train.npz', 'features_held_out.npz', 'reference.joblib'):
            p = FOLDS / f'fold_{fold}' / name
            if sha(p) != previous['input_and_code_hashes'][str(p.relative_to(ROOT))]:
                raise RuntimeError('Existing TRAIN fold changed')
            files.append(p)
        for seed in SEEDS:
            files.append(reference_dir(fold, seed) / 'observed_history_predictions.npz')
    protocol = {'arms': ARMS, 'folds': list(range(5)), 'seeds': SEEDS,
        'windows': TRAJECTORY_WINDOWS, 'lags': OBSERVED_LAGS, 'stats': TRAJECTORY_STATS,
        'min_pairs_for_distribution_stats': 2, 'delta_bin_sigma': DELTA_BIN_SIGMA,
        'feature_count': {'reference': 158, 'trajectory': 298},
        'hard_negative': {'factor': HARD_NEGATIVE_FACTOR, 'rule': 'negative TRAIN row in any attack family; normalize total negative weight; same sample and positive weights'},
        'sample': 'All positive rows plus same seed-specific 60000 sampled negatives; original population weights',
        'models': 'Same five General HGB components and multiclass internal router; original parameters and seeds',
        'threshold_budgets': BUDGETS,
        'threshold_selection': 'Existing even held-source TRAIN scenarios, max pooled F1 then worst family then lower FPR; independent per arm',
        'diagnostic': 'Existing odd held-source TRAIN scenarios; all 25 pairs before selection; reused development, not independent',
        'representation_selection': 'Among candidates with >=20/25 replay wins, positive replay delta in every source and mean replay delta >=0.01, choose highest mean worst-family F1, then pooled F1, then lower clean FPR. General other-family changes are diagnostic; final protection uses the complete system on calibration.',
        'rounding': 'For the single selected candidate, .01 and .05 m feature-path checks on all five folds at seed 701 with unchanged thresholds. Mean replay improvement and each source nonnegative required. No rounding-based reselection.',
        'full_system_stage': 'Select exactly one design on TRAIN, fit full TRAIN, only operating settings on calibration. No source-held full-system claim from full-TRAIN specialists. Apply original protected gates versus received-history reference. Reject if infeasible.',
        'confirmation_readiness': 'Every mean family >=0.78 and every source-mean family >=0.78 on calibration; >=0.80 separately defines target. One frozen confirmation only if protected and ready. No further feature tuning on calibration.',
        'protection': {'mean_other_family_max_drop': .005, 'source_other_family_max_drop': .01,
            'mean_pooled_max_drop': 0., 'source_pooled_max_drop': .005,
            'mean_clean_fpr_max_increase': 0., 'source_clean_fpr_max_increase': .0001, 'absolute_clean_fpr_max': .005},
        'architecture': ['General', 'Drift', 'Noise'], 'specialist_delay_hours': 3,
        'missing_pressure': .5, 'missing_flow': .5, 'generator_and_splits_unchanged': True,
        'locked_test_read': False, 'confirmation_read': False,
        'input_code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in files}}
    frozen_write(OUT / 'protocol_frozen.json', protocol)
    print('protocol frozen', flush=True)


def prepare(fold):
    target = OUT / f'fold_{fold}'
    target.mkdir(exist_ok=True)
    if (target / 'held_bank.npy').exists():
        return
    a = load(FOLDS / f'fold_{fold}/features_held_out.npz')
    manifest = json.loads(MANIFEST.read_text())
    bank, names = build_bank(a, manifest['feature_names'], manifest)
    np.save(target / 'held_bank.npy', bank)
    frozen_write(target / 'feature_names.json', manifest['feature_names'] + names)
    write_json(target / 'cache_signature.json', {'sha256': sha(target / 'held_bank.npy'),
        'protocol_sha256': sha(OUT / 'protocol_frozen.json'), 'rows': len(bank)})
    print('prepared fold', fold, flush=True)


def pair(fold, seed):
    target = OUT / f'fold_{fold}/seed_{seed}'
    target.mkdir(exist_ok=True)
    frozen_write(target / 'signature.json', {'protocol_sha256': sha(OUT / 'protocol_frozen.json'), 'fold': fold, 'seed': seed})
    if (target / 'summary.json').exists():
        return
    names = json.loads(MANIFEST.read_text())['feature_names']
    manifest = json.loads(MANIFEST.read_text())
    train = load(FOLDS / f'fold_{fold}/features_train.npz')
    with np.load(FOLDS / f'fold_{fold}/features_held_out.npz') as z:
        held = {k: z[k] for k in z.files if k != 'X'}
    validate_partition(train, held, manifest, fold)
    positives = np.flatnonzero(train['labels'] > .5)
    negatives = np.flatnonzero(train['labels'] <= .5)
    chosen = np.random.default_rng(seed).choice(negatives, min(60000, len(negatives)), replace=False)
    subset = np.r_[positives, chosen]
    weight = np.r_[np.ones(len(positives)), np.full(len(chosen), len(negatives) / len(chosen))]
    train = {k: v[subset] for k, v in train.items()}
    sample_sha = hashlib.sha256(subset.tobytes()).hexdigest()
    if (fold, seed) != (0, 701):
        expected = json.loads((reference_dir(fold, seed) / 'training_sample.json').read_text())
        assert sample_sha == expected['sample_sha256']
    write_json(target / 'training_sample.json', {'sha256': sample_sha, 'positive': len(positives),
        'negative': len(chosen), 'source': np.unique(train['source']).tolist()})
    del subset, positives, negatives, chosen
    gc.collect()
    bank, extra = build_bank(train, names, manifest)
    model = None
    for arm in ARMS[1:]:
        if (target / f'{arm}.joblib').exists():
            continue
        count = 42 if arm == 'hard_negative' else 182
        w = hard_negative_weights(train['labels'], train['families'], weight) if arm in ('hard_negative', 'combined') else weight
        model = SharedHistoryExpertMixture(names + extra[:count], seed)
        fit_presampled(model, np.column_stack((train['X'], bank[:, :count])), train['labels'], train['families'], w)
        joblib.dump(model, target / f'{arm}.joblib')
        print('fit', fold, seed, arm, flush=True)
    del train, bank, model
    gc.collect()
    held = load(FOLDS / f'fold_{fold}/features_held_out.npz')
    bank = np.load(OUT / f'fold_{fold}/held_bank.npy', mmap_mode='r')
    for arm in ARMS[1:]:
        path = target / f'{arm}_predictions.npz'
        if path.exists():
            continue
        model = joblib.load(target / f'{arm}.joblib')
        p = predict_chunks(model, held['X'], bank[:, :42] if arm == 'hard_negative' else bank)
        atomic_npz(path, **p)
        del p, model
    del held['X'], bank
    gc.collect()
    selection = held['scenario'] % 2 == 0
    diagnostic = ~selection
    scope = {k: held[k][diagnostic] for k in ('labels', 'families')}
    report = {'fold': fold, 'seed': seed, 'scope': 'TRAIN General-only development', 'results': {}, 'thresholds': {}}
    for arm in ARMS:
        path = reference_dir(fold, seed) / 'observed_history_predictions.npz' if arm == 'reference' else target / f'{arm}_predictions.npz'
        p = load(path)
        report['results'][arm], report['thresholds'][arm] = {}, {}
        for stream, score in streams(p).items():
            rule = select(score, held, selection)
            report['thresholds'][arm][stream] = rule
            m = family_scores(score[diagnostic] > rule['threshold'], scope)
            mask = diagnostic & np.isin(held['families'], (0, 2))
            m['replay']['auprc_family_plus_clean'] = float(average_precision_score(held['labels'][mask], score[mask]))
            report['results'][arm][stream] = m
    write_json(target / 'summary.json', report)
    write_json(target / 'artifacts_sha256.json', {p.name: sha(p) for p in target.iterdir() if p.name != 'artifacts_sha256.json'})
    print('completed pair', fold, seed, flush=True)


def aggregate():
    reports = [json.loads((OUT / f'fold_{f}/seed_{s}/summary.json').read_text()) for f in range(5) for s in SEEDS]
    means, deltas, eligible = {}, {}, []
    for arm in ARMS:
        means[arm] = {f: float(np.mean([r['results'][arm]['mixture'][f]['f1'] for r in reports])) for f in FAMILY_NAMES.values()}
        for metric, key in [('pooled', 'f1'), ('clean_fpr', 'clean_fpr'), ('worst_family', 'worst_family_f1')]:
            means[arm][metric] = float(np.mean([r['results'][arm]['mixture']['_overall'][key] for r in reports]))
        if arm == 'reference':
            continue
        delta = [r['results'][arm]['mixture']['replay']['f1'] - r['results']['reference']['mixture']['replay']['f1'] for r in reports]
        source = [float(np.mean([d for r, d in zip(reports, delta) if r['fold'] == f])) for f in range(5)]
        deltas[arm] = {'means': {k: means[arm][k] - means['reference'][k] for k in means[arm]},
                       'replay_source_means': source, 'replay_wins': sum(d > 0 for d in delta)}
        if sum(d > 0 for d in delta) >= 20 and min(source) > 0 and np.mean(delta) >= .01:
            eligible.append(arm)
    selected = max(eligible, key=lambda a: (means[a]['worst_family'], means[a]['pooled'], -means[a]['clean_fpr'])) if eligible else None
    report = {'means': means, 'deltas': deltas, 'eligible': eligible, 'selected': selected,
              'scope': 'General-only TRAIN development, not full-system or independent',
              'next': 'rounding checks then protected full-system calibration' if selected else 'stop: no repeatable >=0.01 General replay improvement',
              'protocol_sha256': sha(OUT / 'protocol_frozen.json')}
    frozen_write(OUT / 'train_selection.json', report)
    lines = ['# Received-trajectory TRAIN campaign', '', report['scope'], '',
             '| Arm | Replay | Pooled | Drift | Noise | Random | Targeted | Clean FPR |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, m in means.items():
        lines.append('| ' + arm + ' | ' + ' | '.join(f'{m[k]:.6f}' for k in ('replay', 'pooled', 'drift', 'noise', 'random', 'targeted', 'clean_fpr')) + ' |')
    lines += ['', f'Selected: {selected}. Next: {report["next"]}.', '', 'All 25 fold/model-seed pairs completed before selection. Source-held TRAIN folds have been used in development before and are not independent confirmation. Other-family protection must be assessed in the full system.']
    (OUT / 'TRAIN_REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def all_runs(workers):
    freeze()
    script = str(Path(__file__).resolve())
    def launch(args, log):
        with log.open('w') as f:
            subprocess.run([sys.executable, script, *args], stdout=f, stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(max_workers=min(workers, 3)) as pool:
        list(pool.map(lambda f: launch(['prepare', '--fold', str(f)], OUT / f'prepare_{f}.log'), range(5)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda pair: launch(['pair', '--fold', str(pair[0]), '--seed', str(pair[1])],
                         OUT / f'pair_{pair[0]}_{pair[1]}.log'), [(f, s) for f in range(5) for s in SEEDS]))
    aggregate()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('freeze', 'prepare', 'pair', 'aggregate', 'all'))
    p.add_argument('--fold', type=int, choices=range(5))
    p.add_argument('--seed', type=int, choices=SEEDS)
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    if args.action == 'all': all_runs(args.workers)
    elif args.action == 'freeze': freeze()
    elif args.action == 'prepare': prepare(args.fold)
    elif args.action == 'pair': pair(args.fold, args.seed)
    else: aggregate()

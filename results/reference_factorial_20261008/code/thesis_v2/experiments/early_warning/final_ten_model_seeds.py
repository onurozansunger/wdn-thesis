"""Ten-seed extension of the user-selected confirmed designs; no design search.

Reuses the original six evaluation sources per network. Each added seed refits
all stochastic full-system components using the original TRAIN/calibration code.
Evaluation is gated on freezing every added model and calibration rule.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_campaign as rc
import screen_router_temperature as st
import protected_history_system as ph
import ltown_setup
import joblib
import numpy as np
from build_feature_cache import ROOT, CAMPAIGN, sha, write_json, atomic_npz
from replicate_observed_history import frozen_write

EXT = CAMPAIGN / 'final_ten_model_seeds_v1'
OLD = tuple(range(701, 706))
NEW = tuple(range(706, 711))
ALL = OLD + NEW
METRICS = ('random', 'replay', 'drift', 'noise', 'targeted', 'pooled_f1', 'clean_fpr')
SOURCES = {'modena': tuple(rc.NETWORKS['modena']['eval_seeds']), 'ltown': ph.FRESH}
ORIGINAL_OUT = rc.out


def read(path):
    return json.loads(path.read_text())


def relative(path):
    return str(path.relative_to(ROOT))


def output(network, *parts):
    if parts and (str(parts[0]) in ('folds', 'reference.joblib') or
                  (str(parts[0]) == 'seed' and int(parts[1]) in OLD)):
        return ORIGINAL_OUT(network, *parts)
    return EXT / f'stage_e_{network}' / Path(*(str(p) for p in parts))


def configure():
    ltown_setup.register()
    rc.out = output
    # Avoid shared status-file races between seeds; logs are per worker.
    def status(network, phase, **details):
        print(time.strftime('%Y-%m-%dT%H:%M:%S'), network, phase, details, flush=True)
    rc.status = status
    st.CAMPAIGN = EXT
    st.OUTPUT = EXT / 'router_temperature_screen_v1'
    ph.OUT = EXT / 'protected_history_system_v1'
    ph.STAGE = EXT / 'stage_e_ltown/seed'
    ph.SELECTION = st.OUTPUT / 'ltown/seed'
    ph.SEEDS = NEW
    ph.VARIANTS = ('full_history',)
    ph.check_design = check_evaluation_freeze


def freeze():
    EXT.mkdir(parents=True, exist_ok=True)
    files = list((ROOT / 'src/wdn').rglob('*.py'))
    files += list(Path(__file__).parent.glob('*.py'))
    data = {}
    for network in SOURCES:
        for corpus in (f'{network}_train', f'{network}_calibration'):
            manifest_path = CAMPAIGN / 'features' / corpus / 'manifest.json'
            manifest = read(manifest_path)
            files.append(manifest_path)
            data[corpus] = manifest['pieces']
            for piece in manifest['pieces']:
                cache = ROOT / piece['cache']
                if sha(cache) != piece['cache_sha256']:
                    raise RuntimeError(f'Canonical cache changed: {cache}')
                files.append(cache)
                directory = ROOT / 'data/thesis_v2' / piece['directory']
                files += [directory / name for name in ('generate_config.yaml', 'corrupted.pkl', 'snapshots.pkl')]
        fold_root = ORIGINAL_OUT(network, 'folds')
        files += list(fold_root.rglob('*.json')) + list(fold_root.rglob('*.npz')) + list(fold_root.rglob('*.joblib'))
        for seed in OLD:
            folder = ORIGINAL_OUT(network, 'seed', seed)
            files += [folder / name for name in ('base_bundle.joblib', 'delayed_bundle.joblib', 'heads_bundle.joblib', 'operating_points.json')]
            if network == 'modena':
                files.append(folder / 'evaluation_report.json')
            else:
                folder = CAMPAIGN / 'protected_history_system_v1/seed' / str(seed)
                files += [folder / name for name in ('history.joblib', 'candidate_general.joblib', 'full_history_selection.json', 'confirmation.json')]
                files.append(CAMPAIGN / 'router_temperature_screen_v1/ltown/seed' / str(seed) / 'selection_frozen.json')
        for source in SOURCES[network]:
            corpus = f'modena_eval_seed{source}' if network == 'modena' else f'ltown_protected_history_confirmation_seed{source}'
            path = CAMPAIGN / 'features' / corpus / 'manifest.json'
            files.append(path)
            data[corpus] = read(path)['pieces']
    files += [CAMPAIGN / 'stage_e_ltown/reference.joblib', rc.reference_path('modena')]
    print('Hashing frozen code, TRAIN/calibration inputs, folds and original artifacts', flush=True)
    hashes = {relative(p): sha(p) for p in sorted(set(files))}
    protocol = {
        'selected_designs': {'modena': 'confirmed Stage-E candidate', 'ltown': 'confirmed full_history: original 42 received-pressure-history features; 158 General inputs'},
        'model_seeds': ALL, 'retained_seeds': OLD, 'added_seeds': NEW,
        'evaluation_source_seeds': SOURCES, 'evaluations_per_network': 60,
        'sequence': 'Freeze recipe; refit every stochastic component for 706–710; select with unchanged calibration algorithms; freeze ALL added models and rules across BOTH networks; evaluate the same six original source datasets; report all 10 seeds without selection.',
        'calibration': 'Original Stage-E selection for Modena. Original Stage-E, symmetric router recalibration, then original protected full_history grid for L-Town. No selective_history or other architecture search.',
        'failure_policy': 'If any seed has no feasible rule, retain and report that failure; do not drop/replace the seed, relax constraints, or inspect evaluation for tuning. No final ten-seed claim until all ten have valid results.',
        'independence': 'Extension across model randomness using the same six evaluation source datasets per network. The original five seeds were already confirmed. This is not a fresh independent confirmation and 60 pairs are not 60 independent datasets.',
        'architecture': {'top_level': ['General', 'Drift', 'Noise'], 'general_internal_components': ['general', 'abrupt', 'replay', 'drift', 'noise'], 'specialized_delay_hours': 3},
        'missing_pressure': .50, 'missing_flow': .50,
        'generator_distribution_severity_splits_unchanged': True,
        'locked_test_read': False, 'new_expert': False, 'replay_specific_inference_rule': False,
        'reporting': 'Equal-weight mean of the 60 model/source metrics, original five and added five separately; model-seed mean SD, source means, pair min/max; report failures and targets regardless of outcome.',
        'supersedes_selection': 'User selects confirmed full_history for final L-Town, superseding joint-model acceptance; prior results preserved.',
        'data_manifest_pieces': data, 'input_code_sha256': hashes,
        'versions': {'python': sys.version, 'numpy': np.__version__, 'joblib': joblib.__version__},
    }
    frozen_write(EXT / 'protocol_frozen.json', protocol)
    ph.OUT.mkdir(parents=True, exist_ok=True)
    frozen_write(ph.OUT / 'protocol_frozen.json', {'extension_protocol_sha256': sha(EXT / 'protocol_frozen.json'), 'variant': 'full_history'})
    print('Ten-model-seed extension protocol frozen', flush=True)


def verify_code():
    protocol = read(EXT / 'protocol_frozen.json')
    for name, digest in protocol['input_code_sha256'].items():
        if name.endswith('.py') and sha(ROOT / name) != digest:
            raise RuntimeError(f'Frozen code changed: {name}')


def oof_checkpointed(network, seed):
    """Same fold loop as stage_oof, with completed-fold prediction checkpoints."""
    folder = output(network, 'seed', seed)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'oof_scores.npz'
    if target.exists():
        return
    folds = sorted(output(network, 'folds').glob('fold_*'))
    assert len(folds) == (6 if network == 'modena' else 5)
    bank = read(CAMPAIGN / 'features' / rc.NETWORKS[network]['train'] / 'manifest.json')['feature_names']
    events = rc.events_for(network, rc.NETWORKS[network]['train'])
    paths = []
    for fold, directory in enumerate(folds):
        checkpoint = folder / f'oof_fold_{fold}.npz'
        paths.append(checkpoint)
        if checkpoint.exists():
            continue
        rc.status(network, 'fitting source-held experts', seed=seed, fold=fold)
        train = dict(np.load(directory / 'features_train.npz'))
        train['event'] = rc.globalise_events(train['event'], train['source'])
        train = rc.add_early_flag(train, events)
        experts = rc.fit_experts(train, bank, bank[:109], seed)
        head = rc.fit_delayed(train, bank, seed)
        del train
        gc.collect()
        held = dict(np.load(directory / 'features_held_out.npz'))
        held['event'] = rc.globalise_events(held['event'], held['source'])
        causal, delayed = rc.apply_experts(experts, head, held)
        mixture = experts['mixture'].predict(held['X'])['mixture']
        payload = {k: held[k] for k in ('labels', 'families', 'event', 'scenario', 'source', 'timestep', 'node')}
        atomic_npz(checkpoint, **payload, causal=causal, delayed=delayed, mixture=mixture)
        del held, experts, head, causal, delayed, mixture, payload
        gc.collect()
        rc.status(network, 'OOF fold checkpoint complete', seed=seed, fold=fold)
    parts = [dict(np.load(path)) for path in paths]
    atomic_npz(target, **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
    rc.status(network, 'oof complete', seed=seed)


def train(network, seed):
    assert seed in NEW
    verify_code()
    folder = output(network, 'seed', seed)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen_write(folder / 'extension_signature.json', {'network': network, 'seed': seed, 'protocol_sha256': sha(EXT / 'protocol_frozen.json')})
        oof_checkpointed(network, seed)
        # Original fit writes two files; detect an interrupted incomplete stage.
        if (folder / 'base_bundle.joblib').exists() and not (folder / 'fit_summary.json').exists():
            (folder / 'base_bundle.joblib').unlink()
        rc.stage_fit(network, seed)
        if (folder / 'heads_bundle.joblib').exists() and not (folder / 'heads_summary.json').exists():
            (folder / 'heads_bundle.joblib').unlink()
        rc.stage_heads(network, seed)
        rc.stage_calibrate(network, seed)
        points = read(folder / 'operating_points.json')
        if 'rule' not in points['arms']['candidate']:
            raise RuntimeError(f'{network} {seed}: original candidate calibration infeasible')
        if network == 'ltown':
            selection = st.screen(network, seed)
            if 'rule' not in selection:
                raise RuntimeError(f'L-Town {seed}: original router calibration infeasible')
            ph.fit(seed)
            selection = ph.selection(seed, 'full_history')
            if 'rule' not in selection:
                raise RuntimeError(f'L-Town {seed}: protected full_history calibration infeasible; seed retained as failure')
            history = joblib.load(ph.OUT / 'seed' / str(seed) / 'history.joblib')
            assert len(history.names) == 158 and len(history.experts) == 5
            bundle = {'names': history.names, 'experts': history.experts, 'profiles': history.profiles,
                      'router': history.router, 'router_columns': np.arange(len(history.names)),
                      'variant': 'full_history', 'top_level_branch': 'General', 'internal_component_count': 5}
            joblib.dump(bundle, ph.OUT / 'seed' / str(seed) / 'candidate_general.joblib')
        write_json(folder / 'training_complete.json', {'network': network, 'seed': seed, 'calibration_only': True, 'locked_test_read': False})


def model_files():
    files = []
    for network in SOURCES:
        for seed in NEW:
            folder = output(network, 'seed', seed)
            assert (folder / 'training_complete.json').exists(), (network, seed, 'not ready')
            files += [folder / name for name in ('base_bundle.joblib', 'delayed_bundle.joblib', 'heads_bundle.joblib', 'operating_points.json', 'fit_summary.json', 'heads_summary.json')]
            if network == 'ltown':
                files += [ph.SELECTION / str(seed) / 'selection_frozen.json']
                files += [ph.OUT / 'seed' / str(seed) / name for name in ('history.joblib', 'candidate_general.joblib', 'full_history_selection.json')]
    return files


def freeze_evaluation():
    verify_code()
    frozen_write(EXT / 'evaluation_frozen.json', {'protocol_sha256': sha(EXT / 'protocol_frozen.json'),
        'variant': 'full_history', 'all_added_seeds_calibrated_before_extension_evaluation': True,
        'model_rule_sha256': {relative(p): sha(p) for p in model_files()},
        'source_datasets_reused': True, 'locked_test_read': False})
    frozen_write(ph.OUT / 'design_frozen.json', {'variant': 'full_history', 'extension_evaluation_freeze_sha256': sha(EXT / 'evaluation_frozen.json')})


def check_evaluation_freeze():
    verify_code()
    record = read(EXT / 'evaluation_frozen.json')
    assert record['protocol_sha256'] == sha(EXT / 'protocol_frozen.json')
    for name, digest in record['model_rule_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f'Frozen model/rule changed: {name}')
    return record


def evaluate(network, seed):
    check_evaluation_freeze()
    assert seed in NEW
    if network == 'modena':
        rc.stage_evaluate(network, seed)
    else:
        ph.confirm(seed)


def report():
    check_evaluation_freeze()
    results = {}
    for network in SOURCES:
        rows = []
        for seed in ALL:
            if network == 'modena':
                folder = output(network, 'seed', seed)
                record = read(folder / 'evaluation_report.json')
                assert record['frozen_rules_sha256'] == sha(folder / 'operating_points.json')
                assert set(map(int, record['per_data_seed'])) == set(SOURCES[network])
                for source, entry in record['per_data_seed'].items():
                    rows.append({'model_seed': seed, 'source_seed': int(source), **{k: entry['arms']['candidate'][k] for k in METRICS}})
            else:
                folder = (CAMPAIGN / 'protected_history_system_v1' if seed in OLD else ph.OUT) / 'seed' / str(seed)
                record = read(folder / 'confirmation.json')
                assert record['variant'] == 'full_history'
                assert {p['source'] for p in record['pairs']} == set(SOURCES[network])
                for pair in record['pairs']:
                    row = {'model_seed': seed, 'source_seed': pair['source']}
                    for k in METRICS:
                        def val(arm):
                            r = pair[arm]
                            return r['_overall']['f1' if k == 'pooled_f1' else k] if k in ('pooled_f1', 'clean_fpr') else r[k]['f1']
                        row[k] = val('candidate')
                        row[f'{k}_baseline'] = val('baseline')
                    rows.append(row)
        assert len(rows) == 60
        def aggregate(selected):
            return {k: float(np.mean([r[k] for r in selected])) for k in METRICS}
        by_model = {str(s): aggregate([r for r in rows if r['model_seed'] == s]) for s in ALL}
        by_source = {str(s): aggregate([r for r in rows if r['source_seed'] == s]) for s in SOURCES[network]}
        result = {'rows': rows, 'mean': aggregate(rows),
                  'original_five_mean': aggregate([r for r in rows if r['model_seed'] in OLD]),
                  'added_five_mean': aggregate([r for r in rows if r['model_seed'] in NEW]),
                  'by_model_seed': by_model, 'by_source_seed': by_source,
                  'model_seed_mean_sd': {k: float(np.std([r[k] for r in by_model.values()], ddof=1)) for k in METRICS},
                  'pair_range': {k: [min(r[k] for r in rows), max(r[k] for r in rows)] for k in METRICS},
                  'pairs_below_0_80': {k: sum(r[k] < .8 for r in rows) for k in METRICS[:5]},
                  'replay_pairs_below_0_75': sum(r['replay'] < .75 for r in rows)}
        if network == 'ltown':
            result['paired_history_delta'] = {k: float(np.mean([r[k]-r[f'{k}_baseline'] for r in rows])) for k in METRICS}
            result['replay_wins'] = sum(r['replay'] > r['replay_baseline'] for r in rows)
        results[network] = result
    document = {'status': 'finalized', 'model_seeds': ALL, 'evaluation_sources': SOURCES,
                'evaluations_per_network': 60, 'independence': read(EXT / 'protocol_frozen.json')['independence'],
                'locked_test_read': False, 'production_deployment_performed': False,
                'protocol_sha256': sha(EXT / 'protocol_frozen.json'),
                'evaluation_freeze_sha256': sha(EXT / 'evaluation_frozen.json'), 'networks': results}
    frozen_write(EXT / 'final_results.json', document)
    lines = ['# Final Modena and L-Town results: ten model seeds', '',
             'Final designs: Modena confirmed Stage-E candidate; L-Town confirmed full received-pressure-history General model (42 history features, 158 inputs).', '',
             'Model seeds **701–710**, with **six source datasets per network**: 60 model/source evaluations each. Seeds 701–705 retain their original results; 706–710 refit every stochastic system component. All added models and calibration rules were frozen before extension evaluation.', '',
             '**This extends the existing confirmation across model randomness. It is not a new independent confirmation; the 60 evaluations share six datasets.** The locked test was not read. No production deployment was performed.', '',
             '| Metric | Modena confirmed | L-Town confirmed history |', '|---|---:|---:|']
    for k in METRICS:
        lines.append(f"| {k} | {results['modena']['mean'][k]:.6f} | {results['ltown']['mean'][k]:.6f} |")
    lines += ['', 'Means give equal weight to each model/source result. Modena F1 uses the original six-decimal saved-report convention.', '',
              '| Network | Original five replay | Added five replay | Ten-seed replay | Pooled F1 |', '|---|---:|---:|---:|---:|']
    for network, result in results.items():
        lines.append(f"| {network} | {result['original_five_mean']['replay']:.6f} | {result['added_five_mean']['replay']:.6f} | {result['mean']['replay']:.6f} | {result['mean']['pooled_f1']:.6f} |")
    lines += ['', 'The JSON includes every result, per-model and per-source means, model-seed variability, pair ranges and below-target counts. A mean above a target does not imply every run reached it.', '',
              'Architecture remains General / Drift / Noise, with five tree components inside General and the declared three-hour specialized delay. Pressure and flow missingness remain 0.50; severity, distribution and scenario splits were unchanged.', '',
              'The user-selected confirmed history model supersedes the earlier joint-model acceptance for final L-Town reporting; historical candidate records are preserved.', '',
              f"[Complete results]({EXT / 'final_results.json'})", f"[Frozen protocol]({EXT / 'protocol_frozen.json'})"]
    target = ROOT / 'thesis_v2/outputs/tables/final_modena_ltown_ten_model_seeds.md'
    target.write_text('\n'.join(lines) + '\n')
    print(json.dumps({n: r['mean'] for n, r in results.items()}, indent=2), flush=True)


def orchestrate(workers):
    EXT.mkdir(parents=True, exist_ok=True)
    with (EXT / 'runner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze()
        jobs = [(n, s) for s in NEW for n in SOURCES]
        def run_job(stage, network, seed):
            logdir = EXT / 'logs'
            logdir.mkdir(exist_ok=True)
            path = logdir / f'{stage}_{network}_{seed}.log'
            with path.open('a') as log:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--stage', stage, '--network', network, '--seed', str(seed)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            print(stage, network, seed, 'exit', result.returncode, flush=True)
            return {'stage': stage, 'network': network, 'seed': seed, 'returncode': result.returncode, 'log': relative(path)}
        for stage in ('train', 'evaluate'):
            if stage == 'evaluate':
                freeze_evaluation()
            write_json(EXT / 'status.json', {'status': 'running', 'stage': stage, 'model_seeds': ALL, 'locked_test_read': False})
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                completed = list(pool.map(lambda pair: run_job(stage, *pair), jobs))
            write_json(EXT / f'{stage}_jobs.json', completed)
            if any(job['returncode'] for job in completed):
                write_json(EXT / 'status.json', {'status': 'incomplete', 'stage': stage, 'jobs': completed, 'locked_test_read': False})
                raise RuntimeError(f'{stage}: failed jobs retained; inspect logs, no evaluation-based retuning')
        report()
        write_json(EXT / 'status.json', {'status': 'finalized', 'model_seeds': ALL, 'locked_test_read': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'freeze', 'train', 'freeze-evaluation', 'evaluate', 'report'), default='all')
    parser.add_argument('--network', choices=tuple(SOURCES))
    parser.add_argument('--seed', type=int, choices=NEW)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    configure()
    if args.stage == 'all': orchestrate(args.workers)
    elif args.stage == 'freeze': freeze()
    elif args.stage == 'train': train(args.network, args.seed)
    elif args.stage == 'freeze-evaluation': freeze_evaluation()
    elif args.stage == 'evaluate': evaluate(args.network, args.seed)
    else: report()

if __name__ == '__main__':
    main()

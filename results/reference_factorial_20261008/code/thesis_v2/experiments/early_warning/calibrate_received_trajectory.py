"""Protected full-system calibration after one TRAIN-selected design.

No confirmation sources or locked test are loaded by this runner.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import run_campaign as rc
from build_feature_cache import CAMPAIGN, ROOT, load_corpus, sha, write_json, atomic_npz, check_distribution, config_of
from experiment_received_trajectory import OUT, SEEDS, build_bank, hard_negative_weights
from replicate_observed_history import frozen_write
from screen_shared_history import load
from screen_router_temperature import TEMPERATURES, UNIFORM_SHRINKAGE, transformed_mixture
from protected_history_system import threshold_reports, protected, BUDGETS
from wdn.latency_deployment import family_scores, FAMILY_NAMES
from wdn.shared_history import SharedHistoryExpertMixture, fit_presampled

REFERENCE = CAMPAIGN / 'protected_history_system_v1/seed'
TARGET = OUT / 'full_system'


def prerequisite():
    selection = json.loads((OUT / 'train_selection.json').read_text())
    rounding = json.loads((OUT / 'rounding_summary.json').read_text())
    if not selection['selected'] or not rounding['passes']:
        raise RuntimeError('TRAIN progression gate failed; calibration access is prohibited')
    return selection['selected']


def freeze():
    arm = prerequisite()
    TARGET.mkdir(exist_ok=True)
    files = [OUT / n for n in ('protocol_frozen.json', 'train_selection.json', 'rounding_summary.json')]
    files += [Path(__file__), Path(__file__).with_name('check_trajectory_rounding.py')]
    files += [ROOT / p for p in json.loads((OUT / 'protocol_frozen.json').read_text())['input_code_sha256']]
    manifest_path = CAMPAIGN / 'features/ltown_calibration/manifest.json'
    files.append(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    original_protocol = CAMPAIGN / 'protected_history_system_v1/protocol_frozen.json'
    original_hashes = json.loads(original_protocol.read_text())['input_code_hashes']
    files.append(original_protocol)
    benchmark = config_of(ROOT / 'data/thesis_v2/ew_ltown_ltown_train_seed60811')
    for piece in manifest['pieces']:
        directory = ROOT / 'data/thesis_v2' / piece['directory']
        check_distribution(directory, benchmark)
        for name in ('snapshots.pkl', 'corrupted.pkl', 'generate_config.yaml'):
            path = directory / name
            if sha(path) != original_hashes[str(path.relative_to(ROOT))]:
                raise RuntimeError('Canonical calibration observations or configuration changed')
            files.append(path)
    for seed in SEEDS:
        files += [REFERENCE / str(seed) / n for n in ('history.joblib', 'full_history_selection.json')]
        for piece in manifest['pieces']:
            files += [ROOT / piece['cache'], REFERENCE / str(seed) / 'calibration' / f"source_{piece['seed']}.npz"]
    frozen_write(TARGET / 'protocol_frozen.json', {'arm': arm, 'training_seeds': SEEDS,
        'component_seeds': [600 + s for s in SEEDS], 'threshold_budgets': BUDGETS,
        'temperatures': TEMPERATURES, 'shrinkage': UNIFORM_SHRINKAGE,
        'reference': 'Confirmed full-history candidate with its saved calibration-only operating rule; same specialist decisions',
        'confirmation_gate': 'Protected on every seed; every source mean replay improves; each overall and source mean family F1 >=0.78. >=0.80 is target, .78-.80 near target. No confirmation generated here.',
        'code_input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in files}})


def fit(seed, arm):
    folder = TARGET / str(seed)
    folder.mkdir(exist_ok=True)
    if (folder / 'general.joblib').exists(): return
    a, manifest = load_corpus('ltown_train')
    names = a.pop('_feature_names')
    component_seed = 600 + seed
    positive = np.flatnonzero(a['labels'] > .5)
    negative = np.flatnonzero(a['labels'] <= .5)
    chosen = np.random.default_rng(component_seed).choice(negative, min(60000, len(negative)), replace=False)
    subset = np.r_[positive, chosen]
    weight = np.r_[np.ones(len(positive)), np.full(len(chosen), len(negative) / len(chosen))]
    a = {k: v[subset] for k, v in a.items()}
    del subset, positive, negative, chosen
    gc.collect()
    bank, added = build_bank(a, names, manifest)
    if arm == 'hard_negative': bank, added = bank[:, :42], added[:42]
    if arm in ('hard_negative', 'combined'): weight = hard_negative_weights(a['labels'], a['families'], weight)
    model = SharedHistoryExpertMixture(names + added, component_seed)
    fit_presampled(model, np.column_stack((a['X'], bank)), a['labels'], a['families'], weight)
    joblib.dump(model, folder / 'general.joblib')
    write_json(folder / 'fit.json', {'arm': arm, 'seed': seed, 'component_seed': component_seed,
        'features': model.names, 'sample_rows': len(weight), 'model_sha256': sha(folder / 'general.joblib')})


def score(seed, arm):
    folder = TARGET / str(seed)
    manifest = json.loads((CAMPAIGN / 'features/ltown_calibration/manifest.json').read_text())
    names = manifest['feature_names']
    model = joblib.load(folder / 'general.joblib')
    paths = []
    for piece in manifest['pieces']:
        target = folder / f"calibration_{piece['seed']}.npz"
        paths.append(target)
        if target.exists(): continue
        if sha(ROOT / piece['cache']) != piece['cache_sha256']:
            raise RuntimeError('Canonical calibration cache changed')
        a = load(ROOT / piece['cache'])
        old = load(REFERENCE / str(seed) / 'calibration' / f"source_{piece['seed']}.npz")
        for k in ('labels', 'families', 'scenario', 'source'):
            np.testing.assert_array_equal(a[k], old[k])
        bank, _ = build_bank(a, names, manifest)
        if arm == 'hard_negative': bank = bank[:, :42]
        # Match native full-system inference precision and the retained
        # reference cache; TRAIN screening's float32 storage is not reused.
        parts = []
        for start in range(0, len(a['labels']), 50000):
            chunk = model.predict(np.column_stack((a['X'][start:start + 50000], bank[start:start + 50000])))
            parts.append({k: chunk[k] for k in ('experts', 'routing')})
        p = {k: np.concatenate([part[k] for part in parts]) for k in ('experts', 'routing')}
        del parts
        payload = {k: old[k] for k in ('labels', 'families', 'scenario', 'source', 'specialist', 'history_experts', 'history_routing')}
        payload.update(candidate_experts=p['experts'], candidate_routing=p['routing'])
        atomic_npz(target, **payload)
        del a, old, bank, p, payload
        gc.collect()
        print(seed, 'scored calibration source', piece['seed'], flush=True)
    return paths


def select(seed, paths):
    folder = TARGET / str(seed)
    if (folder / 'selection.json').exists(): return
    parts = [load(p) for p in paths]
    a = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    reference = json.loads((REFERENCE / str(seed) / 'full_history_selection.json').read_text())
    old_rule = reference['rule']
    baseline_score = transformed_mixture(a['history_experts'], a['history_routing'], old_rule['router_temperature'], old_rule['uniform_shrinkage'])
    baseline = threshold_reports(baseline_score, [old_rule['mixture_threshold']], a)[0]
    for key, value in rc.compact(reference['report']).items():
        if abs(rc.compact(baseline)[key] - value) > 1e-8:
            raise RuntimeError('Received-history calibration reference did not reproduce exactly')
    baseline_decision = a['specialist'] | (baseline_score > old_rule['mixture_threshold'])
    source_rows = {int(s): np.flatnonzero(a['source'] == s) for s in np.unique(a['source'])}
    source_base = {s: family_scores(baseline_decision[rows], {k: a[k][rows] for k in ('labels', 'families')}) for s, rows in source_rows.items()}
    clean = a['families'] == 0
    best, frontier, feasible = None, None, 0
    for temperature in TEMPERATURES:
        for shrinkage in UNIFORM_SHRINKAGE:
            score = transformed_mixture(a['candidate_experts'], a['candidate_routing'], temperature, shrinkage)
            cs = np.sort(score[clean])
            thresholds = np.array([np.nextafter(cs[-1], np.inf) if b == 0 else cs[min(int(np.floor((1-b)*len(cs))), len(cs)-1)] for b in BUDGETS])
            for budget, threshold, report in zip(BUDGETS, thresholds, threshold_reports(score, thresholds, a)):
                key = (report['_overall']['worst_family_f1'], report['_overall']['f1'], -report['_overall']['clean_fpr'])
                if frontier is None or key > frontier[0]: frontier = (key, report)
                if not protected(report, baseline): continue
                decision = a['specialist'] | (score > threshold)
                sources = {s: family_scores(decision[rows], {k: a[k][rows] for k in ('labels', 'families')}) for s, rows in source_rows.items()}
                if not all(protected(sources[s], source_base[s], source=True) for s in sources): continue
                feasible += 1
                if best is None or key > best[0]:
                    rule = {**old_rule, 'router_temperature': temperature, 'uniform_shrinkage': shrinkage,
                            'mixture_clean_budget': float(budget), 'mixture_threshold': float(threshold)}
                    direct = family_scores(decision, a)
                    for f in FAMILY_NAMES.values():
                        assert abs(direct[f]['f1'] - report[f]['f1']) < 1e-12
                    assert abs(direct['_overall']['f1'] - report['_overall']['f1']) < 1e-12
                    best = (key, rule, report, sources)
    result = {'seed': seed, 'status': 'selected' if best else 'no_protected_operating_point',
              'baseline': baseline, 'baseline_sources': source_base, 'feasible_points': feasible,
              'unconstrained_frontier': frontier[1], 'baseline_reproduced_exactly': True}
    if best: result.update(rule=best[1], report=best[2], source_reports=best[3])
    frozen_write(folder / 'selection.json', result)
    print(seed, result['status'], rc.compact(result['report']) if best else '', flush=True)


def aggregate():
    results = [json.loads((TARGET / str(s) / 'selection.json').read_text()) for s in SEEDS]
    protected_all = all(r['status'] == 'selected' for r in results)
    report = {'protected_all_seeds': protected_all, 'confirmation_ready': False,
              'scope': 'Full-system calibration only; no new independent performance claim', 'arm': prerequisite()}
    report['reference_means'] = {f: float(np.mean([r['baseline'][f]['f1'] for r in results])) for f in FAMILY_NAMES.values()}
    if protected_all:
        means = {f: float(np.mean([r['report'][f]['f1'] for r in results])) for f in FAMILY_NAMES.values()}
        source_means = {s: {f: float(np.mean([r['source_reports'][s][f]['f1'] for r in results])) for f in FAMILY_NAMES.values()} for s in results[0]['source_reports']}
        source_improvement = {s: float(np.mean([r['source_reports'][s]['replay']['f1'] - r['baseline_sources'][s]['replay']['f1'] for r in results])) for s in source_means}
        report.update(means=means, source_means=source_means, source_replay_deltas=source_improvement,
            pooled=float(np.mean([r['report']['_overall']['f1'] for r in results])),
            clean_fpr=float(np.mean([r['report']['_overall']['clean_fpr'] for r in results])),
            mean_target_reached=min(means.values()) >= .80,
            confirmation_ready=min(means.values()) >= .78 and min(v for m in source_means.values() for v in m.values()) >= .78 and min(source_improvement.values()) > 0)
    report['next'] = 'freeze one fresh confirmation' if report['confirmation_ready'] else 'stop; retain confirmed received-history candidate'
    frozen_write(TARGET / 'decision.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('all', 'seed', 'aggregate'))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action == 'seed':
        arm = prerequisite(); fit(args.seed, arm); select(args.seed, score(args.seed, arm))
    elif args.action == 'aggregate': aggregate()
    else:
        freeze()
        def work(seed):
            with (TARGET / f'seed_{seed}.log').open('w') as f:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), 'seed', '--seed', str(seed)], stdout=f, stderr=subprocess.STDOUT, check=True)
        with ThreadPoolExecutor(max_workers=3) as pool: list(pool.map(work, SEEDS))
        aggregate()

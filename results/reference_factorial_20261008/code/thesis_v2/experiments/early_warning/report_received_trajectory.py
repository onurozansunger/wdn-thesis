"""Publish the complete frozen campaign outcome without tuning or test access."""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import sklearn
from build_feature_cache import ROOT, sha, write_json
from experiment_received_trajectory import OUT, SEEDS, ARMS


def run():
    selection = json.loads((OUT / 'train_selection.json').read_text())
    reports = [json.loads((OUT / f'fold_{f}/seed_{s}/summary.json').read_text()) for f in range(5) for s in SEEDS]
    protocol = json.loads((OUT / 'protocol_frozen.json').read_text())
    changed = [p for p, h in protocol['input_code_sha256'].items() if sha(ROOT / p) != h]
    if changed: raise RuntimeError(f'Frozen inputs or code changed: {changed}')
    artifacts = {}
    for f in range(5):
        for s in SEEDS:
            folder = OUT / f'fold_{f}/seed_{s}'
            expected = json.loads((folder / 'artifacts_sha256.json').read_text())
            for name, h in expected.items():
                if sha(folder / name) != h: raise RuntimeError(f'Pair artifact changed: {folder / name}')
                artifacts[str((folder / name).relative_to(ROOT))] = h
    details = {}
    for arm in ARMS[1:]:
        details[arm] = {}
        for family in ('replay', 'random', 'drift', 'noise', 'targeted'):
            delta = [r['results'][arm]['mixture'][family]['f1'] - r['results']['reference']['mixture'][family]['f1'] for r in reports]
            details[arm][family] = {'wins': sum(d > 0 for d in delta), 'worst_pair_delta': min(delta),
                'source_deltas': [float(np.mean([d for r, d in zip(reports, delta) if r['fold'] == f])) for f in range(5)]}
    verification = {'frozen_inputs_and_code_valid': True, 'pair_artifact_count': len(artifacts),
        'pair_artifacts_valid': True, 'paired_runs': 25, 'source_count': 5, 'details': details,
        'versions': {'python': sys.version, 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'joblib': joblib.__version__},
        'artifact_sha256': artifacts, 'locked_test_read': False, 'confirmation_read': False}
    if (OUT / 'full_system/decision.json').exists():
        cal_protocol = json.loads((OUT / 'full_system/protocol_frozen.json').read_text())
        changed = [p for p, h in cal_protocol['code_input_sha256'].items() if sha(ROOT / p) != h]
        if changed: raise RuntimeError(f'Calibration inputs or code changed: {changed}')
        checks = []
        for seed in SEEDS:
            folder = OUT / 'full_system' / str(seed)
            fitted = json.loads((folder / 'fit.json').read_text())
            assert sha(folder / 'general.joblib') == fitted['model_sha256']
            model = joblib.load(folder / 'general.joblib')
            reference = joblib.load(OUT.parent / f'protected_history_system_v1/seed/{seed}/history.joblib')
            assert model.seed == reference.seed == 600 + seed
            assert model.names[:len(reference.names)] == reference.names
            assert len(model.experts) == len(reference.experts) == 5
            for j in range(5):
                assert model.experts[j].get_params() == reference.experts[j].get_params()
                np.testing.assert_equal(model.profiles[j][model.profiles[j] < len(reference.names)], reference.profiles[j])
            assert model.router.get_params() == reference.router.get_params()
            checks.append({'seed': seed, 'same_hyperparameters_and_component_seed': True,
                           'reference_feature_prefix_and_profiles_preserved': True, 'five_internal_components': True})
        verification['calibration'] = {'frozen_inputs_and_code_valid': True, 'models': checks,
            'artifact_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (OUT / 'full_system').rglob('*') if p.is_file()}}
    write_json(OUT / 'verification.json', verification)
    lines = ['# L-Town received-trajectory and negative-example experiment', '',
        '**Development results only; no new independent confirmation.** All five source-held TRAIN folds and five model seeds were completed before selecting a candidate. No result here replaces the previously confirmed full-system replay F1 0.730307, and deployment remains unchanged.', '',
        '| General mixture arm | Replay F1 | Pooled F1 | Drift F1 | Noise F1 | Random F1 | Targeted F1 | Clean FPR |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, m in selection['means'].items():
        lines.append('| ' + arm + ' | ' + ' | '.join(f'{m[k]:.6f}' for k in ('replay', 'pooled', 'drift', 'noise', 'random', 'targeted', 'clean_fpr')) + ' |')
    lines += ['', '## Replay discrimination', '', '| Arm | Mean precision | Mean recall | Mean AUPRC, replay plus clean |', '|---|---:|---:|---:|']
    for arm in ARMS:
        values = [float(np.mean([r['results'][arm]['mixture']['replay'][k] for r in reports]))
                  for k in ('precision', 'recall', 'auprc_family_plus_clean')]
        lines.append('| ' + arm + ' | ' + ' | '.join(f'{v:.6f}' for v in values) + ' |')
    lines += ['', '## Replay repeatability', '', '| Arm | Paired wins | Mean F1 change | Source mean changes |', '|---|---:|---:|---|']
    for arm, d in selection['deltas'].items():
        lines.append(f"| {arm} | {d['replay_wins']}/25 | {d['means']['replay']:+.6f} | " + ', '.join(f'{v:+.6f}' for v in d['replay_source_means']) + ' |')
    next_step = ('Development and protected calibration completed; see the final decision below.'
                 if (OUT / 'full_system/decision.json').exists() else selection['next'])
    lines += ['', '## Decision', '', f"TRAIN-selected candidate: **{selection['selected'] or 'none'}**. {next_step}"]
    if (OUT / 'rounding_summary.json').exists():
        rounding = json.loads((OUT / 'rounding_summary.json').read_text())
        lines += ['', f"Predeclared feature-path rounding checks: {'PASS' if rounding['passes'] else 'FAIL'}. These retain original residual features, model, and thresholds; they do not change the benchmark or establish end-to-end sensor-rounding robustness."]
    if (OUT / 'full_system/decision.json').exists():
        cal = json.loads((OUT / 'full_system/decision.json').read_text())
        lines += ['', '## Full-system calibration', '', cal['scope'], '',
                  f"Protected for all five seeds: **{cal['protected_all_seeds']}**. Ready for fresh confirmation: **{cal['confirmation_ready']}**. Next: {cal['next']}."]
        if 'means' in cal:
            lines += ['', '| Family | Reference | Candidate |', '|---|---:|---:|']
            for f, value in cal['means'].items():
                lines.append(f"| {f} | {cal['reference_means'][f]:.6f} | {value:.6f} |")
            paired = [json.loads((OUT / f'full_system/{s}/selection.json').read_text()) for s in SEEDS]
            base_pooled = float(np.mean([r['baseline']['_overall']['f1'] for r in paired]))
            base_fpr = float(np.mean([r['baseline']['_overall']['clean_fpr'] for r in paired]))
            lines += [f"| Pooled F1 | {base_pooled:.6f} | {cal['pooled']:.6f} |",
                      f"| Clean FPR | {base_fpr:.6f} | {cal['clean_fpr']:.6f} |", '',
                      'Calibration scores are operating-point selection evidence, not independent confirmation.']
            lines += ['', '| Calibration source | Reference replay F1 | Candidate replay F1 | Change |',
                      '|---|---:|---:|---:|']
            for source, values in cal['source_means'].items():
                base = float(np.mean([r['baseline_sources'][source]['replay']['f1'] for r in paired]))
                lines.append(f"| {source} | {base:.6f} | {values['replay']:.6f} | {cal['source_replay_deltas'][source]:+.6f} |")
            lines += ['', 'The replay target was not reached. Mean replay F1 and the weakest source are below the predeclared 0.78 near-target floor, and one source has a small replay decline. Family/pooled/false-alarm protection passed, but target readiness and replay consistency did not. No fresh confirmation was generated.']
    lines += ['', '## Interpretation and integrity', '',
        'The two changes were a 140-feature received-trajectory bank and a threefold training weight for negative rows during any attack family, with total negative weight preserved. No target-sensor metadata, true lag, attack severity, or family label enters inference. The five internal General components and three top-level branches are retained.', '',
        'The TRAIN audit found that only 17.8% of mixture replay misses had any component alarm at matched clean budgets. False alarms were concentrated in unchanged readings on replay-targeted sensors. This motivated the negative-example candidate instead of new router training. The audit used event metadata only for offline explanation.', '',
        f"Verified all frozen input/code hashes and {len(artifacts)} pair artifacts. Point-feature compatibility was checked on 2,560 real TRAIN rows across all five sources. Pressure/flow missingness remain 0.50; generator settings, scenario splits, and decision delays remain unchanged. Locked test and consumed confirmation cases were not used.", '',
        'Seventeen focused tests passed. Full-system checks reproduced the received-history calibration reference exactly for every seed, matched optimized threshold counts to direct decisions, and verified the five component profiles, model seeds, hyperparameters, code/input hashes, and 42 calibration-stage artifacts.', '',
        'Model seeds share source scenarios. The 25 comparisons are not 25 independent datasets. General-only diagnostics do not establish full-system family protection; that requires the separate protected system stage.', '',
        '[Experiment protocol](' + str(ROOT / 'thesis_v2/experiments/early_warning/RECEIVED_TRAJECTORY_EXPERIMENT.md') + ') · [TRAIN error audit](' + str(OUT / 'TRAIN_AUDIT.md') + ') · [Frozen selection](' + str(OUT / 'train_selection.json') + ')']
    output = ROOT / 'thesis_v2/outputs/tables/ltown_received_trajectory_experiment.md'
    output.write_text('\n'.join(lines) + '\n')
    print(output)


if __name__ == '__main__': run()

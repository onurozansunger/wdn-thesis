"""Verify and report the completed frozen flow/fusion experiment."""
from pathlib import Path
import json
import joblib
import numpy as np
from build_feature_cache import ROOT,sha,write_json
from experiment_flow_fusion import OUT,SEEDS,EXCLUSIONS,manifest,FOLDS
from wdn.reliable_general_mixture import ReliableGeneralMixture


def verify():
    protocol=json.loads((OUT/'protocol_frozen.json').read_text())
    for path,digest in protocol['input_code_sha256'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Frozen code/input changed: '+path)
    downstream=json.loads((OUT/'downstream_code_frozen.json').read_text())
    for path,digest in downstream['code_sha256'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Prespecified downstream code changed: '+path)
    artifacts=0
    for piece in manifest()['pieces']:
        folder=OUT/'raw'/str(piece['seed'])
        for name,digest in json.loads((folder/'complete.json').read_text())['hashes'].items():
            assert sha(folder/name)==digest;artifacts+=1
    for first,second in EXCLUSIONS:
        folder=OUT/f'nested_{first}_{second}'
        fit=json.loads((folder/'fit_scope.json').read_text())['fit_sources']
        expected={p['seed'] for i,p in enumerate(manifest()['pieces']) if i not in (first,second)}
        assert set(fit)==expected
        for name,digest in json.loads((folder/'cache_hashes.json').read_text()).items():
            assert sha(folder/name)==digest;artifacts+=1
        for seed in SEEDS:
            done=json.loads((folder/f'seed_{seed}/complete.json').read_text())
            assert set(done['training_sources'])==expected
            assert not set(done['training_sources'])&set(done['excluded_sources'])
            for name,digest in done['hashes'].items():
                assert sha(folder/f'seed_{seed}'/name)==digest;artifacts+=1
    for fold in range(5):
        folder=OUT/f'fold_{fold}'
        for name,digest in json.loads((folder/'cache_hashes.json').read_text()).items():
            assert sha(folder/name)==digest;artifacts+=1
        for seed in SEEDS:
            target=folder/f'seed_{seed}'
            for name,digest in json.loads((target/'artifacts_sha256.json').read_text()).items():
                assert sha(target/name)==digest;artifacts+=1
            for arm in ('fusion','flow','combined'):
                model=joblib.load(target/f'{arm}.joblib')
                assert len(model.experts)==5
                assert len(model.names)==(158 if arm=='fusion' else 202)
                assert isinstance(model,ReliableGeneralMixture)==(arm!='flow')
                shared={i for i,n in enumerate(model.names) if n.startswith('history_')}
                for expert,columns in zip(model.experts,model.profiles):
                    assert shared <= set(columns)
                    assert expert.n_features_in_==len(columns)
                base=model.base if isinstance(model,ReliableGeneralMixture) else model
                assert base.router.n_features_in_==len(model.names)
    report={'protocol_inputs_verified':len(protocol['input_code_sha256']),
            'artifact_hashes_verified':artifacts,'nested_source_exclusion_verified':True,
            'outer_pairs':25,'teacher_feature_mode_fits':100,'architecture_verified':True}
    if (OUT/'full_system/decision.json').exists():
        cal=OUT/'full_system'
        frozen=json.loads((cal/'protocol_frozen.json').read_text())['input_code_sha256']
        for path,digest in frozen.items():
            assert sha(ROOT/path)==digest
        for seed in SEEDS:
            folder=cal/str(seed)
            fitted=json.loads((folder/'fit.json').read_text())
            assert sha(folder/'general.joblib')==fitted['model_sha256']
            selection=json.loads((folder/'selection.json').read_text())
            assert selection['baseline_reproduced_exactly']
        report['calibration_frozen_inputs_verified']=len(frozen)
        report['calibration_reference_exact_reproduction_seeds']=5
    if (OUT/'confirmation/summary.json').exists():
        folder=OUT/'confirmation'
        for path,digest in json.loads((folder/'design_frozen.json').read_text())['input_code_sha256'].items():assert sha(ROOT/path)==digest
        for path,digest in json.loads((folder/'data_manifest.json').read_text())['input_sha256'].items():assert sha(ROOT/path)==digest
        for seed in SEEDS:
            for name,digest in json.loads((folder/str(seed)/'artifacts_sha256.json').read_text()).items():assert sha(folder/str(seed)/name)==digest
        report['fresh_confirmation_verified']=True
    write_json(OUT/'verification.json',report)
    return report


def report():
    verification=verify()
    train=json.loads((OUT/'train_selection.json').read_text());selected=train['selected']
    confirmation_path=OUT/'confirmation/summary.json'
    confirmation=json.loads(confirmation_path.read_text()) if confirmation_path.exists() else None
    lines=['# Received flow and internal probability combination', '',
        'These are development findings. The independently confirmed received-pressure-history candidate remains replay F1 **0.730307** until a new independent confirmation is completed. No deployment change is implied.', '',
        '## Source-held TRAIN: General branch only', '',
        'All 25 outer fold/model-seed pairs completed before design selection. Model seeds reuse five source datasets and are not independent datasets. Teachers used nested source exclusions, including fitted normal references.', '',
        '| Arm | Replay F1 | Pooled F1 | Drift | Noise | Random | Targeted | Clean FPR |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,m in train['means'].items():
        lines.append('| '+arm+' | '+' | '.join(f'{m[k]:.6f}' for k in ('replay','pooled','drift','noise','random','targeted','clean_fpr'))+' |')
    lines += ['', '| Candidate | Replay paired gain | Positive pairs | Source mean replay gains |', '|---|---:|---:|---|']
    for arm,d in train['deltas'].items():
        lines.append(f"| {arm} | {d['means']['replay']:+.6f} | {d['replay_wins']}/25 | "+', '.join(f'{v:+.6f}' for v in d['replay_source_means'])+' |')
    lines += ['',f'TRAIN-selected candidate: **{selected}**.' if selected else 'No candidate passed the frozen TRAIN progression gate. This experiment stops before calibration and fresh confirmation.']
    rounding_path=OUT/'rounding_summary.json'
    if rounding_path.exists():
        rounding=json.loads(rounding_path.read_text())
        lines += ['',f"Pressure feature-path rounding checks: **{'PASS' if rounding['passes'] else 'FAIL'}**. These do not establish end-to-end sensor quantization robustness."]
    cal_path=OUT/'full_system/decision.json'
    cal=None
    if cal_path.exists():
        cal=json.loads(cal_path.read_text())
        lines += ['', '## Full-system calibration', '', 'These calibration results use a different corpus from independent confirmation and cannot be compared as a paired change from 0.730307.', '',
            f"Protection across all five seeds: **{'PASS' if cal['protected_all_seeds'] else 'FAIL'}**.",
            f"Confirmation readiness: **{'PASS' if cal['confirmation_ready'] else 'FAIL'}**."]
        if cal['protected_all_seeds']:
            lines += ['', '| Family | Reference F1 | Candidate F1 | Change |','|---|---:|---:|---:|']
            for f,v in cal['means'].items():
                b=cal['reference_means'][f];lines.append(f'| {f} | {b:.6f} | {v:.6f} | {v-b:+.6f} |')
            lines += ['', f"Candidate pooled F1: {cal['pooled']:.6f}; clean FPR: {cal['clean_fpr']:.8f}.", '',
                      '| Source | Replay F1 | Paired replay gain |','|---|---:|---:|']
            for s,m in cal['source_means'].items():lines.append(f"| {s} | {m['replay']:.6f} | {cal['source_replay_deltas'][s]:+.6f} |")
        lines += ['', cal['next']+'.']
    if confirmation:
        lines[2]='One fresh independent confirmation is complete. These results were not used to alter the design or operating settings. No deployment change is implied.'
        lines += ['', '## Fresh independent confirmation', '',
            'Six new generator sources, five model seeds per source. The model seeds share scenarios; these are not 30 independent datasets.', '',
            '| Metric | Paired reference | Candidate | Change |','|---|---:|---:|---:|']
        for k,v in confirmation['means']['candidate'].items():
            lines.append(f"| {k} | {confirmation['means']['baseline'][k]:.6f} | {v:.6f} | {confirmation['paired_deltas'][k]:+.6f} |")
        lines += ['',f"Protected improvement confirmed: **{confirmation['protected_improvement_confirmed']}**. All family means >=0.80: **{confirmation['all_family_mean_target_0_80']}**. All family means >=0.78: **{confirmation['all_family_mean_near_0_78']}**.",
            f"Replay wins: {confirmation['replay_wins']}/30; candidate replay range: {confirmation['replay_range'][0]:.6f} to {confirmation['replay_range'][1]:.6f}."]
    lines += ['', '## Interpretation and verification', '',
        'The flow arm adds 44 shared features based on actually received flows and pressures. The fusion arm calibrates five component probabilities and trains a regularized correction of internal routing using binary detection loss. The original three top-level branches and three-hour Drift/Noise delay remain.', '',
        f"Verified {verification['protocol_inputs_verified']} frozen code/input files and {verification['artifact_hashes_verified']} artifact hashes, all nested source exclusions and all 75 candidate model interfaces.", '',
        'Pressure and flow missingness remain 0.50. Generator distribution, severity and scenario splits are unchanged. No locked test or consumed independent confirmation cases were accessed by this experiment. Other-family protections apply to the complete system; General-only family changes must not be represented as deployed regressions or improvements.', '']
    target=ROOT/'thesis_v2/outputs/tables/ltown_flow_fusion_experiment.md';target.write_text('\n'.join(lines))
    stopped=selected is None or (rounding_path.exists() and not json.loads(rounding_path.read_text())['passes']) or (cal is not None and not cal['confirmation_ready'])
    state={'status':'complete_target_unmet' if stopped else 'progression_pending',
        'selected':selected,'train_pairs':25,'fresh_confirmation_run':False,'locked_test_read':False,
        'deployment_changed':False,'confirmed_candidate_replay_f1':.7303071947615193,'report':str(target)}
    if confirmation:
        achieved=confirmation['protected_improvement_confirmed'] and confirmation['all_family_mean_target_0_80']
        state.update(status='complete_target_reached' if achieved else 'complete_target_unmet',fresh_confirmation_run=True,
            new_independently_evaluated_replay_f1=confirmation['means']['candidate']['replay'])
        if confirmation['protected_improvement_confirmed']:state['confirmed_candidate_replay_f1']=confirmation['means']['candidate']['replay']
    write_json(OUT/'execution_state.json',state)
    print(json.dumps(state,indent=2))


if __name__=='__main__':report()

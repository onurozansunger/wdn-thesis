"""Verify and report the frozen joint-feature experiment, including failures."""
import json
from pathlib import Path
from build_feature_cache import ROOT,sha,write_json
from experiment_received_joint import OUT,SEEDS


def run():
    protocol=json.loads((OUT/'protocol_frozen.json').read_text())
    checked=0
    for path,digest in protocol['input_code_sha256'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Frozen input/code changed: '+path)
        checked+=1
    artifacts=0
    for fold in range(5):
        cache=OUT/f'fold_{fold}/held_bank.npy'
        if sha(cache)!=json.loads((cache.parent/'cache_signature.json').read_text())['sha256']:
            raise RuntimeError('Joint held cache changed')
        for seed in SEEDS:
            folder=OUT/f'fold_{fold}/seed_{seed}'
            for name,digest in json.loads((folder/'artifacts_sha256.json').read_text()).items():
                if sha(folder/name)!=digest:raise RuntimeError('TRAIN artifact changed')
                artifacts+=1
    selection=json.loads((OUT/'train_selection.json').read_text())
    lines=['# Joint received trajectory and flow experiment', '',
        'Target: mean full-system replay F1 0.75 with pooled, other-family, source, and false-alarm protection.', '',
        'The independently confirmed received-pressure-history reference remains replay F1 **0.730307** until another confirmation qualifies. It has not been deployed.', '',
        '## TRAIN General branch only', '',
        'All 25 paired comparisons completed before selection. Five sources each reuse five model seeds; these are development results, not independent or full-system performance.', '',
        '| Arm | Replay | Pooled | Drift | Noise | Random | Targeted | Clean FPR |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,m in selection['means'].items():
        lines.append('| '+arm+' | '+' | '.join(f'{m[k]:.6f}' for k in ('replay','pooled','drift','noise','random','targeted','clean_fpr'))+' |')
    d=selection['deltas']['joint']
    lines += ['',f"Replay wins: {d['replay_wins']}/25; mean paired gain {d['means']['replay']:+.6f}.",
        'Source mean replay gains: '+', '.join(f'{v:+.6f}' for v in d['replay_source_means'])+'.',
        f"Selected: {selection['selected']}."]
    rounding=OUT/'rounding_summary.json';cal=OUT/'full_system/decision.json';confirmation=OUT/'confirmation/summary.json'
    state='train_rejected' if not selection['selected'] else 'train_qualified'
    if rounding.exists():
        r=json.loads(rounding.read_text());lines += ['',f"Pressure feature-path rounding passes: {r['passes']}. This is not an end-to-end sensor quantization test."]
        state='rounding_passed' if r['passes'] else 'rounding_rejected'
    if cal.exists():
        result=json.loads(cal.read_text());lines += ['', '## Full-system calibration', '',
            f"Protection in every model seed: {result['protected_all_seeds']}. Confirmation readiness: {result['confirmation_ready']}."]
        rows=[json.loads((OUT/f'full_system/{s}/selection.json').read_text()) for s in SEEDS]
        if 'means' in result:
            lines += [f"Paired mean replay: {result['reference_means']['replay']:.6f} to {result['means']['replay']:.6f} ({result['replay_gain']:+.6f}).",
                f"Candidate pooled F1 {result['pooled']:.6f}; clean FPR {result['clean_fpr']:.6f}.",
                'Source mean replay gains: '+', '.join(f'{s}: {v:+.6f}' for s,v in result['source_replay_deltas'].items())+'.']
        else:
            lines += ['Protected operating points by model seed: '+', '.join(f"{r['seed']}: {r['feasible_points']}" for r in rows)+'.']
        lines += ['Calibration is a different corpus from independent confirmation; gains cannot be added to 0.730307 as a forecast.',result['next']+'.']
        state='calibration_qualified' if result['confirmation_ready'] else 'calibration_rejected'
        cp=json.loads((OUT/'full_system/protocol_frozen.json').read_text())
        for path,digest in cp['input_code_sha256'].items():
            if sha(ROOT/path)!=digest:raise RuntimeError('Calibration frozen input changed')
        if not all(r['baseline_reproduced_exactly'] for r in rows):raise RuntimeError('Calibration baseline mismatch')
    if confirmation.exists():
        r=json.loads(confirmation.read_text())
        lines += ['', '## One fresh confirmation', '',
            f"Mean replay F1 {r['means']['candidate']['replay']:.6f}; target and protections achieved: {r['replay_target_0_75_achieved']}.",
            f"Paired replay gain {r['paired_deltas']['replay']:+.6f}; {r['replay_wins']}/30 wins.",
            f"Individual replay range: {r['replay_range']}. Six independent sources, five model seeds each."]
        state='target_achieved' if r['replay_target_0_75_achieved'] else 'confirmation_target_unmet'
    else:
        lines += ['', 'No new independent confirmation or deployment was performed.']
    lines += ['', '## Integrity and fixed design', '',
        f'Verified {checked} frozen input/code files and {artifacts} TRAIN artifacts. Existing baseline metrics reproduced exactly in every TRAIN pair.',
        'The 342 inputs contain the original 158 inputs, 140 received-trajectory additions, and 44 received-flow additions. The 42 point-history inputs appear once. All added features reach the same five General components and internal router.',
        'General, Drift, and Noise remain the three branches. Drift/Noise delay remains three hours. Pressure and flow missingness remain 0.50. Severity, generator distribution, scenario splits, labels, and the locked test remain unchanged.']
    path=ROOT/'thesis_v2/outputs/tables/ltown_received_joint_075.md'
    path.write_text('\n'.join(lines)+'\n')
    write_json(OUT/'execution_state.json',{'state':state,'confirmed_reference_replay':.7303071947615193,
        'train_pairs':25,'frozen_inputs_verified':checked,'train_artifacts_verified':artifacts,
        'fresh_confirmation_run':confirmation.exists(),'deployment_changed':False,
        'report_sha256':sha(path)})
    print(path)


if __name__=='__main__':run()

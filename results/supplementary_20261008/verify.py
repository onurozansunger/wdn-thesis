#!/usr/bin/env python3
"""Verify the compact E10 scientific record using only the Python standard library.

This recomputes metrics/aggregates from published confusion-count JSON, verifies
published file bytes, and checks recorded provenance chains. It does not execute
models, access the locked test, or verify omitted score/model/telemetry bytes.
It cannot independently establish endpoint alignment, reproduce model inference,
or rerun the exact LightGBM threshold search without the omitted score arrays.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys

FAMILIES = ('random', 'replay', 'drift', 'noise', 'targeted')
SEEDS = tuple(range(701, 711))
NETWORKS = ('modena', 'ltown')
CAPS = (0., .0001, .00025, .0005, .001, .0025, .005)
CAMPAIGN = 'runs/operational/early_warning_multiseed_v1'
ORIGINAL_ANALYSIS = 'output/supervisor_revision_20261008/analysis'
TOL = 1e-12


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def close(actual, expected, where):
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f'{where}: keys differ')
        for key in expected:
            close(actual[key], expected[key], f'{where}/{key}')
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f'{where}: lengths differ')
        for i, (a, b) in enumerate(zip(actual, expected)):
            close(a, b, f'{where}/{i}')
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(actual == expected, f'{where}: value differs')
    elif isinstance(expected, int):
        require(actual == expected, f'{where}: count differs ({actual}, {expected})')
    else:
        require(math.isfinite(actual) and math.isfinite(expected) and
                abs(actual - expected) <= TOL, f'{where}: number differs ({actual}, {expected})')


def subset(actual, expected, where):
    """Historical records sometimes omit TN or newer derived metrics."""
    for key, value in expected.items():
        require(key in actual, f'{where}: absent historical key {key}')
        if isinstance(value, dict):
            subset(actual[key], value, f'{where}/{key}')
        else:
            close(actual[key], value, f'{where}/{key}')


def check_metrics(m, where):
    require(set(m) == {'_overall', *FAMILIES}, f'{where}: metric populations differ')
    for family, values in m.items():
        for k in ('tp', 'fp', 'fn', 'tn'):
            require(isinstance(values[k], int) and not isinstance(values[k], bool) and values[k] >= 0,
                    f'{where}/{family}/{k}: invalid count')
        denom = 2 * values['tp'] + values['fp'] + values['fn']
        close(values['f1'], 2 * values['tp'] / denom if denom else 0., f'{where}/{family}/f1')
    o = m['_overall']
    require(0 <= o['clean_period_fp'] <= o['clean_rows'] <= o['fp'] + o['tn'], f'{where}: clean counts invalid')
    close(o['clean_fpr'], o['clean_period_fp'] / max(1, o['clean_rows']), f'{where}/clean_fpr')
    if 'all_negative_fpr' in o:
        close(o['all_negative_fpr'], o['fp'] / max(1, o['fp'] + o['tn']), f'{where}/all_negative_fpr')
    if 'family_macro_f1' in o:
        close(o['family_macro_f1'], statistics.mean(m[f]['f1'] for f in FAMILIES), f'{where}/macro')
        close(o['worst_family_f1'], min(m[f]['f1'] for f in FAMILIES), f'{where}/worst')
    return {**{f: m[f]['f1'] for f in FAMILIES}, 'pooled_f1': o['f1'],
            **{k: o[k] for k in ('clean_fpr', 'all_negative_fpr', 'family_macro_f1', 'worst_family_f1') if k in o}}


def safe_relative(value):
    p = PurePosixPath(value)
    require(isinstance(value, str) and value and not p.is_absolute() and '..' not in p.parts and '\\' not in value,
            f'Unsafe manifest path: {value!r}')
    return p


def verify_manifest(repo, manifest_path):
    manifest = read(manifest_path)
    entries = manifest['files']
    require(isinstance(entries, list) and entries, 'Manifest is empty')
    entries = entries + manifest.get('generated_files', [])
    by_original = {}; by_published = {}
    for entry in entries:
        original = str(safe_relative(entry['original_path'])) if entry.get('original_path') else None
        published = str(safe_relative(entry['published_path']))
        require((original is None or original not in by_original) and published not in by_published, 'Duplicate manifest path')
        p = repo / published
        require(p.is_file() and not p.is_symlink(), f'Missing or symbolic publication file: {published}')
        require(p.resolve().is_relative_to(repo.resolve()), f'Publication path escapes repository: {published}')
        require(p.stat().st_size == entry['bytes'], f'File size differs: {published}')
        require(re.fullmatch('[0-9a-f]{64}', entry['sha256']) is not None, f'Invalid SHA256: {published}')
        require(sha(p) == entry['sha256'], f'File hash differs: {published}')
        if original is not None:
            by_original[original] = p
        by_published[published] = entry
    return by_original, by_published


def verify_campaign(analysis, original, published_files=None):
    """original(path) resolves a workspace-relative historical path to its published copy."""
    operating = analysis / 'operating_points'; extracted = analysis / 'hybrid_scores'
    policy_record = read(operating / 'protocol.json'); policy = policy_record['policy']
    policy_hash = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    close(policy_hash, policy_record['policy_sha256'], 'canonical policy SHA256')
    code_hash = sha(analysis / 'compare_operating_points.py'); extractor_hash = sha(analysis / 'extract_hybrid_scores.py')
    frozen = read(operating / 'analysis_code_frozen.json')
    close(frozen['analysis_code_sha256'], code_hash, 'frozen comparison source')
    close(frozen['policy_sha256'], policy_hash, 'frozen comparison policy')
    require(frozen['all_twenty_calibrations_required_before_new_evaluation_analysis'] is True, 'Calibration-first requirement missing')
    require(policy['model_seeds'] == list(SEEDS) and set(policy['networks']) == set(NETWORKS), 'Campaign identities differ')
    close(policy['calibration_clean_fpr_caps'], list(CAPS), 'calibration caps')
    require(len(policy['lambda_grid']) == 82 and policy['lambda_grid'].count(1.) == 1, 'Threshold grid differs')
    require(policy['locked_test_read'] is policy['fitting_permitted'] is policy['new_data_permitted'] is False,
            'Unexpected fit/data/locked-test protocol')
    summary = read(operating / 'complete_summary.json')
    require(summary['status'] == 'complete' and summary['shared_sources_not_independent_cells'] is True, 'Incomplete or mis-scoped summary')
    close(summary['policy_sha256'], policy_hash, 'summary policy')
    fingerprints = {}; omitted = {}; counts = defaultdict(int)

    def fingerprint(path, digest, require_bytes=False):
        safe_relative(path)
        require(re.fullmatch('[0-9a-f]{64}', digest) is not None, f'Invalid provenance hash: {path}')
        if path in fingerprints:
            close(digest, fingerprints[path], f'consistent provenance/{path}')
        fingerprints[path] = digest
        if require_bytes:
            close(sha(original(path)), digest, f'published input bytes/{path}')
            counts['included_provenance_hash_links'] += 1
        else:
            omitted[path] = digest

    def extraction(meta, network, seed, role, source):
        require((meta['network'], meta['model_seed'], meta['role'], meta['source_seed']) ==
                (network, seed, role, source), 'Extraction identity differs')
        close(meta['extractor_sha256'], extractor_hash, 'extractor code hash')
        require(meta['baseline_endpoints_aligned'] is True and meta['fit_called'] is False and meta['locked_test_read'] is False,
                'Extraction provenance flags differ')
        require(role != 'evaluation' or meta['original_report_checked'] is True, 'Original evaluation report check missing')
        require(meta['calibration_pooled_check_required'] is (role == 'calibration'), 'Calibration pooled-check role differs')
        check_metrics(meta['original_counts'], 'extracted original counts')
        close(sum(meta['original_counts']['_overall'][k] for k in ('tp', 'fp', 'fn', 'tn')), meta['rows'], 'extracted row total')
        fingerprint(meta['manifest'], meta['manifest_sha256'], True)
        fingerprint(meta['feature_cache'], meta['feature_cache_sha256'])
        for path, value in meta['model_hashes'].items():
            fingerprint(path, value, path.endswith('.json'))
        require(meta['original_rule'] in meta['model_hashes'], 'Original rule absent from input hashes')
        if meta['retained_general_cache']:
            fingerprint(meta['retained_general_cache'], meta['retained_general_cache_sha256'])
        fingerprint(f'{ORIGINAL_ANALYSIS}/hybrid_scores/{network}/seed{seed}/{role}_source{source}.npz', meta['npz_sha256'])
        counts['extraction_sidecars'] += 1

    for network in NETWORKS:
        sources = policy['evaluation_sources'][network]; calibration_sources = policy['calibration_sources'][network]
        require(len(sources) == len(set(sources)) == 6 and len(calibration_sources) == (5 if network == 'modena' else 4), 'Source counts differ')
        block = summary['networks'][network]
        require(block['status'] == 'complete' and block['completed_cells'] == block['expected_cells'] == 60 and not block['missing_cells'], 'Incomplete network')
        require(len(list((operating/network).glob('seed*/evaluation_source*.json'))) == 60, 'Unexpected evaluation-file count')
        groups = defaultdict(list)
        for seed in SEEDS:
            folder = operating/network/f'seed{seed}'; side_folder = extracted/network/f'seed{seed}'
            cp = folder/'calibration_selection.json'; cal = read(cp); validation_path = side_folder/'calibration_validation.json'; validation = read(validation_path)
            require((cal['network'], cal['model_seed']) == (network, seed), 'Calibration identity differs')
            close(cal['analysis_code_sha256'], code_hash, 'calibration code'); close(cal['policy_sha256'], policy_hash, 'calibration policy')
            require(cal['evaluation_read_for_selection'] is cal['fit_called'] is cal['locked_test_read'] is False, 'Calibration provenance flags differ')
            require(validation['original_pooled_report_matches'] is True, 'Original calibration reproduction flag missing')
            close(validation['extractor_sha256'], extractor_hash, 'calibration extractor')
            close(sha(validation_path), cal['calibration_validation_sha256'], 'calibration-validation hash')
            close(validation['source_npz_sha256'], cal['source_npz_sha256'], 'calibration source hash chain')
            close(validation['sources'], calibration_sources, 'calibration sources')
            pooled = {f: defaultdict(int) for f in ('_overall', *FAMILIES)}
            for source in calibration_sources:
                meta = read(side_folder/f'calibration_source{source}.json'); extraction(meta, network, seed, 'calibration', source)
                close(meta['thresholds'], cal['hybrid']['original_thresholds'], 'Original calibration thresholds')
                close(meta['npz_sha256'], validation['source_npz_sha256'][str(source)], 'calibration source score hash')
                for f, values in meta['original_counts'].items():
                    for k in ('tp', 'fp', 'fn', 'tn', 'clean_rows', 'clean_period_fp'):
                        if k in values: pooled[f][k] += values[k]
            for f, values in pooled.items():
                subset(validation['counts'][f], dict(values), f'pooled original calibration/{f}')
            check_metrics(validation['counts'], 'pooled original calibration'); subset(cal['hybrid']['original'], validation['counts'], 'calibration original linkage')
            for path, value in cal['original_rule_and_baseline_score_sha256'].items(): fingerprint(path, value, path.endswith('.json'))
            expected_choices = {(cap, obj) for cap in CAPS for obj in ('pooled', 'balanced')}
            h = cal['hybrid']; require(len(h['path_calibration']) == 82, 'Hybrid calibration path incomplete')
            for m in h['path_calibration']: check_metrics(m, 'hybrid calibration path')
            one = policy['lambda_grid'].index(1.)
            close(h['path_calibration'][one], h['original'], 'lambda-one original calibration')
            for branch, thresholds in h['path_thresholds'].items():
                require(len(thresholds) == 82 and all(a >= b for a,b in zip(thresholds, thresholds[1:])), 'Threshold path not nested')
                close(thresholds[one], h['original_thresholds'][branch], 'original branch threshold')
            for method in (h, cal['baseline']['delta0'], cal['baseline']['delta3']):
                require({(v['cap'],v['objective']) for v in method['selected']} == expected_choices and len(method['selected']) == 14, 'Calibration selections incomplete')
                for v in method['selected']:
                    check_metrics(v['calibration'], 'selected calibration'); require(v['calibration']['_overall']['clean_fpr'] <= v['cap'], 'Calibration cap exceeded')
                    if method is h:
                        i = v['lambda_index']; close(v['calibration'], h['path_calibration'][i], 'selected hybrid path')
                        close(v['lambda'], policy['lambda_grid'][i], 'selected multiplier')
                        for branch in h['path_thresholds']: close(v['thresholds'][branch], h['path_thresholds'][branch][i], 'selected branch threshold')
                        def rank(j):
                            o = h['path_calibration'][j]['_overall']; a,b = (o['f1'],o['worst_family_f1']) if v['objective']=='pooled' else (o['worst_family_f1'],o['f1'])
                            return a,b,-o['clean_fpr'],-policy['lambda_grid'][j]
                        eligible = [j for j,m in enumerate(h['path_calibration']) if m['_overall']['clean_fpr'] <= v['cap']]
                        require(i == max(eligible, key=rank), 'Hybrid calibration objective/tie selection differs')
            counts['calibration_selections'] += 1; counts['calibration_validation_records'] += 1
            historical_base = CAMPAIGN if seed <= 705 else CAMPAIGN+'/final_ten_model_seeds_v1'
            old_hybrid = read(original(f'{historical_base}/stage_e_modena/seed/{seed}/evaluation_report.json' if network=='modena' else f'{historical_base}/protected_history_system_v1/seed/{seed}/confirmation.json'))
            for source in sources:
                cell = read(folder/f'evaluation_source{source}.json')
                require((cell['network'],cell['model_seed'],cell['source_seed']) == (network,seed,source), 'Evaluation identity differs')
                close(cell['policy_sha256'], policy_hash, 'evaluation policy'); close(cell['calibration_selection_sha256'], sha(cp), 'evaluation calibration hash')
                require(cell['original_counts_reproduced'] is True and cell['fit_called'] is False and cell['locked_test_read'] is False, 'Evaluation provenance flags differ')
                meta = read(side_folder/f'evaluation_source{source}.json'); extraction(meta,network,seed,'evaluation',source)
                close(meta['thresholds'], cal['hybrid']['original_thresholds'], 'Original evaluation thresholds')
                close(meta['npz_sha256'],cell['hybrid_source_npz_sha256'],'evaluation score fingerprint'); subset(cell['hybrid']['original'],meta['original_counts'],'original extraction-to-evaluation counts')
                fingerprint(f'{CAMPAIGN}/single_model_baseline_v1/{network}/evaluation_predictions_{source}.npz',cell['baseline_source_npz_sha256'])
                m = cell['hybrid']['original']
                if network=='modena':
                    close(old_hybrid['per_data_seed'][str(source)]['rows'], meta['rows'], 'Original Modena row total')
                    historic = old_hybrid['per_data_seed'][str(source)]['arms']['candidate']
                    for k in ('tp','fp','fn','clean_period_fp','clean_fpr'): close(m['_overall'][k],historic[k],f'original Modena/{k}')
                    for f in FAMILIES: require(abs(m[f]['f1']-historic[f]) <= 5.00001e-7,'Historical rounded Modena family F1 differs')
                else:
                    historic = next(p['candidate'] for p in old_hybrid['pairs'] if p['source']==source)
                    subset(m,historic,'original L-Town counts')
                old_base = read(original(f'{CAMPAIGN}/single_model_baseline_v1/{network}/evaluation_source_{source}.json'))
                for old in old_base['rows']:
                    if old['model_seed']==seed: subset(cell['baseline'][f"delta{old['delay_hours']}"]['original'][old['objective']],old['metrics'],'original baseline counts')
                def add(key, metrics):
                    flat = check_metrics(metrics, key); groups[key].append({'seed':seed,'source':source,**flat})
                add('hybrid_original',m)
                require(len(cell['hybrid']['path'])==82,'Evaluation hybrid path incomplete')
                for i,x in enumerate(cell['hybrid']['path']): add(f'hybrid_path_{i}',x)
                close(cell['hybrid']['path'][one],m,'original evaluation at lambda-one')
                for v in cell['hybrid']['selected']:
                    cv = next(x for x in h['selected'] if (x['cap'],x['objective'])==(v['cap'],v['objective']))
                    close(v['lambda_index'],cv['lambda_index'],'evaluation frozen hybrid selection');close(v['metrics'],cell['hybrid']['path'][v['lambda_index']],'evaluation selected hybrid metrics')
                    add(f"hybrid_{v['objective']}_cap{v['cap']:g}",v['metrics'])
                for delta,b in cell['baseline'].items():
                    for obj,x in b['original'].items(): add(f'lightgbm_{delta}_original_{obj}',x)
                    for i,x in enumerate(b['path']): add(f'lightgbm_{delta}_path_{i}',x)
                    for v in b['selected']:
                        cv = next(x for x in cal['baseline'][delta]['selected'] if (x['cap'],x['objective'])==(v['cap'],v['objective']))
                        close(v['threshold'],cv['threshold'],'evaluation frozen baseline threshold');add(f"lightgbm_{delta}_{v['objective']}_cap{v['cap']:g}",v['metrics'])
                for name,x in cell['frozen_rule_diagnostics'].items():
                    add(f'diagnostic_{name}',x)
                    if name=='original': close(x,m,'original diagnostic');continue
                    removal = name in ('general_only','general_plus_drift','general_plus_noise','specialists_only')
                    for f in ('_overall',*FAMILIES):
                        for k in ('tp','fp'): require(x[f][k]<=m[f][k] if removal else x[f][k]>=m[f][k], 'Diagnostic count monotonicity differs')
                        close(x[f]['tp']+x[f]['fn'],m[f]['tp']+m[f]['fn'],'diagnostic positive population')
                        close(x[f]['fp']+x[f]['tn'],m[f]['fp']+m[f]['tn'],'diagnostic negative population')
                counts['evaluation_cells'] += 1
        require(set(groups)==set(block['groups']), 'Aggregate group inventory differs')
        def mean(rows):
            return {k: statistics.mean(r[k] for r in rows) for k in rows[0] if k not in ('seed','source')}
        def marginal(rows, field, ids): return {str(v):mean([r for r in rows if r[field]==v]) for v in ids}
        for name,rows in groups.items():
            require(len(rows)==60,'Aggregate group incomplete')
            bm=marginal(rows,'seed',SEEDS);bs=marginal(rows,'source',sources)
            expected={'mean':mean(rows),'by_model':bm,'by_source':bs,
                      'model_mean_sd':{k:statistics.stdev(v[k] for v in bm.values()) for k in mean(rows)},
                      'source_mean_sd':{k:statistics.stdev(v[k] for v in bs.values()) for k in mean(rows)}}
            close(block['groups'][name],expected,f'{network}/{name}');counts['aggregate_groups'] += 1
        paired={}
        for cap in CAPS:
            for obj in ('pooled','balanced'):
                for delta in (0,3):
                    left={(r['seed'],r['source']):r for r in groups[f'hybrid_{obj}_cap{cap:g}']};right={(r['seed'],r['source']):r for r in groups[f'lightgbm_delta{delta}_{obj}_cap{cap:g}']}
                    diff=[{'seed':seed,'source':source,**{k:left[seed,source][k]-right[seed,source][k] for k in mean(list(left.values()))}} for seed,source in sorted(left)]
                    paired[f'hybrid_minus_delta{delta}_{obj}_cap{cap:g}']={'mean_difference':mean(diff),'by_model':marginal(diff,'seed',SEEDS),'by_source':marginal(diff,'source',sources),'higher_cell_count':{k:sum(r[k]>0 for r in diff) for k in mean(diff)},'equal_cell_count':{k:sum(r[k]==0 for r in diff) for k in mean(diff)}}
        close(block['paired_comparisons'],paired,f'{network}/paired comparisons');counts['paired_comparisons']+=len(paired)
    require(counts['evaluation_cells']==120 and counts['calibration_selections']==20 and counts['extraction_sidecars']==210 and counts['calibration_validation_records']==20,'Campaign counts differ')
    if published_files is not None:
        for p in analysis.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.py','.csv','.pdf','.png','.txt'):
                require(p.resolve() in published_files, f'Unmanifested analysis file: {p}')
    counts['omitted_input_fingerprints_consistent_not_byte_verified']=len(omitted)
    return dict(counts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--manifest',type=Path)
    args=parser.parse_args();repo=args.repo_root.resolve();base=repo/'results/supplementary_20261008'
    manifest=args.manifest or base/'manifest.json'
    mapping, published=verify_manifest(repo,manifest)
    def original(path):
        require(path in mapping,f'Required original record absent from manifest: {path}')
        return mapping[path]
    result=verify_campaign(base/'analysis',original,{(repo/p).resolve() for p in published})
    print(json.dumps({'status':'PASS','published_file_hashes_verified':len(published),**result,
        'scope':'Published JSON count arithmetic, complete aggregation and recorded provenance chains verified. Omitted NPZ/model/telemetry bytes, model inference, endpoint alignment and exact baseline score search are not independently reproduced.'},indent=2))


if __name__=='__main__':
    try: main()
    except (ValueError,KeyError,TypeError,StopIteration,OSError,json.JSONDecodeError) as exc:
        print(f'FAIL: {exc}',file=sys.stderr);sys.exit(1)

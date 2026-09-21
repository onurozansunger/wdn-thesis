"""Supplementary, frozen single-classifier comparison; never selects a hybrid."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import lightgbm as lgb
import joblib
import numpy as np

from build_feature_cache import ROOT, CAMPAIGN, sha, write_json, atomic_npz
from screen_observed_history import build_bank
from wdn.delayed_decision_features import delayed_decision_features
from wdn.models.residual_hybrid import training_weights

OUT = CAMPAIGN / 'single_model_baseline_v1'
FINAL = CAMPAIGN / 'final_ten_model_seeds_v1/final_results.json'
PLAN = Path(__file__).with_name('SINGLE_MODEL_BASELINE.md')
SEEDS = tuple(range(701, 711))
NETWORKS = ('modena', 'ltown')
DELTAS = (0, 3)
FAMILIES = {1:'random', 2:'replay', 3:'drift', 4:'noise', 5:'targeted'}
OBJECTIVES = ('pooled', 'balanced')
CAP = .005
PARAMS = dict(n_estimators=400, num_leaves=31, min_child_samples=20,
              learning_rate=.03, reg_lambda=20., reg_alpha=1.,
              colsample_bytree=.85, n_jobs=4, verbosity=-1,
              deterministic=True, force_col_wise=True)
META = ('labels','families','source','scenario','event','timestep','node')


def read(p): return json.loads(Path(p).read_text())
def rel(p): return str(Path(p).relative_to(ROOT))
def frozen(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    value = json.loads(json.dumps(value))
    if p.exists():
        if read(p) != value: raise RuntimeError(f'Frozen record differs: {p}')
    else: write_json(p, value)


def manifest_name(network, role, source=None):
    if role in ('train','calibration'): return f'{network}_{role}'
    assert role == 'evaluation'
    return (f'modena_eval_seed{source}' if network == 'modena' else
            f'ltown_protected_history_confirmation_seed{source}')


def manifest(network, role, source=None):
    return CAMPAIGN/'features'/manifest_name(network, role, source)/'manifest.json'


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    final = read(FINAL)
    files = [PLAN, Path(__file__).resolve(), FINAL,
             ROOT/'tests/test_single_model_baseline.py',
             ROOT/'tests/test_delayed_decision_features.py']
    files += list((ROOT/'src/wdn').rglob('*.py'))
    files += list(Path(__file__).parent.glob('*.py'))
    manifests, inputs = {}, {}
    for network in NETWORKS:
        paths = [manifest(network, role) for role in ('train','calibration')]
        paths += [manifest(network, 'evaluation', s) for s in final['evaluation_sources'][network]]
        for path in paths:
            d = read(path); files.append(path)
            manifests[rel(path)] = d
            for piece in d['pieces']:
                assert piece['config_pressure_missing'] == piece['config_flow_missing'] == .5
                inputs[piece['cache']] = piece['cache_sha256']
                if network == 'ltown':
                    directory = ROOT/'data/thesis_v2'/piece['directory']
                    for name in ('generate_config.yaml','snapshots.pkl','corrupted.pkl'):
                        # Hashing verifies immutable input identity, not model performance.
                        raw = directory/name
                        inputs[rel(raw)] = sha(raw)
    frozen(OUT/'protocol_frozen.json', {
        'created_utc': '2026-09-14', 'study': 'supplementary single-classifier comparison',
        'model_seeds':SEEDS, 'networks':NETWORKS, 'delays':DELTAS,
        'sources':final['evaluation_sources'], 'parameters':PARAMS,
        'calibration_objectives':OBJECTIVES, 'clean_fpr_cap':CAP,
        'prespecified_plan':rel(PLAN), 'code_sha256':{rel(p):sha(p) for p in sorted(set(files))},
        'input_sha256':inputs, 'manifests':manifests,
        'versions':{'python':sys.version,'numpy':np.__version__,'lightgbm':lgb.__version__,'joblib':joblib.__version__},
        'final_results_sha256':sha(FINAL), 'locked_test_read':False,
        'already_used_evaluation_sources':True, 'fresh_independent_confirmation':False})
    print('Protocol frozen before training', flush=True)


def verify():
    p = read(OUT/'protocol_frozen.json')
    for name, digest in p['code_sha256'].items():
        if sha(ROOT/name) != digest: raise RuntimeError('Frozen code/input changed: '+name)
    return p


def load_piece(network, path, piece):
    p = read(OUT/'protocol_frozen.json')
    m = p['manifests'][rel(path)]
    cache = ROOT/piece['cache']
    if sha(cache) != p['input_sha256'][piece['cache']]:
        raise RuntimeError('Feature cache changed: '+str(cache))
    with np.load(cache) as z: a = {k:z[k] for k in (*META,'X')}
    assert len(a['labels']) == piece['rows']
    assert np.all(np.isin(a['labels'], [0,1]))
    names = m['feature_names']
    # Compute temporal columns before sampling: missing hours are not invented.
    forward, forward_names = delayed_decision_features(a, names, 3)
    if network == 'ltown':
        directory = ROOT/'data/thesis_v2'/piece['directory']
        for name in ('generate_config.yaml','snapshots.pkl','corrupted.pkl'):
            if sha(directory/name) != p['input_sha256'][rel(directory/name)]:
                raise RuntimeError('Received observation source changed')
        histories, extra_names = build_bank(a, names, {'pieces':[piece]}, (0.,))
        X = np.column_stack((a['X'], histories[0.], forward)).astype(np.float32)
        causal_width = len(names)+len(extra_names)
        names = names+extra_names+forward_names
    else:
        X = np.column_stack((a['X'], forward)).astype(np.float32)
        causal_width = len(names)
        names = names+forward_names
    a.pop('X')
    assert causal_width == (116 if network == 'modena' else 158)
    assert X.shape[1] == causal_width+59
    return a, X, names, causal_width


def prepare(network, role):
    assert role in ('train','calibration')
    verify()
    folder = OUT/'cache'/f'{network}_{role}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'ready.json').exists(): return
    path = manifest(network, role); m = read(path)
    width = 175 if network == 'modena' else 217
    target = folder/'features.npy'
    X = np.lib.format.open_memmap(target, mode='w+', dtype=np.float32,
                                 shape=(m['total_rows'],width))
    parts, offset = [], 0
    for piece in m['pieces']:
        print(network, role, 'building source', piece['seed'], flush=True)
        a, block, names, causal = load_piece(network, path, piece)
        n = len(block); X[offset:offset+n] = block
        parts.append(a); offset += n
        del a, block; gc.collect()
    assert offset == len(X)
    X.flush(); del X
    atomic_npz(folder/'metadata.npz', **{k:np.concatenate([a[k] for a in parts]) for k in META})
    frozen(folder/'ready.json', {'rows':offset,'feature_names':names,'causal_width':causal,
        'protocol_sha256':sha(OUT/'protocol_frozen.json'),
        'features_sha256':sha(target),'metadata_sha256':sha(folder/'metadata.npz')})
    print(network, role, 'feature bank ready', offset, flush=True)


def cache(network, role):
    folder = OUT/'cache'/f'{network}_{role}'
    ready = read(folder/'ready.json')
    assert ready['protocol_sha256'] == sha(OUT/'protocol_frozen.json')
    with np.load(folder/'metadata.npz') as z: a = {k:z[k] for k in z.files}
    return np.load(folder/'features.npy', mmap_mode='r'), a, ready


def report(score, threshold, a):
    pred = score > threshold; y = a['labels'] > 0; f = a['families']
    result = {}
    for code, name in [(None,'_overall'), *FAMILIES.items()]:
        mask = np.ones(len(y),bool) if code is None else f == code
        tp = int(np.sum(mask & y & pred)); fp = int(np.sum(mask & ~y & pred))
        fn = int(np.sum(mask & y & ~pred)); tn = int(np.sum(mask & ~y & ~pred))
        result[name] = dict(tp=tp,fp=fp,fn=fn,tn=tn,f1=2*tp/max(1,2*tp+fp+fn))
    clean = f == 0
    result['_overall'].update(clean_period_fp=int(pred[clean].sum()),clean_rows=int(clean.sum()),
        clean_fpr=float(pred[clean].mean()),family_macro_f1=float(np.mean([result[n]['f1'] for n in FAMILIES.values()])),
        worst_family_f1=min(result[n]['f1'] for n in FAMILIES.values()))
    return result


def choose_thresholds(score, a, cap=CAP):
    """Exact tied-score threshold sweep; strict > semantics in every stage."""
    order = np.argsort(-score,kind='stable'); s = score[order]
    y = a['labels'][order] > 0; f = a['families'][order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]),len(s)-1]
    tp = np.cumsum(y,dtype=np.int64)[ends]
    pooled = 2*tp/np.maximum(1,int(y.sum())+ends+1)
    worst = np.ones(len(ends))
    for code in FAMILIES:
        mask = f == code
        f_tp = np.cumsum(mask & y,dtype=np.int64)[ends]
        f_pred = np.cumsum(mask,dtype=np.int64)[ends]
        worst = np.minimum(worst,2*f_tp/np.maximum(1,int(np.sum(mask & y))+f_pred))
    clean = f == 0
    fpr = np.cumsum(clean,dtype=np.int64)[ends]/int(clean.sum())
    thresholds = np.r_[float(np.max(s)),np.nextafter(s[ends],-np.inf)]
    pooled = np.r_[0.,pooled]; worst = np.r_[0.,worst]; fpr = np.r_[0.,fpr]
    allowed = np.flatnonzero(fpr <= cap)
    result = {}
    for objective in OBJECTIVES:
        first, second = (pooled,worst) if objective == 'pooled' else (worst,pooled)
        best = allowed[np.lexsort((-fpr[allowed],second[allowed],first[allowed]))[-1]]
        threshold = float(thresholds[best]); metrics = report(score,threshold,a)
        assert abs(metrics['_overall']['f1']-pooled[best]) < 1e-12
        assert metrics['_overall']['clean_fpr'] <= cap
        result[objective] = {'threshold':threshold,'calibration':metrics}
    return result


def fit(network, seed):
    verify(); start = time.monotonic()
    folder = OUT/network/str(seed); folder.mkdir(parents=True, exist_ok=True)
    if all((folder/f'delta{d}_selection.json').exists() for d in DELTAS): return
    X, a, ready = cache(network,'train')
    positive = np.flatnonzero(a['labels'] > 0); negative = np.flatnonzero(a['labels'] == 0)
    chosen = np.random.default_rng(600+seed).choice(negative,min(60000,len(negative)),replace=False)
    selected = np.r_[positive,chosen]
    event = np.where(a['event'] >= 0,a['source']*10**6+a['event'],-1)
    weights = training_weights(a['labels'],a['families'],event,a['scenario'],selected)
    sampled = np.array(X[selected],dtype=np.float32); y = a['labels'][selected]
    sample_hash = hashlib.sha256(selected.tobytes()).hexdigest()
    del X,a,event,selected,negative,chosen; gc.collect()
    for delta in DELTAS:
        modelpath = folder/f'delta{delta}.joblib'
        if modelpath.exists(): continue
        width = ready['causal_width'] if delta == 0 else sampled.shape[1]
        print(network,seed,'fitting',delta,'hours',len(y),'rows',width,'features',flush=True)
        model = lgb.LGBMClassifier(**PARAMS,random_state=5100+seed)
        model.fit(sampled[:,:width],y,sample_weight=weights)
        joblib.dump(model,modelpath.with_suffix('.tmp'))
        modelpath.with_suffix('.tmp').replace(modelpath)
        frozen(folder/f'delta{delta}_fit.json', {'network':network,'seed':seed,'delay_hours':delta,
            'feature_names':ready['feature_names'][:width],'training_sample_sha256':sample_hash,
            'sampled_rows':len(y),'positive_rows':len(positive),'negative_rows':len(y)-len(positive),
            'model_sha256':sha(modelpath),'elapsed_seconds':time.monotonic()-start})
        del model; gc.collect()
    del sampled,y,weights; gc.collect()
    X,a,ready = cache(network,'calibration')
    for delta in DELTAS:
        target = folder/f'delta{delta}_selection.json'
        if target.exists(): continue
        model = joblib.load(folder/f'delta{delta}.joblib')
        width = model.n_features_in_
        scores = np.empty(len(X),float)
        for start in range(0,len(X),100000):
            scores[start:start+100000] = model.predict_proba(X[start:start+100000,:width])[:,1]
        print(network,seed,'calibrating',delta,'hours',flush=True)
        selections = choose_thresholds(scores,a)
        atomic_npz(folder/f'delta{delta}_calibration_scores.npz',score=scores)
        frozen(target, {'model_sha256':sha(folder/f'delta{delta}.joblib'),
            'protocol_sha256':sha(OUT/'protocol_frozen.json'),'objectives':selections,
            'calibration_rows':len(X),'evaluation_read':False})
        del scores,model; gc.collect()
    print(network,seed,'training and calibration complete',flush=True)


def freeze_evaluation():
    verify(); files = []
    for n in NETWORKS:
        for s in SEEDS:
            for d in DELTAS:
                folder = OUT/n/str(s)
                files += [folder/f'delta{d}{suffix}' for suffix in ('.joblib','_fit.json','_selection.json')]
    frozen(OUT/'evaluation_frozen.json', {'protocol_sha256':sha(OUT/'protocol_frozen.json'),
        'all_40_models_and_80_operating_points_frozen':True,
        'model_rule_sha256':{rel(p):sha(p) for p in files},'locked_test_read':False})


def check_evaluation():
    protocol = verify(); freeze = read(OUT/'evaluation_frozen.json')
    assert freeze['protocol_sha256'] == sha(OUT/'protocol_frozen.json')
    for p,digest in freeze['model_rule_sha256'].items():
        if sha(ROOT/p) != digest: raise RuntimeError('Model or rule changed: '+p)
    return protocol


def evaluate(network):
    protocol = check_evaluation()
    for source in protocol['sources'][network]:
        target = OUT/network/f'evaluation_source_{source}.json'
        if target.exists(): continue
        print(network,'evaluating frozen models on source',source,flush=True)
        path = manifest(network,'evaluation',source); m = read(path)
        assert len(m['pieces']) == 1
        a,X,names,causal = load_piece(network,path,m['pieces'][0])
        predictions,rows = {},[]
        for seed in SEEDS:
            folder = OUT/network/str(seed)
            for delta in DELTAS:
                model = joblib.load(folder/f'delta{delta}.joblib')
                assert read(folder/f'delta{delta}_fit.json')['feature_names'] == names[:model.n_features_in_]
                score = np.empty(len(X),float)
                for start in range(0,len(X),100000):
                    score[start:start+100000] = model.predict_proba(X[start:start+100000,:model.n_features_in_])[:,1]
                selection = read(folder/f'delta{delta}_selection.json')
                for objective,point in selection['objectives'].items():
                    rows.append({'model_seed':seed,'source_seed':source,'delay_hours':delta,
                        'objective':objective,'threshold':point['threshold'],'metrics':report(score,point['threshold'],a)})
                predictions[f'seed{seed}_delta{delta}'] = score
                del model; gc.collect()
        atomic_npz(OUT/network/f'evaluation_predictions_{source}.npz',**a,**predictions)
        frozen(target, {'network':network,'source':source,'rows':rows,
            'evaluation_freeze_sha256':sha(OUT/'evaluation_frozen.json'),
            'endpoint_count':len(X),'locked_test_read':False})
        del X,a,predictions,rows; gc.collect()
        print(network,'source completed',source,flush=True)


def summarize():
    protocol = check_evaluation(); final = read(FINAL); output = {}
    metrics = (*FAMILIES.values(),'family_macro_f1','pooled_f1','clean_fpr')
    def values(r):
        x = r['metrics']; o = x['_overall']
        return {**{n:x[n]['f1'] for n in FAMILIES.values()},'family_macro_f1':o['family_macro_f1'],
                'pooled_f1':o['f1'],'clean_fpr':o['clean_fpr']}
    for n in NETWORKS:
        rows = [r for s in protocol['sources'][n] for r in read(OUT/n/f'evaluation_source_{s}.json')['rows']]
        hybrid = {(r['model_seed'],r['source_seed']):r for r in final['networks'][n]['rows']}
        arms = {}
        for delta in DELTAS:
            for obj in OBJECTIVES:
                selected = [r for r in rows if r['delay_hours']==delta and r['objective']==obj]
                assert len(selected)==60
                flat = [dict(model_seed=r['model_seed'],source_seed=r['source_seed'],**values(r)) for r in selected]
                mean = {k:float(np.mean([r[k] for r in flat])) for k in metrics}
                source_means = {str(s):{k:float(np.mean([r[k] for r in flat if r['source_seed']==s])) for k in metrics} for s in protocol['sources'][n]}
                model_means = {str(s):{k:float(np.mean([r[k] for r in flat if r['model_seed']==s])) for k in metrics} for s in SEEDS}
                diff = []
                for r in flat:
                    h = hybrid[(r['model_seed'],r['source_seed'])]
                    h = dict(h,family_macro_f1=float(np.mean([h[f] for f in FAMILIES.values()])))
                    diff.append({k:h[k]-r[k] for k in metrics})
                arms[f'delta{delta}_{obj}'] = {'rows':flat,'mean':mean,'by_source':source_means,'by_model':model_means,
                    'model_mean_sd':{k:float(np.std([r[k] for r in model_means.values()],ddof=1)) for k in metrics},
                    'source_mean_sd':{k:float(np.std([r[k] for r in source_means.values()],ddof=1)) for k in metrics},
                    'hybrid_minus_baseline':{k:float(np.mean([r[k] for r in diff])) for k in metrics},
                    'hybrid_higher_pairs':{k:sum(r[k]>0 for r in diff) for k in metrics},
                    'range':{k:[min(r[k] for r in flat),max(r[k] for r in flat)] for k in metrics}}
        output[n] = arms
    frozen(OUT/'summary.json', {'status':'completed','networks':output,'protocol_sha256':sha(OUT/'protocol_frozen.json'),
        'evaluation_freeze_sha256':sha(OUT/'evaluation_frozen.json'),'final_hybrid_results_sha256':sha(FINAL),
        'locked_test_read':False,'fresh_independent_confirmation':False})
    print(json.dumps({n:{a:r['mean'] for a,r in arms.items()} for n,arms in output.items()},indent=2),flush=True)


def run_all():
    freeze()
    for n in NETWORKS:
        for role in ('train','calibration'): prepare(n,role)
    def worker(task):
        n,s = task; logs = OUT/'logs'; logs.mkdir(exist_ok=True)
        with (logs/f'{n}_{s}.log').open('a') as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--stage','fit','--network',n,'--seed',str(s)],
                cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Worker complete',n,s,flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker,[(n,s) for n in NETWORKS for s in SEEDS]))
    freeze_evaluation()
    for n in NETWORKS: evaluate(n)
    summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','freeze','prepare','fit','freeze-evaluation','evaluate','report'],default='all')
    parser.add_argument('--network',choices=NETWORKS); parser.add_argument('--role',choices=['train','calibration'])
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args = parser.parse_args()
    if args.stage=='all': run_all()
    elif args.stage=='freeze': freeze()
    elif args.stage=='prepare': prepare(args.network,args.role)
    elif args.stage=='fit': fit(args.network,args.seed)
    elif args.stage=='freeze-evaluation': freeze_evaluation()
    elif args.stage=='evaluate': evaluate(args.network)
    else: summarize()

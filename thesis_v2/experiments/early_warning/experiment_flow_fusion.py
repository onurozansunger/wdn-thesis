"""Frozen four-arm received-flow / internal-reliability TRAIN experiment.

Nested leave-source-out teachers exclude BOTH the outer diagnostic source and
the inner prediction source from reference fitting and detector training.
Ten unordered exclusion pairs share computation without crossing that boundary.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import joblib
import numpy as np
from sklearn.metrics import average_precision_score
from build_feature_cache import ROOT, CAMPAIGN, sha, write_json, atomic_npz, check_distribution, config_of
from audit_replay_train import reference_dir
from replicate_observed_history import frozen_write, validate_partition
from screen_shared_history import load, predict_chunks, select, streams, BUDGETS
from wdn.latency_deployment import specialist_bank, FAMILY_NAMES, family_scores
from wdn.run_expert_redesign import CampaignData
from wdn.models.blind_reference import BlindPressureReference
from wdn.models.robust_reference import RobustBlindReference
from wdn.observed_history import observed_history_features
from wdn.received_flow_context import ReceivedFlowReference, flow_context_features
from wdn.reliable_general_mixture import ReliableCombination, ReliableGeneralMixture
from wdn.shared_history import SharedHistoryExpertMixture, fit_presampled

OUT = CAMPAIGN / 'flow_fusion_v1'
FOLDS = CAMPAIGN / 'stage_e_ltown/folds'
MANIFEST = CAMPAIGN / 'features/ltown_train/manifest.json'
SEEDS = tuple(range(701, 706))
ARMS = ('reference', 'fusion', 'flow', 'combined')
EXCLUSIONS = tuple(itertools.combinations(range(5), 2))


def manifest(): return json.loads(MANIFEST.read_text())


class ReceivedData:
    features = CampaignData.features
    def __init__(self, piece):
        self.piece = piece
        self.events = json.loads((OUT / 'raw' / str(piece['seed']) / 'events.json').read_text())
    def scenario(self, sid):
        if sid not in self.piece['scenarios']: raise ValueError('Non-TRAIN scenario access')
        return load(OUT / 'raw' / str(self.piece['seed']) / f'{sid}.npz')


def freeze():
    OUT.mkdir(exist_ok=True)
    m = manifest()
    files = [MANIFEST, Path(__file__), Path(__file__).with_name('screen_shared_history.py'),
             Path(__file__).with_name('build_feature_cache.py'),
             Path(__file__).with_name('experiment_received_trajectory.py'),
             Path(__file__).with_name('FLOW_FUSION_EXPERIMENT.md')]
    files += sorted((ROOT / 'src/wdn').rglob('*.py'))
    previous = json.loads((CAMPAIGN / 'received_trajectory_v1/protocol_frozen.json').read_text())['input_code_sha256']
    benchmark = config_of(ROOT / 'data/thesis_v2' / m['pieces'][0]['directory'])
    for piece in m['pieces']:
        directory = ROOT / 'data/thesis_v2' / piece['directory']
        check_distribution(directory, benchmark)
        for name in ('snapshots.pkl', 'corrupted.pkl', 'generate_config.yaml', 'events.json'):
            path = directory / name
            if sha(path) != previous[str(path.relative_to(ROOT))]: raise RuntimeError('Frozen TRAIN raw input changed')
            files.append(path)
    for fold in range(5):
        for name in ('features_train.npz', 'features_held_out.npz', 'reference.joblib'):
            p = FOLDS / f'fold_{fold}' / name
            if sha(p) != previous[str(p.relative_to(ROOT))]: raise RuntimeError('Frozen TRAIN fold changed')
            files.append(p)
        for seed in SEEDS:
            directory = reference_dir(fold, seed)
            files += [directory / 'observed_history_predictions.npz', directory / ('model.joblib' if (fold, seed) == (0, 701) else 'observed_history.joblib')]
    protocol = dict(arms=ARMS, folds=list(range(5)), seeds=SEEDS, nested_exclusions=EXCLUSIONS,
        features={'reference': 158, 'flow': 202, 'flow_reference': 'rank16 masked robust flow factorization and ridge10 normal received pressure regression',
                  'flow_lags': [1,3,6,12,24], 'quantization': '.1 standardized units'},
        fusion='Five positive-slope sigmoid calibrations on population-weighted inner source-held predictions; L2=.001 linear softmax correction to original internal routing, optimized for binary detection log loss. Same five component slots.',
        outer_sample='Exact existing all positives + seed-specific 60000 negatives with population weights',
        inner_sample='Per source all positives + seed-specific 20000 negatives with inverse sampling weights; three TRAIN sources per teacher. Same sample in both feature modes.',
        fitting_boundaries='Each inner teacher, pressure reference, flow reference and normalization excludes outer source and inner held source. One teacher per unordered excluded pair and model seed, reused only for the two legitimate outer-fold roles.',
        selection='All25 pairs completed first. Even held-source TRAIN scenarios select threshold, odd diagnose. >=20 replay wins, mean replay gain>=.01, positive mean gain each source; rank mean worst-family F1 then pooled then lower FPR. General changes in other families diagnostic; complete-system protections decisive.',
        threshold_budgets=BUDGETS,
        rounding='Selected candidate only, all5 folds seed701, pressure feature-path .01 and .05m. Mean replay gain positive, each source gain nonnegative, frozen thresholds, no reselection.',
        protection={'other_family_mean_drop': .005, 'other_family_source_drop': .01, 'pooled_mean_drop': 0., 'pooled_source_drop': .005, 'clean_fpr_mean_increase': 0., 'clean_fpr_source_increase': .0001},
        readiness='Complete-system calibration every mean family and every source-mean family>=.78; every source mean replay improves. .80 target. If not ready stop this experiment; do not inspect confirmation.',
        architecture=['General','Drift','Noise'], specialist_delay_hours=3, missing_pressure=.5, missing_flow=.5,
        generator_splits_severity_unchanged=True, locked_test_read=False, confirmation_read=False,
        input_code_sha256={str(p.relative_to(ROOT)): sha(p) for p in files})
    frozen_write(OUT / 'protocol_frozen.json', protocol)
    print('Frozen protocol', flush=True)


def raw(fold):
    piece = manifest()['pieces'][fold]
    folder = OUT / 'raw' / str(piece['seed'])
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists(): return
    data = CampaignData(ROOT / 'data/thesis_v2' / piece['directory'])
    for sid in piece['scenarios']:
        a = data.scenario(sid)
        idx = sorted((i for i,s in enumerate(data.snapshots) if s.scenario_id == sid), key=lambda i:data.snapshots[i].timestep)
        a['flow'] = np.stack([data.corrupted[i].flow_obs.numpy() for i in idx])
        a['flow_mask'] = np.stack([data.corrupted[i].flow_mask.numpy()>0 for i in idx])
        atomic_npz(folder / f'{sid}.npz', **a)
    write_json(folder / 'events.json', data.events)
    del data; gc.collect()
    with np.load(FOLDS / f'fold_{fold}/features_held_out.npz') as z:
        a = {k:z[k] for k in z.files if k != 'X'}
    positive, negative = np.flatnonzero(a['labels']>.5), np.flatnonzero(a['labels']<=.5)
    samples = {}
    for seed in SEEDS:
        chosen = np.random.default_rng(seed * 100000 + piece['seed']).choice(negative, min(20000,len(negative)), replace=False)
        samples[str(seed)] = np.r_[positive, chosen]
    pool = np.unique(np.concatenate(list(samples.values())))
    atomic_npz(folder / 'pool.npz', rows=pool, **{k:v[pool] for k,v in a.items()})
    for seed, indices in samples.items():
        atomic_npz(folder / f'sample_{seed}.npz', indices=np.searchsorted(pool,indices),
                   weights=np.r_[np.ones(len(positive)), np.full(len(indices)-len(positive), len(negative)/(len(indices)-len(positive)))])
    write_json(folder / 'complete.json', {'source':piece['seed'], 'rows':len(pool), 'received_only':True,
        'hashes':{p.name:sha(p) for p in folder.iterdir() if p.suffix in ('.json','.npz')}})
    print('raw prepared', fold, flush=True)


def normal_arrays(fit_folds):
    parts = {k:[] for k in ('values','mask','flow','flow_mask')}
    for fold in fit_folds:
        data = ReceivedData(manifest()['pieces'][fold])
        for sid in data.piece['scenarios']:
            a = data.scenario(sid); normal = a['families']==0
            for k in parts: parts[k].append(a[k][normal])
    return {k:np.concatenate(v) for k,v in parts.items()}


def fit_references(folder, fit_folds, pressure=True):
    if (folder/'flow_reference.joblib').exists() and (not pressure or (folder/'reference.joblib').exists()): return
    normal = normal_arrays(fit_folds)
    if pressure and not (folder/'reference.joblib').exists():
        reference = RobustBlindReference(BlindPressureReference(rank=16).fit(normal['values'],normal['mask']))
        reference.calibrate_scale(normal['values'],normal['mask'])
        joblib.dump(reference, folder/'reference.joblib')
    if not (folder/'flow_reference.joblib').exists():
        flow = ReceivedFlowReference().fit(normal['values'],normal['mask'],normal['flow'],normal['flow_mask'])
        joblib.dump(flow, folder/'flow_reference.joblib')
    write_json(folder/'fit_scope.json', {'fit_sources':[manifest()['pieces'][f]['seed'] for f in fit_folds],
        'normal_rows':len(normal['values']), 'received_only':True})
    print('references prepared', folder.name, flush=True)


def additional(a, names, flow_reference, rounding=0.):
    point, flow = np.empty((len(a['labels']),42),np.float32), np.empty((len(a['labels']),44),np.float32)
    filled = np.zeros(len(point),bool)
    for piece in manifest()['pieces']:
        source = np.flatnonzero(a['source']==piece['seed'])
        if not len(source): continue
        data = ReceivedData(piece)
        for scenario in np.unique(a['scenario'][source]):
            rows = source[a['scenario'][source]==scenario]
            obs = data.scenario(int(scenario-piece['seed']*1000))
            scale = a['X'][rows,names.index('normal_error_scale')]
            point[rows], pn = observed_history_features(obs['values'],obs['mask'],obs['timestep'],a['timestep'][rows],a['node'][rows],scale,rounding_m=rounding)
            flow[rows], fn = flow_context_features(flow_reference,obs['values'],obs['mask'],obs['flow'],obs['flow_mask'],obs['timestep'],a['timestep'][rows],a['node'][rows],scale,rounding)
            filled[rows]=True
    if not filled.all(): raise ValueError('Source outside frozen TRAIN manifest')
    return np.column_stack((point,flow)),pn+fn


def nested_prepare(first, second):
    folder = OUT/f'nested_{first}_{second}'; folder.mkdir(exist_ok=True)
    excluded = {first,second}; fitting = sorted(set(range(5))-excluded)
    fit_references(folder,fitting)
    reference, flow = joblib.load(folder/'reference.joblib'),joblib.load(folder/'flow_reference.joblib')
    for f,piece in enumerate(manifest()['pieces']):
        path=folder/f'source_{f}.npz'
        if path.exists(): continue
        data=ReceivedData(piece)
        pool=load(OUT/'raw'/str(piece['seed'])/'pool.npz')
        a,names=specialist_bank(data,piece['scenarios'],reference)
        a={k:np.asarray(v)[pool['rows']] for k,v in a.items() if k!='normal_reference_mae_m'}
        a['scenario']=a['scenario'].astype(np.int64)+piece['seed']*1000
        a['source']=np.full(len(a['labels']),piece['seed'],np.int64)
        for k in ('labels','families','scenario','node','timestep'): np.testing.assert_array_equal(a[k],pool[k])
        bank,added=additional(a,names,flow)
        a['X']=np.column_stack((a['X'],bank)).astype(np.float32)
        atomic_npz(path,**a)
        frozen_write(folder/'names.json',names+added)
        del a,pool,bank;gc.collect()
        print('nested cache',first,second,f,flush=True)
    write_json(folder/'cache_hashes.json',{p.name:sha(p) for p in folder.iterdir() if p.suffix in ('.npz','.joblib','.json') and p.name!='cache_hashes.json'})


def nested_fit(first,second,seed):
    folder=OUT/f'nested_{first}_{second}'
    target=folder/f'seed_{seed}';target.mkdir(exist_ok=True)
    if (target/'complete.json').exists():return
    names=json.loads((folder/'names.json').read_text())
    parts=[];weights=[]
    for f in sorted(set(range(5))-{first,second}):
        a=load(folder/f'source_{f}.npz')
        sample=load(OUT/'raw'/str(manifest()['pieces'][f]['seed'])/f'sample_{seed}.npz')
        parts.append({k:v[sample['indices']] for k,v in a.items()});weights.append(sample['weights'])
    a={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};w=np.concatenate(weights)
    del parts;gc.collect()
    for mode,count in (('reference',158),('flow',202)):
        model_path=target/f'{mode}.joblib'
        if not model_path.exists():
            model=SharedHistoryExpertMixture(names[:count],seed)
            fit_presampled(model,a['X'][:,:count],a['labels'],a['families'],w)
            joblib.dump(model,model_path)
        else:model=joblib.load(model_path)
        for f in (first,second):
            path=target/f'{mode}_held_{f}.npz'
            if path.exists():continue
            held=load(folder/f'source_{f}.npz')
            sample=load(OUT/'raw'/str(manifest()['pieces'][f]['seed'])/f'sample_{seed}.npz')
            held={k:v[sample['indices']] for k,v in held.items()}
            pred=predict_chunks(model,held['X'][:,:count])
            atomic_npz(path,experts=pred['experts'],routing=pred['routing'],labels=held['labels'],weights=sample['weights'],source=held['source'])
            del held,pred
        del model;gc.collect()
    write_json(target/'complete.json', {'training_sources':np.unique(a['source']).tolist(),
        'excluded_sources':[manifest()['pieces'][f]['seed'] for f in (first,second)],
        'hashes':{p.name:sha(p) for p in target.iterdir() if p.suffix in ('.npz','.joblib')}})
    print('nested fit complete',first,second,seed,flush=True)


def outer_prepare(fold):
    folder=OUT/f'fold_{fold}';folder.mkdir(exist_ok=True)
    fit_references(folder,sorted(set(range(5))-{fold}),pressure=False)
    if (folder/'held_bank.npy').exists():return
    a=load(FOLDS/f'fold_{fold}/features_held_out.npz')
    bank,names=additional(a,manifest()['feature_names'],joblib.load(folder/'flow_reference.joblib'))
    np.save(folder/'held_bank.npy',bank)
    frozen_write(folder/'names.json',manifest()['feature_names']+names)
    write_json(folder/'cache_hashes.json', {p.name:sha(p) for p in folder.iterdir() if p.suffix in ('.npy','.joblib','.json') and p.name!='cache_hashes.json'})
    print('outer prepared',fold,flush=True)


def fit_combination(fold,seed,mode):
    parts=[]
    for held in sorted(set(range(5))-{fold}):
        first,second=sorted((fold,held))
        parts.append(load(OUT/f'nested_{first}_{second}'/f'seed_{seed}'/f'{mode}_held_{held}.npz'))
    a={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if manifest()['pieces'][fold]['seed'] in a['source']:raise ValueError('Outer source leaked into fusion fitting')
    return ReliableCombination().fit(a['experts'],a['routing'],a['labels'],a['weights'])


def pair(fold,seed):
    folder=OUT/f'fold_{fold}';target=folder/f'seed_{seed}';target.mkdir(exist_ok=True)
    if (target/'summary.json').exists():return
    frozen_write(target/'signature.json',{'protocol_sha256':sha(OUT/'protocol_frozen.json'),'fold':fold,'seed':seed})
    if not (target/'flow.joblib').exists():
        train=load(FOLDS/f'fold_{fold}/features_train.npz')
        with np.load(FOLDS/f'fold_{fold}/features_held_out.npz') as z:held={k:z[k] for k in z.files if k!='X'}
        validate_partition(train,held,manifest(),fold)
        positive,negative=np.flatnonzero(train['labels']>.5),np.flatnonzero(train['labels']<=.5)
        chosen=np.random.default_rng(seed).choice(negative,min(60000,len(negative)),replace=False)
        subset=np.r_[positive,chosen];w=np.r_[np.ones(len(positive)),np.full(len(chosen),len(negative)/len(chosen))]
        sample_sha=hashlib.sha256(subset.tobytes()).hexdigest()
        if (fold,seed)!=(0,701):
            assert sample_sha==json.loads((reference_dir(fold,seed)/'training_sample.json').read_text())['sample_sha256']
        train={k:v[subset] for k,v in train.items()}
        bank,added=additional(train,manifest()['feature_names'],joblib.load(folder/'flow_reference.joblib'))
        model=SharedHistoryExpertMixture(manifest()['feature_names']+added,seed)
        fit_presampled(model,np.column_stack((train['X'],bank)),train['labels'],train['families'],w)
        joblib.dump(model,target/'flow.joblib')
        write_json(target/'sample.json',{'sha256':sample_sha,'rows':len(subset),'sources':np.unique(train['source']).tolist()})
        del train,held,bank,model;gc.collect()
    for arm,mode in (('fusion','reference'),('combined','flow')):
        if (target/f'{arm}.joblib').exists():continue
        ref=reference_dir(fold,seed)/('model.joblib' if (fold,seed)==(0,701) else 'observed_history.joblib')
        base=joblib.load(ref if mode=='reference' else target/'flow.joblib')
        combination=fit_combination(fold,seed,mode)
        joblib.dump(ReliableGeneralMixture(base,combination),target/f'{arm}.joblib')
    held=load(FOLDS/f'fold_{fold}/features_held_out.npz')
    bank=np.load(folder/'held_bank.npy',mmap_mode='r')
    for arm in ARMS[1:]:
        path=target/f'{arm}_predictions.npz'
        if path.exists():continue
        model=joblib.load(target/f'{arm}.joblib')
        pred=predict_chunks(model,held['X'],bank[:,:42] if arm=='fusion' else bank)
        atomic_npz(path,**pred);del model,pred;gc.collect()
    del held['X'],bank;gc.collect()
    selection=held['scenario']%2==0;diagnostic=~selection
    scope={k:held[k][diagnostic] for k in ('labels','families')}
    report={'fold':fold,'seed':seed,'results':{},'thresholds':{},'scope':'General-only source-held TRAIN development; not independent'}
    for arm in ARMS:
        path=reference_dir(fold,seed)/'observed_history_predictions.npz' if arm=='reference' else target/f'{arm}_predictions.npz'
        p=load(path);report['results'][arm]={};report['thresholds'][arm]={}
        for name,score in streams(p).items():
            rule=select(score,held,selection);report['thresholds'][arm][name]=rule
            metrics=family_scores(score[diagnostic]>rule['threshold'],scope)
            mask=diagnostic & np.isin(held['families'],(0,2))
            metrics['replay']['auprc_family_plus_clean']=float(average_precision_score(held['labels'][mask],score[mask]))
            report['results'][arm][name]=metrics
    write_json(target/'summary.json',report)
    write_json(target/'artifacts_sha256.json',{p.name:sha(p) for p in target.iterdir() if p.name!='artifacts_sha256.json'})
    print('pair complete',fold,seed,flush=True)


def aggregate():
    # Reuse the exactly declared, already verified progression calculation.
    import experiment_received_trajectory as previous
    previous.OUT,previous.ARMS,previous.SEEDS=OUT,ARMS,SEEDS
    previous.aggregate()


def all_runs(workers):
    freeze()
    def stage(action,tasks):
        def run(task):
            args=[str(v) for v in task]
            with (OUT/(action+'_'+'_'.join(args)+'.log')).open('a') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),action,*args],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(run,tasks))
    stage('raw',[(f,) for f in range(5)])
    stage('nested_prepare',EXCLUSIONS)
    stage('outer_prepare',[(f,) for f in range(5)])
    stage('nested_fit',[(a,b,s) for a,b in EXCLUSIONS for s in SEEDS])
    stage('pair',[(f,s) for f in range(5) for s in SEEDS])
    aggregate()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','raw','nested_prepare','nested_fit','outer_prepare','pair','aggregate','all'))
    p.add_argument('indices',nargs='*',type=int);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    if a.action=='all':all_runs(a.workers)
    else:globals()[a.action](*a.indices)

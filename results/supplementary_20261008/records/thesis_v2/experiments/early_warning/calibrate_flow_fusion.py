"""Conditional full-system calibration; cannot read any confirmation corpus."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import joblib
import numpy as np
from build_feature_cache import ROOT,CAMPAIGN,sha,write_json,atomic_npz,load_corpus,check_distribution,config_of
from experiment_flow_fusion import OUT,SEEDS,manifest,additional,fit_references
from screen_shared_history import load
from replicate_observed_history import frozen_write
from audit_replay_train import reference_dir
from wdn.shared_history import SharedHistoryExpertMixture,fit_presampled
from wdn.reliable_general_mixture import ReliableCombination,ReliableGeneralMixture
from wdn.observed_history import observed_history_features
from wdn.received_flow_context import flow_context_features
from wdn.run_expert_redesign import CampaignData
import calibrate_received_trajectory as protected_stage

TARGET=OUT/'full_system'
REFERENCE=CAMPAIGN/'protected_history_system_v1/seed'


def prerequisite():
    selection=json.loads((OUT/'train_selection.json').read_text())
    rounding=json.loads((OUT/'rounding_summary.json').read_text())
    if not selection['selected'] or not rounding['passes']:raise RuntimeError('TRAIN progression or sensitivity failed; calibration access prohibited')
    return selection['selected']


def freeze():
    arm=prerequisite();TARGET.mkdir(exist_ok=True)
    files=[OUT/n for n in ('protocol_frozen.json','train_selection.json','rounding_summary.json')]
    downstream=OUT/'downstream_code_frozen.json'
    files.append(downstream)
    for path,digest in json.loads(downstream.read_text())['code_sha256'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Prespecified downstream code changed: '+path)
        files.append(ROOT/path)
    files += [Path(__file__),Path(__file__).with_name('check_flow_fusion_rounding.py'),Path(__file__).with_name('calibrate_received_trajectory.py'),Path(__file__).with_name('protected_history_system.py')]
    old=json.loads((OUT/'protocol_frozen.json').read_text())['input_code_sha256']
    for path,digest in old.items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Frozen TRAIN input/code changed: '+path)
        files.append(ROOT/path)
    cm=CAMPAIGN/'features/ltown_calibration/manifest.json';files.append(cm)
    cal=json.loads(cm.read_text())
    original=CAMPAIGN/'protected_history_system_v1/protocol_frozen.json'
    original_hashes=json.loads(original.read_text())['input_code_hashes'];files.append(original)
    for piece in cal['pieces']:
        directory=ROOT/'data/thesis_v2'/piece['directory']
        check_distribution(directory,config_of(ROOT/'data/thesis_v2'/manifest()['pieces'][0]['directory']))
        for name in ('snapshots.pkl','corrupted.pkl','generate_config.yaml'):
            p=directory/name
            if sha(p)!=original_hashes[str(p.relative_to(ROOT))]:raise RuntimeError('Canonical calibration input changed')
            files.append(p)
        files.append(ROOT/piece['cache'])
    for seed in SEEDS:
        files += [REFERENCE/str(seed)/n for n in ('history.joblib','full_history_selection.json')]
        files += [REFERENCE/str(seed)/'calibration'/f"source_{p['seed']}.npz" for p in cal['pieces']]
        for fold in range(5):
            files += [reference_dir(fold,seed)/'observed_history_predictions.npz',OUT/f'fold_{fold}/seed_{seed}/flow_predictions.npz']
    frozen_write(TARGET/'protocol_frozen.json',{'arm':arm,'seeds':SEEDS,'component_seeds':[s+600 for s in SEEDS],
        'reference':'Confirmed received-pressure-history design with unchanged specialized decisions',
        'operating_grid':'Original protected temperature/shrinkage/budget grid',
        'confirmation_readiness':'Every mean and source-mean family >=.78 and each source replay improvement positive; .80 target',
        'input_code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}})
    fit_references(TARGET,list(range(5)),pressure=False)


def full_combination(seed,mode):
    parts=[]
    for fold,piece in enumerate(manifest()['pieces']):
        directory=OUT/'raw'/str(piece['seed'])
        rows=load(directory/'pool.npz')['rows'][load(directory/f'sample_{seed}.npz')['indices']]
        sample=load(directory/f'sample_{seed}.npz')
        path=reference_dir(fold,seed)/'observed_history_predictions.npz' if mode=='reference' else OUT/f'fold_{fold}/seed_{seed}/flow_predictions.npz'
        pred=load(path)
        pool=load(directory/'pool.npz')
        parts.append({'experts':pred['experts'][rows],'routing':pred['routing'][rows],
                      'labels':pool['labels'][sample['indices']],'weights':sample['weights']})
    a={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return ReliableCombination().fit(a['experts'],a['routing'],a['labels'],a['weights'])


def fit(seed,arm):
    folder=TARGET/str(seed);folder.mkdir(exist_ok=True)
    if (folder/'general.joblib').exists():return
    if arm=='fusion':
        base=joblib.load(REFERENCE/str(seed)/'history.joblib')
    else:
        a,_=load_corpus('ltown_train');names=a.pop('_feature_names')
        positive,negative=np.flatnonzero(a['labels']>.5),np.flatnonzero(a['labels']<=.5)
        chosen=np.random.default_rng(seed+600).choice(negative,min(60000,len(negative)),replace=False)
        indices=np.r_[positive,chosen];w=np.r_[np.ones(len(positive)),np.full(len(chosen),len(negative)/len(chosen))]
        a={k:v[indices] for k,v in a.items()};gc.collect()
        bank,added=additional(a,names,joblib.load(TARGET/'flow_reference.joblib'))
        base=SharedHistoryExpertMixture(names+added,seed+600)
        fit_presampled(base,np.column_stack((a['X'],bank)),a['labels'],a['families'],w)
        del a,bank;gc.collect()
    model=ReliableGeneralMixture(base,full_combination(seed,'reference' if arm=='fusion' else 'flow')) if arm in ('fusion','combined') else base
    joblib.dump(model,folder/'general.joblib')
    write_json(folder/'fit.json',{'arm':arm,'seed':seed,'component_seed':seed+600,'features':model.names,'model_sha256':sha(folder/'general.joblib')})


def cal_bank(a,names,piece,flow_reference):
    data=CampaignData(ROOT/'data/thesis_v2'/piece['directory'])
    bank=np.empty((len(a['labels']),86),np.float32);filled=np.zeros(len(bank),bool)
    for scenario in np.unique(a['scenario']):
        sid=int(scenario-piece['seed']*1000)
        if sid not in piece['scenarios']:raise ValueError('Scenario outside calibration manifest')
        rows=np.flatnonzero(a['scenario']==scenario);obs=data.scenario(sid)
        idx=sorted((i for i,s in enumerate(data.snapshots) if s.scenario_id==sid),key=lambda i:data.snapshots[i].timestep)
        q=np.stack([data.corrupted[i].flow_obs.numpy() for i in idx]);qm=np.stack([data.corrupted[i].flow_mask.numpy()>0 for i in idx])
        scale=a['X'][rows,names.index('normal_error_scale')]
        bank[rows,:42],_=observed_history_features(obs['values'],obs['mask'],obs['timestep'],a['timestep'][rows],a['node'][rows],scale)
        bank[rows,42:],_=flow_context_features(flow_reference,obs['values'],obs['mask'],q,qm,obs['timestep'],a['timestep'][rows],a['node'][rows],scale)
        filled[rows]=True
    if not filled.all():raise ValueError('Unmatched calibration row')
    return bank


def score(seed,arm):
    folder=TARGET/str(seed);cal=json.loads((CAMPAIGN/'features/ltown_calibration/manifest.json').read_text())
    model=joblib.load(folder/'general.joblib');flow=joblib.load(TARGET/'flow_reference.joblib');paths=[]
    for piece in cal['pieces']:
        path=folder/f"calibration_{piece['seed']}.npz";paths.append(path)
        if path.exists():continue
        if sha(ROOT/piece['cache'])!=piece['cache_sha256']:raise RuntimeError('Calibration cache changed')
        a=load(ROOT/piece['cache']);old=load(REFERENCE/str(seed)/'calibration'/f"source_{piece['seed']}.npz")
        for k in ('labels','families','scenario','source'):np.testing.assert_array_equal(a[k],old[k])
        bank=cal_bank(a,cal['feature_names'],piece,flow)
        if arm=='fusion':bank=bank[:,:42]
        parts=[]
        for start in range(0,len(bank),50000):
            prediction=model.predict(np.column_stack((a['X'][start:start+50000],bank[start:start+50000])))
            parts.append({k:prediction[k] for k in ('experts','routing')})
        p={k:np.concatenate([part[k] for part in parts]) for k in ('experts','routing')}
        payload={k:old[k] for k in ('labels','families','scenario','source','specialist','history_experts','history_routing')}
        payload.update(candidate_experts=p['experts'],candidate_routing=p['routing']);atomic_npz(path,**payload)
        del a,old,bank,parts,p,payload;gc.collect();print(seed,'scored',piece['seed'],flush=True)
    return paths


protected_stage.OUT=OUT
protected_stage.TARGET=TARGET
protected_stage.prerequisite=prerequisite


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('all','seed','aggregate'));p.add_argument('--seed',type=int,choices=SEEDS);args=p.parse_args()
    if args.action=='seed':
        arm=prerequisite();fit(args.seed,arm);protected_stage.select(args.seed,score(args.seed,arm))
    elif args.action=='aggregate':protected_stage.aggregate()
    else:
        freeze()
        def work(seed):
            with (TARGET/f'seed_{seed}.log').open('a') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'seed','--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,check=True)
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(work,SEEDS))
        protected_stage.aggregate()

"""Protected calibration for the prespecified 0.75 joint representation."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import joblib
import numpy as np
from build_feature_cache import ROOT,CAMPAIGN,sha,write_json,atomic_npz,load_corpus,check_distribution,config_of
from experiment_received_joint import OUT,SEEDS,manifest,additional,flow_reference_path
from screen_shared_history import load
from replicate_observed_history import frozen_write
from wdn.shared_history import SharedHistoryExpertMixture,fit_presampled
from wdn.received_joint_context import received_joint_features
from wdn.run_expert_redesign import CampaignData
from wdn.latency_deployment import FAMILY_NAMES
import calibrate_received_trajectory as protected_stage

TARGET=OUT/'full_system'
REFERENCE=CAMPAIGN/'protected_history_system_v1/seed'
OTHER=('random','drift','noise','targeted')


def prerequisite():
    if json.loads((OUT/'train_selection.json').read_text())['selected']!='joint' or not json.loads((OUT/'rounding_summary.json').read_text())['passes']:
        raise RuntimeError('TRAIN or rounding gate failed; calibration prohibited')
    return 'joint'


def freeze():
    prerequisite();TARGET.mkdir(exist_ok=True)
    files=[OUT/n for n in ('protocol_frozen.json','train_selection.json','rounding_summary.json')]
    for path,digest in json.loads((OUT/'protocol_frozen.json').read_text())['input_code_sha256'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Frozen input/code changed: '+path)
        files.append(ROOT/path)
    cm=CAMPAIGN/'features/ltown_calibration/manifest.json';files.append(cm)
    cal=json.loads(cm.read_text())
    original=CAMPAIGN/'protected_history_system_v1/protocol_frozen.json';files.append(original)
    expected=json.loads(original.read_text())['input_code_hashes']
    for piece in cal['pieces']:
        directory=ROOT/'data/thesis_v2'/piece['directory']
        check_distribution(directory,config_of(ROOT/'data/thesis_v2'/manifest()['pieces'][0]['directory']))
        for name in ('snapshots.pkl','corrupted.pkl','generate_config.yaml'):
            path=directory/name
            if sha(path)!=expected[str(path.relative_to(ROOT))]:raise RuntimeError('Canonical calibration input changed')
            files.append(path)
        files.append(ROOT/piece['cache'])
    for seed in SEEDS:
        files += [REFERENCE/str(seed)/n for n in ('history.joblib','full_history_selection.json')]
        files += [REFERENCE/str(seed)/'calibration'/f"source_{p['seed']}.npz" for p in cal['pieces']]
    shutil.copyfile(flow_reference_path(None),TARGET/'flow_reference.joblib')
    files.append(TARGET/'flow_reference.joblib')
    frozen_write(TARGET/'protocol_frozen.json',{'arm':'joint','seeds':SEEDS,'component_seeds':[s+600 for s in SEEDS],
        'operating_grid':'Unchanged protected temperature/shrinkage/budget grid and source guards',
        'confirmation_readiness':'Protected every seed; positive replay mean gain every source; mean replay gain>=.02; other-family means>=.80',
        'input_code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}})


def fit(seed):
    folder=TARGET/str(seed);folder.mkdir(exist_ok=True)
    if (folder/'general.joblib').exists():return
    a,_=load_corpus('ltown_train');names=a.pop('_feature_names')
    positive,negative=np.flatnonzero(a['labels']>.5),np.flatnonzero(a['labels']<=.5)
    chosen=np.random.default_rng(seed+600).choice(negative,min(60000,len(negative)),replace=False)
    indices=np.r_[positive,chosen];w=np.r_[np.ones(len(positive)),np.full(len(chosen),len(negative)/len(chosen))]
    a={k:v[indices] for k,v in a.items()};gc.collect()
    bank,added=additional(a,names,joblib.load(TARGET/'flow_reference.joblib'))
    model=SharedHistoryExpertMixture(names+added,seed+600)
    fit_presampled(model,np.column_stack((a['X'],bank)),a['labels'],a['families'],w)
    joblib.dump(model,folder/'general.joblib')
    write_json(folder/'fit.json',{'arm':'joint','seed':seed,'component_seed':seed+600,'features':model.names,
        'sample_sha256':hashlib.sha256(indices.tobytes()).hexdigest(), 'sample_rows':len(indices),
        'sources':np.unique(a['source']).tolist(),'model_sha256':sha(folder/'general.joblib')})


def cal_bank(a,names,piece,reference):
    data=CampaignData(ROOT/'data/thesis_v2'/piece['directory'])
    bank=np.empty((len(a['labels']),226),np.float32);filled=np.zeros(len(bank),bool)
    for scenario in np.unique(a['scenario']):
        sid=int(scenario-piece['seed']*1000)
        if sid not in piece['scenarios']:raise ValueError('Scenario outside permitted manifest')
        rows=np.flatnonzero(a['scenario']==scenario);obs=data.scenario(sid)
        idx=sorted((i for i,s in enumerate(data.snapshots) if s.scenario_id==sid),key=lambda i:data.snapshots[i].timestep)
        q=np.stack([data.corrupted[i].flow_obs.numpy() for i in idx]);qm=np.stack([data.corrupted[i].flow_mask.numpy()>0 for i in idx])
        bank[rows],_=received_joint_features(reference,obs['values'],obs['mask'],q,qm,obs['timestep'],
            a['timestep'][rows],a['node'][rows],a['X'][rows,names.index('normal_error_scale')])
        filled[rows]=True
    if not filled.all():raise ValueError('Unmatched source rows')
    return bank


def score(seed):
    folder=TARGET/str(seed);cal=json.loads((CAMPAIGN/'features/ltown_calibration/manifest.json').read_text())
    model=joblib.load(folder/'general.joblib');flow=joblib.load(TARGET/'flow_reference.joblib');paths=[]
    for piece in cal['pieces']:
        path=folder/f"calibration_{piece['seed']}.npz";paths.append(path)
        if path.exists():continue
        if sha(ROOT/piece['cache'])!=piece['cache_sha256']:raise RuntimeError('Calibration cache changed')
        a=load(ROOT/piece['cache']);old=load(REFERENCE/str(seed)/'calibration'/f"source_{piece['seed']}.npz")
        for k in ('labels','families','scenario','source'):np.testing.assert_array_equal(a[k],old[k])
        bank=cal_bank(a,cal['feature_names'],piece,flow);parts=[]
        for start in range(0,len(bank),50000):
            p=model.predict(np.column_stack((a['X'][start:start+50000],bank[start:start+50000])))
            parts.append({k:p[k] for k in ('experts','routing')})
        payload={k:old[k] for k in ('labels','families','scenario','source','specialist','history_experts','history_routing')}
        payload.update({f'candidate_{k}':np.concatenate([part[k] for part in parts]) for k in ('experts','routing')})
        atomic_npz(path,**payload)
        del a,old,bank,parts,p,payload;gc.collect();print(seed,'scored calibration',piece['seed'],flush=True)
    return paths


def decision(results):
    passed=all(r['status']=='selected' for r in results)
    report={'protected_all_seeds':passed,'confirmation_ready':False,'arm':'joint',
        'scope':'Full-system calibration; not independent',
        'reference_means':{f:float(np.mean([r['baseline'][f]['f1'] for r in results])) for f in FAMILY_NAMES.values()}}
    if passed:
        means={f:float(np.mean([r['report'][f]['f1'] for r in results])) for f in FAMILY_NAMES.values()}
        sources={s:{f:float(np.mean([r['source_reports'][s][f]['f1'] for r in results])) for f in FAMILY_NAMES.values()} for s in results[0]['source_reports']}
        delta={s:float(np.mean([r['source_reports'][s]['replay']['f1']-r['baseline_sources'][s]['replay']['f1'] for r in results])) for s in sources}
        gain=means['replay']-report['reference_means']['replay']
        checks={'mean_replay_gain_at_least_002':gain>=.02,'each_source_replay_gain_positive':min(delta.values())>0,
            'other_family_means_at_least_080':min(means[f] for f in OTHER)>=.80}
        report.update(means=means,source_means=sources,source_replay_deltas=delta,replay_gain=gain,
            pooled=float(np.mean([r['report']['_overall']['f1'] for r in results])),
            clean_fpr=float(np.mean([r['report']['_overall']['clean_fpr'] for r in results])),
            readiness_checks=checks,confirmation_ready=all(checks.values()))
    report['next']='freeze one fresh confirmation' if report['confirmation_ready'] else 'stop; retain confirmed received-history reference'
    return report


def aggregate():
    prerequisite()
    report=decision([json.loads((TARGET/str(s)/'selection.json').read_text()) for s in SEEDS])
    frozen_write(TARGET/'decision.json',report);print(json.dumps(report,indent=2),flush=True)


protected_stage.OUT=OUT;protected_stage.TARGET=TARGET;protected_stage.prerequisite=prerequisite


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('all','seed','aggregate'));p.add_argument('--seed',type=int,choices=SEEDS);args=p.parse_args()
    if args.action=='seed':
        prerequisite();fit(args.seed);protected_stage.select(args.seed,score(args.seed))
    elif args.action=='aggregate':aggregate()
    else:
        freeze()
        def work(seed):
            with (TARGET/f'seed_{seed}.log').open('a') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'seed','--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(work,SEEDS))
        aggregate()

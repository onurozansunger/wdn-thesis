"""One fresh confirmation, accessible only after protected calibration readiness.

No generation, reservation or evaluation occurs at import. Once frozen, all six
sources and all five model seeds must be evaluated without adaptive changes.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import joblib
import numpy as np
from build_feature_cache import ROOT,CAMPAIGN,CORPORA,build,sha,write_json,atomic_npz,check_distribution,config_of
from experiment_flow_fusion import OUT,SEEDS
from calibrate_flow_fusion import TARGET,REFERENCE,cal_bank
from protected_history_system import fixed_predictions,threshold_reports
from screen_router_temperature import transformed_mixture
from screen_shared_history import load
from replicate_observed_history import frozen_write
from wdn.latency_deployment import FAMILY_NAMES

FRESH=tuple(range(120811,126811,1000))
PURPOSE='ltown_flow_fusion_confirmation'
CONFIRM=OUT/'confirmation'


def ready():
    decision=json.loads((TARGET/'decision.json').read_text())
    if not decision['protected_all_seeds'] or not decision['confirmation_ready']:
        raise RuntimeError('Calibration readiness failed; fresh confirmation is prohibited')
    return decision


def freeze():
    decision=ready()
    CONFIRM.mkdir(exist_ok=True)
    from reserve_seeds import observed_seeds,KNOWN_FORBIDDEN
    reservation=CAMPAIGN/'seed_manifest.json';record=json.loads(reservation.read_text())
    if PURPOSE in record['generator_seeds']:
        if record['generator_seeds'][PURPOSE]!=list(FRESH):raise RuntimeError('Reservation changed')
    else:
        occupied=set(observed_seeds())|set(KNOWN_FORBIDDEN)
        occupied|={s for seeds in record['generator_seeds'].values() for s in seeds}
        if occupied & set(FRESH):raise RuntimeError('Fresh seed collision; no evaluation allowed')
        record['generator_seeds'][PURPOSE]=list(FRESH)
        record.setdefault('amendments',[]).append({'purpose':PURPOSE,'seeds':FRESH,'reserved_before_generation':True,
            'reason':'One qualified, frozen flow/fusion confirmation; no post-confirmation tuning'})
        write_json(reservation,record)
    files=[Path(__file__),TARGET/'decision.json',TARGET/'protocol_frozen.json',TARGET/'flow_reference.joblib',
        OUT/'protocol_frozen.json',CAMPAIGN/'stage_e_ltown/reference.joblib',ROOT/'data/L-Town.inp']
    files+=sorted((ROOT/'src/wdn').rglob('*.py'))
    files+=sorted(Path(__file__).parent.glob('*.py'))
    for protocol,key in ((OUT/'protocol_frozen.json','input_code_sha256'),(TARGET/'protocol_frozen.json','input_code_sha256')):
        for path,digest in json.loads(protocol.read_text())[key].items():
            if sha(ROOT/path)!=digest:raise RuntimeError('Frozen development input changed: '+path)
            files.append(ROOT/path)
    for seed in SEEDS:
        files += [TARGET/str(seed)/n for n in ('general.joblib','selection.json')]
        files += [REFERENCE/str(seed)/n for n in ('history.joblib','full_history_selection.json')]
        files += [CAMPAIGN/f'stage_e_ltown/seed/{seed}'/n for n in ('base_bundle.joblib','delayed_bundle.joblib','heads_bundle.joblib')]
        files.append(CAMPAIGN/f'router_temperature_screen_v1/ltown/seed/{seed}/selection_frozen.json')
    frozen_write(CONFIRM/'design_frozen.json',{'candidate':decision['arm'],'fresh_sources':FRESH,'model_seeds':SEEDS,
        'scenarios_per_source':24,'generation_after_freeze':True,'locked_test_read':False,
        'success':'At least24/30 replay wins and positive mean replay gain in every fresh source; pooled mean preserved, other-family mean loss<=.005, clean FPR mean preserved. Source pooled loss<=.005, other-family loss<=.01, FPR increase<=.0001. Absolute FPR<=.005 for every run. Report >=.80 attainment separately from >=.78 proximity; no tuning or second confirmation.',
        'input_code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}})


def check_design():
    ready()
    design=json.loads((CONFIRM/'design_frozen.json').read_text())
    for path,digest in design['input_code_sha256'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Frozen confirmation design changed: '+path)
    return design


def prepare():
    check_design()
    fingerprints={}
    for source in FRESH:
        corpus=f'{PURPOSE}_seed{source}'
        directory=f'ew_ltown_{PURPOSE}_seed{source}'
        subprocess.run([sys.executable,str(Path(__file__).with_name('generate_corpus.py')),'--network','ltown',
            '--purpose',PURPOSE,'--seed',str(source),'--scenarios','24'],cwd=ROOT,check=True)
        CORPORA[corpus]={'reference':CAMPAIGN/'stage_e_ltown/reference.joblib','network':'ltown','pieces':[(directory,source,'all:24')]}
        if not (CAMPAIGN/'features'/corpus/'manifest.json').exists():build(corpus)
        for name in ('snapshots.pkl','corrupted.pkl','generate_config.yaml','events.json'):
            path=ROOT/'data/thesis_v2'/directory/name
            fingerprints[str(path.relative_to(ROOT))]=sha(path)
        path=CAMPAIGN/'features'/corpus/'manifest.json'
        fingerprints[str(path.relative_to(ROOT))]=sha(path)
    frozen_write(CONFIRM/'data_manifest.json',{'design_sha256':sha(CONFIRM/'design_frozen.json'),'input_sha256':fingerprints})


def confirm(seed):
    design=check_design();folder=CONFIRM/str(seed);folder.mkdir(exist_ok=True)
    if (folder/'result.json').exists():return
    data_frozen=json.loads((CONFIRM/'data_manifest.json').read_text())
    assert data_frozen['design_sha256']==sha(CONFIRM/'design_frozen.json')
    for path,digest in data_frozen['input_sha256'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Fresh confirmation observations changed: '+path)
    model=joblib.load(TARGET/str(seed)/'general.joblib');baseline=joblib.load(REFERENCE/str(seed)/'history.joblib')
    flow=joblib.load(TARGET/'flow_reference.joblib')
    rule=json.loads((TARGET/str(seed)/'selection.json').read_text())['rule']
    before=json.loads((REFERENCE/str(seed)/'full_history_selection.json').read_text())['rule']
    results=[]
    for source in FRESH:
        manifest=json.loads((CAMPAIGN/'features'/f'{PURPOSE}_seed{source}'/'manifest.json').read_text());piece=manifest['pieces'][0]
        directory=ROOT/'data/thesis_v2'/piece['directory']
        check_distribution(directory,config_of(ROOT/'data/thesis_v2/ew_ltown_ltown_train_seed60811'))
        if sha(ROOT/piece['cache'])!=piece['cache_sha256']:raise RuntimeError('Confirmation cache hash mismatch')
        path=folder/f'source_{source}.npz'
        if not path.exists():
            a=load(ROOT/piece['cache']);bank=cal_bank(a,manifest['feature_names'],piece,flow)
            specialist,_=fixed_predictions(seed,a,manifest['feature_names'])
            parts=[]
            for start in range(0,len(bank),50000):
                x=a['X'][start:start+50000];b=bank[start:start+50000]
                p=baseline.predict(np.column_stack((x,b[:,:42])))
                q=model.predict(np.column_stack((x,b[:,:42] if design['candidate']=='fusion' else b)))
                parts.append({'baseline_experts':p['experts'],'baseline_routing':p['routing'],
                    'candidate_experts':q['experts'],'candidate_routing':q['routing']})
            payload={k:a[k] for k in ('labels','families','source','scenario')};payload['specialist']=specialist
            payload.update({k:np.concatenate([part[k] for part in parts]) for k in parts[0]})
            atomic_npz(path,**payload);del a,bank,parts,payload;gc.collect()
        a=load(path)
        b=transformed_mixture(a['baseline_experts'],a['baseline_routing'],before['router_temperature'],before['uniform_shrinkage'])
        c=transformed_mixture(a['candidate_experts'],a['candidate_routing'],rule['router_temperature'],rule['uniform_shrinkage'])
        results.append({'source':source,'baseline':threshold_reports(b,[before['mixture_threshold']],a)[0],
            'candidate':threshold_reports(c,[rule['mixture_threshold']],a)[0]})
        del a,b,c;gc.collect();print('confirmation scored',seed,source,flush=True)
    frozen_write(folder/'result.json',{'seed':seed,'design_sha256':sha(CONFIRM/'design_frozen.json'),
        'data_manifest_sha256':sha(CONFIRM/'data_manifest.json'),'pairs':results})
    write_json(folder/'artifacts_sha256.json',{p.name:sha(p) for p in folder.iterdir() if p.name!='artifacts_sha256.json'})


def aggregate():
    check_design()
    pairs=[p|{'seed':s} for s in SEEDS for p in json.loads((CONFIRM/str(s)/'result.json').read_text())['pairs']]
    keys=list(FAMILY_NAMES.values())+['pooled','clean_fpr']
    def value(p,arm,k):return p[arm]['_overall']['f1' if k=='pooled' else k] if k in ('pooled','clean_fpr') else p[arm][k]['f1']
    means={arm:{k:float(np.mean([value(p,arm,k) for p in pairs])) for k in keys} for arm in ('baseline','candidate')}
    delta={k:means['candidate'][k]-means['baseline'][k] for k in keys}
    source_delta={str(s):{k:float(np.mean([value(p,'candidate',k)-value(p,'baseline',k) for p in pairs if p['source']==s])) for k in keys} for s in FRESH}
    other=('random','drift','noise','targeted');wins=sum(value(p,'candidate','replay')>value(p,'baseline','replay') for p in pairs)
    checks={'replay_wins':wins>=24,'replay_every_source':min(d['replay'] for d in source_delta.values())>0,
        'pooled_preserved':delta['pooled']>=0,'clean_fpr_preserved':delta['clean_fpr']<=0,
        'other_families_preserved':min(delta[f] for f in other)>=-.005,
        'source_protection':all(d['pooled']>=-.005 and d['clean_fpr']<=.0001 and min(d[f] for f in other)>=-.01 for d in source_delta.values()),
        'absolute_fpr':all(value(p,'candidate','clean_fpr')<=.005 for p in pairs)}
    report={'means':means,'paired_deltas':delta,'source_deltas':source_delta,'replay_wins':wins,'checks':checks,
        'protected_improvement_confirmed':all(checks.values()),'all_family_mean_target_0_80':min(means['candidate'][f] for f in FAMILY_NAMES.values())>=.8,
        'all_family_mean_near_0_78':min(means['candidate'][f] for f in FAMILY_NAMES.values())>=.78,
        'replay_range':[min(value(p,'candidate','replay') for p in pairs),max(value(p,'candidate','replay') for p in pairs)],
        'pairs':pairs,'scope':'One fresh confirmation: six generator sources, five model seeds each; not30 independent datasets; no confirmation-based tuning','deployment_changed':False}
    frozen_write(CONFIRM/'summary.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='pairs'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','prepare','confirm','aggregate','all'));p.add_argument('--seed',type=int,choices=SEEDS);a=p.parse_args()
    if a.action=='confirm':confirm(a.seed)
    elif a.action=='all':
        freeze();prepare()
        def work(seed):
            with (CONFIRM/f'seed_{seed}.log').open('a') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'confirm','--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(work,SEEDS))
        aggregate()
    else:globals()[a.action]()

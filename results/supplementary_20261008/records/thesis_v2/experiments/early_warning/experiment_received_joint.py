"""Single frozen joint-feature challenger on the existing 25 TRAIN pairs."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import joblib
import numpy as np
from sklearn.metrics import average_precision_score
from build_feature_cache import ROOT, CAMPAIGN, sha, write_json, atomic_npz
from audit_replay_train import reference_dir
from replicate_observed_history import frozen_write, validate_partition
from screen_shared_history import load, predict_chunks, streams, select
from wdn.latency_deployment import family_scores
from wdn.received_joint_context import received_joint_features
from wdn.shared_history import SharedHistoryExpertMixture, fit_presampled
import experiment_flow_fusion as flow_stage
import experiment_received_trajectory as trajectory_stage

OUT = CAMPAIGN / 'received_joint_075_v1'
FOLDS = flow_stage.FOLDS
SEEDS = tuple(range(701, 706))
ARMS = ('reference', 'joint')
manifest = flow_stage.manifest


def flow_reference_path(fold):
    return flow_stage.OUT / (f'fold_{fold}' if fold is not None else 'full_system') / 'flow_reference.joblib'


def additional(a, names, reference, rounding=0.):
    bank = np.empty((len(a['labels']), 226), np.float32)
    filled = np.zeros(len(bank), bool)
    for piece in manifest()['pieces']:
        source = np.flatnonzero(a['source'] == piece['seed'])
        if not len(source): continue
        data = flow_stage.ReceivedData(piece)
        for scenario in np.unique(a['scenario'][source]):
            rows = source[a['scenario'][source] == scenario]
            obs = data.scenario(int(scenario - piece['seed'] * 1000))
            bank[rows], added = received_joint_features(reference, obs['values'], obs['mask'],
                obs['flow'], obs['flow_mask'], obs['timestep'], a['timestep'][rows], a['node'][rows],
                a['X'][rows, names.index('normal_error_scale')], rounding)
            filled[rows] = True
    if not filled.all(): raise ValueError('Query source outside frozen TRAIN manifest')
    return bank, added


def freeze():
    OUT.mkdir(exist_ok=True)
    # Verify unchanged old scientific inputs before reusing any fitted artifact.
    files = set()
    for old in (flow_stage.OUT, trajectory_stage.OUT):
        protocol = old / 'protocol_frozen.json'
        files.add(protocol)
        for path, digest in json.loads(protocol.read_text())['input_code_sha256'].items():
            p = ROOT / path
            if p in files: continue
            if sha(p) != digest: raise RuntimeError('Frozen input changed: ' + path)
            files.add(p)
    for piece in manifest()['pieces']:
        directory = flow_stage.OUT / 'raw' / str(piece['seed'])
        metadata = directory / 'complete.json'; files.add(metadata)
        for name, digest in json.loads(metadata.read_text())['hashes'].items():
            p = directory / name
            if sha(p) != digest: raise RuntimeError('Received TRAIN cache changed')
            files.add(p)
    for fold in range(5):
        tr = trajectory_stage.OUT / f'fold_{fold}'
        fl = flow_stage.OUT / f'fold_{fold}'
        if sha(tr / 'held_bank.npy') != json.loads((tr / 'cache_signature.json').read_text())['sha256']:
            raise RuntimeError('Trajectory cache changed')
        for name, digest in json.loads((fl / 'cache_hashes.json').read_text()).items():
            if sha(fl / name) != digest: raise RuntimeError('Flow cache changed')
            files.add(fl / name)
        fitted = json.loads((fl / 'fit_scope.json').read_text())
        expected = [p['seed'] for i,p in enumerate(manifest()['pieces']) if i != fold]
        if fitted['fit_sources'] != expected: raise RuntimeError('Flow reference source leakage')
        files.update([tr / 'held_bank.npy', tr / 'feature_names.json', tr / 'cache_signature.json', fl / 'cache_hashes.json'])
        for seed in SEEDS:
            files.add(trajectory_stage.OUT / f'fold_{fold}/seed_{seed}/summary.json')
            sample = flow_stage.OUT / f'fold_{fold}/seed_{seed}/sample.json'
            expected = json.loads((sample.parent / 'artifacts_sha256.json').read_text())
            if sha(sample) != expected['sample.json']: raise RuntimeError('Training sample record changed')
            files.add(sample)
    # Only the fitted normal TRAIN flow reference is reused from the old full stage.
    fingerprint = flow_stage.OUT / 'full_system/fitted_reference_fingerprint.json'
    fp = json.loads(fingerprint.read_text())
    if sha(flow_reference_path(None)) != fp['flow_reference_sha256']:
        raise RuntimeError('Full TRAIN flow reference changed')
    if fp['fit_scope']['fit_sources'] != [p['seed'] for p in manifest()['pieces']]:
        raise RuntimeError('Full flow reference fitting scope changed')
    files.update([fingerprint, flow_reference_path(None)])
    scripts = ['experiment_received_joint.py', 'check_received_joint_rounding.py',
               'calibrate_received_joint.py', 'confirm_received_joint.py', 'report_received_joint.py',
               'REPLAY_TARGET_075_PLAN.md']
    files.update(Path(__file__).with_name(n) for n in scripts)
    files.update(sorted((ROOT / 'src/wdn').rglob('*.py')))
    files.add(ROOT / 'tests/test_received_joint.py')
    # Include indirect downstream code before any new performance is scored.
    files.update(sorted(Path(__file__).parent.glob('*.py')))
    protocol = {'arms': ARMS, 'folds': list(range(5)), 'seeds': SEEDS,
        'features': {'reference': 158, 'joint': 342, 'added_bank': '42 point + 140 trajectory + 44 flow; no duplicates'},
        'training': 'Same five components, internal router, parameters, population weights, all positives and exact seed-specific 60000 negatives',
        'selection': 'All25 pairs; >=20 replay wins, positive replay source means, mean replay gain>=.01; unchanged even TRAIN threshold-selection / odd diagnostic partition',
        'rounding': '.01 and .05 m additional pressure feature paths; every fold seed701; unchanged thresholds; every source delta>=0 and mean>0',
        'calibration': 'Original protected grid and per-seed/source guards; every source mean replay gain>0; mean replay gain>=.02; other-family means>=.80',
        'confirmation': 'One fresh six-source x five-seed confirmation only after qualification; mean replay>=.75 plus paired/source/pooled/other-family/FPR protections',
        'architecture': ['General','Drift','Noise'], 'specialist_delay_hours': 3,
        'missing_pressure': .5, 'missing_flow': .5, 'generator_splits_severity_unchanged': True,
        'locked_test_read': False, 'confirmation_read': False,
        'input_code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}}
    frozen_write(OUT / 'protocol_frozen.json', protocol)
    print('Joint design and downstream code frozen', flush=True)


def prepare(fold):
    target = OUT / f'fold_{fold}'; target.mkdir(exist_ok=True)
    if (target / 'cache_signature.json').exists(): return
    tr = trajectory_stage.OUT / f'fold_{fold}'; fl = flow_stage.OUT / f'fold_{fold}'
    tn = json.loads((tr / 'feature_names.json').read_text())
    fn = json.loads((fl / 'names.json').read_text())
    assert tn[:158] == fn[:158] and len(tn) == 298 and len(fn) == 202
    names = tn + fn[158:]
    assert len(names) == len(set(names)) == 342
    trajectory = np.load(tr / 'held_bank.npy', mmap_mode='r')
    flow = np.load(fl / 'held_bank.npy', mmap_mode='r')
    for start in range(0, len(flow), 50000):
        np.testing.assert_array_equal(trajectory[start:start+50000, :42], flow[start:start+50000, :42])
    bank = np.column_stack((trajectory, flow[:,42:]))
    # Independently recompute endpoints spread over the entire held TRAIN source.
    a = load(FOLDS / f'fold_{fold}/features_held_out.npz')
    rows = np.unique(np.r_[np.linspace(0,len(bank)-1,1000,dtype=int), np.flatnonzero(a['labels']>0)[::10]])
    sample = {k:v[rows] for k,v in a.items()}
    recomputed, added = additional(sample, manifest()['feature_names'], joblib.load(flow_reference_path(fold)))
    np.testing.assert_array_equal(recomputed, bank[rows])
    assert manifest()['feature_names'] + added == names
    np.save(target / 'held_bank.npy', bank)
    frozen_write(target / 'names.json', names)
    write_json(target / 'cache_signature.json', {'sha256':sha(target/'held_bank.npy'),
        'rows':len(bank), 'recomputed_rows':len(rows), 'point_path_exact_all_rows':True,
        'protocol_sha256':sha(OUT/'protocol_frozen.json')})
    print('Prepared joint fold', fold, flush=True)


def pair(fold, seed):
    target = OUT / f'fold_{fold}/seed_{seed}'; target.mkdir(exist_ok=True)
    frozen_write(target/'signature.json', {'fold':fold, 'seed':seed, 'protocol_sha256':sha(OUT/'protocol_frozen.json')})
    if (target/'summary.json').exists(): return
    if not (target/'joint.joblib').exists():
        train = load(FOLDS / f'fold_{fold}/features_train.npz')
        with np.load(FOLDS/f'fold_{fold}/features_held_out.npz') as z:
            held = {k:z[k] for k in z.files if k!='X'}
        validate_partition(train, held, manifest(), fold)
        positive, negative = np.flatnonzero(train['labels']>.5), np.flatnonzero(train['labels']<=.5)
        chosen = np.random.default_rng(seed).choice(negative,min(60000,len(negative)),replace=False)
        subset = np.r_[positive,chosen]
        weight = np.r_[np.ones(len(positive)),np.full(len(chosen),len(negative)/len(chosen))]
        digest = hashlib.sha256(subset.tobytes()).hexdigest()
        expected = json.loads((flow_stage.OUT/f'fold_{fold}/seed_{seed}/sample.json').read_text())
        assert digest == expected['sha256']
        train = {k:v[subset] for k,v in train.items()}; del held; gc.collect()
        bank, added = additional(train,manifest()['feature_names'],joblib.load(flow_reference_path(fold)))
        model = SharedHistoryExpertMixture(manifest()['feature_names']+added,seed)
        fit_presampled(model,np.column_stack((train['X'],bank)),train['labels'],train['families'],weight)
        joblib.dump(model,target/'joint.joblib')
        write_json(target/'training_sample.json',{'sha256':digest,'sources':np.unique(train['source']).tolist(),'rows':len(subset)})
        del train,bank,model; gc.collect()
    held = load(FOLDS/f'fold_{fold}/features_held_out.npz')
    prediction = target/'joint_predictions.npz'
    if not prediction.exists():
        bank = np.load(OUT/f'fold_{fold}/held_bank.npy',mmap_mode='r')
        model = joblib.load(target/'joint.joblib')
        atomic_npz(prediction,**predict_chunks(model,held['X'],bank))
        del bank,model
    del held['X']; gc.collect()
    selection = held['scenario']%2==0; diagnostic = ~selection
    scope = {k:held[k][diagnostic] for k in ('labels','families')}
    report = {'fold':fold,'seed':seed,'results':{},'thresholds':{},'scope':'TRAIN General-only development; not independent'}
    old = json.loads((trajectory_stage.OUT/f'fold_{fold}/seed_{seed}/summary.json').read_text())
    for arm in ARMS:
        path = reference_dir(fold,seed)/'observed_history_predictions.npz' if arm=='reference' else prediction
        p = load(path); report['results'][arm]={}; report['thresholds'][arm]={}
        for name, score in streams(p).items():
            rule = select(score,held,selection)
            report['thresholds'][arm][name]=rule
            metrics = family_scores(score[diagnostic]>rule['threshold'],scope)
            mask = diagnostic & np.isin(held['families'],(0,2))
            metrics['replay']['auprc_family_plus_clean']=float(average_precision_score(held['labels'][mask],score[mask]))
            report['results'][arm][name]=metrics
            if arm=='reference':
                assert rule==old['thresholds']['reference'][name]
                assert metrics==old['results']['reference'][name]
    write_json(target/'summary.json',report)
    write_json(target/'artifacts_sha256.json',{p.name:sha(p) for p in target.iterdir() if p.name!='artifacts_sha256.json'})
    print('Completed joint pair',fold,seed,flush=True)


def aggregate():
    trajectory_stage.OUT=OUT; trajectory_stage.ARMS=ARMS; trajectory_stage.SEEDS=SEEDS
    trajectory_stage.aggregate()


def work(action, tasks, workers):
    def one(args):
        logpath=OUT/(action+'_'+'_'.join(map(str,args))+'.log')
        command=[sys.executable,str(Path(__file__).resolve()),action,'--fold',str(args[0])]
        if len(args)>1: command+=['--seed',str(args[1])]
        with logpath.open('a') as log: subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    with ThreadPoolExecutor(max_workers=workers) as pool: list(pool.map(one,tasks))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','prepare','pair','aggregate','all'))
    p.add_argument('--fold',type=int,choices=range(5));p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--workers',type=int,default=3)
    args=p.parse_args()
    if args.action=='all':
        freeze();work('prepare',[(f,) for f in range(5)],args.workers)
        work('pair',[(f,s) for f in range(5) for s in SEEDS],args.workers);aggregate()
    elif args.action=='pair':pair(args.fold,args.seed)
    elif args.action=='prepare':prepare(args.fold)
    else:globals()[args.action]()

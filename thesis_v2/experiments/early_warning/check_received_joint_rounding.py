"""Frozen pressure feature-path checks; no threshold reselection."""
import argparse
import json
import joblib
import numpy as np
from experiment_received_joint import OUT,FOLDS,manifest,additional,flow_reference_path
from screen_shared_history import load,predict_chunks
from build_feature_cache import write_json,sha
from wdn.latency_deployment import family_scores


def run(fold):
    if json.loads((OUT/'train_selection.json').read_text())['selected']!='joint':
        raise RuntimeError('No qualifying TRAIN candidate')
    a=load(FOLDS/f'fold_{fold}/features_held_out.npz')
    target=OUT/f'fold_{fold}/seed_701';summary=json.loads((target/'summary.json').read_text())
    model=joblib.load(target/'joint.joblib');reference=joblib.load(flow_reference_path(fold))
    threshold=summary['thresholds']['joint']['mixture']['threshold'];rows=a['scenario']%2==1
    result={'fold':fold,'seed':701,'candidate':'joint','fixed_threshold':threshold,
        'reference':summary['results']['reference']['mixture'],'rounded':{},
        'selection_sha256':sha(OUT/'train_selection.json'),
        'scope':'Additional pressure feature paths only; original116 inputs and flows unchanged'}
    for resolution in (.01,.05):
        bank,_=additional(a,manifest()['feature_names'],reference,resolution)
        score=predict_chunks(model,a['X'],bank)['mixture']
        result['rounded'][str(resolution)]=family_scores(score[rows]>threshold,{k:a[k][rows] for k in ('labels','families')})
    write_json(OUT/f'rounding_fold_{fold}.json',result)


def aggregate():
    import check_trajectory_rounding as previous
    previous.OUT=OUT;previous.aggregate()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5));a=p.parse_args()
    aggregate() if a.fold is None else run(a.fold)

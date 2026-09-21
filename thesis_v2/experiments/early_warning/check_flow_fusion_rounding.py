"""Predeclared pressure-path sensitivity for one TRAIN-selected candidate."""
import argparse
import json
import joblib
import numpy as np
from experiment_flow_fusion import OUT,FOLDS,manifest,additional
from build_feature_cache import write_json,sha
from screen_shared_history import load,predict_chunks
from wdn.latency_deployment import family_scores


def run(fold):
    selection=json.loads((OUT/'train_selection.json').read_text());arm=selection['selected']
    if not arm:raise RuntimeError('No selected design')
    a=load(FOLDS/f'fold_{fold}/features_held_out.npz')
    target=OUT/f'fold_{fold}/seed_701';summary=json.loads((target/'summary.json').read_text())
    model=joblib.load(target/f'{arm}.joblib');flow=joblib.load(OUT/f'fold_{fold}/flow_reference.joblib')
    threshold=summary['thresholds'][arm]['mixture']['threshold'];rows=a['scenario']%2==1
    report={'fold':fold,'candidate':arm,'seed':701,'fixed_threshold':threshold,
        'reference':summary['results']['reference']['mixture'],'rounded':{},
        'selection_sha256':sha(OUT/'train_selection.json'),'scope':'Additional pressure feature paths only; flows and original116 features unchanged'}
    for resolution in (.01,.05):
        bank,_=additional(a,manifest()['feature_names'],flow,resolution)
        if arm=='fusion':bank=bank[:,:42]
        score=predict_chunks(model,a['X'],bank)['mixture']
        report['rounded'][str(resolution)]=family_scores(score[rows]>threshold,{k:a[k][rows] for k in ('labels','families')})
    write_json(OUT/f'rounding_fold_{fold}.json',report)


def aggregate():
    import check_trajectory_rounding as old
    old.OUT=OUT;old.aggregate()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5));a=p.parse_args()
    aggregate() if a.fold is None else run(a.fold)

import copy
import sys
from pathlib import Path
import numpy as np
from wdn.received_joint_context import received_joint_features
from wdn.received_flow_context import ReceivedFlowReference,flow_context_features
from wdn.received_trajectory import received_trajectory_features


def fixture():
    rng=np.random.default_rng(75)
    flow=rng.normal(size=(80,20));pressure=30+flow[:,:4]@rng.normal(size=(4,18))
    pm=rng.random(pressure.shape)>.5;qm=rng.random(flow.shape)>.5
    ref=ReceivedFlowReference().fit(pressure[:30],pm[:30],flow[:30],qm[:30])
    pm[45,0]=True
    return ref,pressure,pm,flow,qm,np.arange(80),np.array([45]),np.array([0]),np.ones(1)


def test_joint_preserves_both_frozen_representations_and_masks():
    args=fixture();a,names=received_joint_features(*args)
    ref,p,pm,q,qm,t,qt,qn,s=args
    tr,tn=received_trajectory_features(p,pm,t,qt,qn,s)
    fl,fn=flow_context_features(ref,p,pm,q,qm,t,qt,qn,s)
    np.testing.assert_array_equal(a[:,:182],tr);np.testing.assert_array_equal(a[:,182:],fl)
    assert a.shape==(1,226) and names==tn+fn and len(set(names))==226
    p=p.copy();q=q.copy();p[~pm]=np.nan;q[~qm]=np.nan
    b,_=received_joint_features(ref,p,pm,q,qm,t,qt,qn,s)
    np.testing.assert_array_equal(a,b)


def test_joint_future_values_and_masks_do_not_change_current_features():
    args=fixture();a,_=received_joint_features(*args)
    ref,p,pm,q,qm,t,qt,qn,s=args
    p[46:]+=100;q[46:]-=100;pm[46:]=~pm[46:];qm[46:]=~qm[46:]
    b,_=received_joint_features(ref,p,pm,q,qm,t,qt,qn,s)
    np.testing.assert_array_equal(a,b)


def test_joint_schema_reaches_every_existing_component():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'thesis_v2/experiments/early_warning'))
    from experiment_received_joint import manifest
    from wdn.shared_history import SharedHistoryExpertMixture
    _,names=received_joint_features(*fixture())
    model=SharedHistoryExpertMixture(manifest()['feature_names']+names,701)
    assert len(model.names)==342 and len(model.profiles)==5
    for columns in model.profiles:assert set(range(116,342)).issubset(columns)


def test_readiness_requires_protection_effect_size_and_every_source():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'thesis_v2/experiments/early_warning'))
    from calibrate_received_joint import decision
    def metrics(replay):
        return {**{f:{'f1':.9} for f in ('random','drift','noise','targeted')},
            'replay':{'f1':replay},'_overall':{'f1':.9,'clean_fpr':.0005}}
    rows=[{'status':'selected','baseline':metrics(.7),'report':metrics(.721),
           'source_reports':{'1':metrics(.721),'2':metrics(.721)},
           'baseline_sources':{'1':metrics(.7),'2':metrics(.7)}} for _ in range(5)]
    assert decision(rows)['confirmation_ready']
    for mutate in (lambda r:r[0].update(status='no_protected_operating_point'),
        lambda r:[x.update(report=metrics(.719)) for x in r],
        lambda r:[x['source_reports'].update({'1':metrics(.699)}) for x in r],
        lambda r:[x['report']['noise'].update(f1=.799) for x in r]):
        changed=copy.deepcopy(rows);mutate(changed)
        assert not decision(changed)['confirmation_ready']

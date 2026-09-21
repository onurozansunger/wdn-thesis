import sys
from pathlib import Path
import numpy as np
import pytest
from wdn.received_flow_context import ReceivedFlowReference, flow_context_features
from wdn.reliable_general_mixture import ReliableCombination, ReliableGeneralMixture


def fixture():
    rng=np.random.default_rng(24)
    q=rng.normal(size=(90,20));p=q[:,:4]@rng.normal(size=(4,18))+30
    pm=rng.random(p.shape)>.5;qm=rng.random(q.shape)>.5
    reference=ReceivedFlowReference().fit(p,pm,q,qm)
    return reference,p,pm,q,qm


def test_missing_flow_values_do_not_enter_fit_or_inference():
    ref,p,pm,q,qm=fixture()
    changed=q.copy();changed[~qm]=np.nan
    changed_p=p.copy();changed_p[~pm]=np.nan
    other=ReceivedFlowReference().fit(changed_p,pm,changed,qm)
    np.testing.assert_allclose(ref.coef,other.coef)
    for a,b in zip(ref.predict(q,qm),other.predict(changed,qm)):
        np.testing.assert_allclose(a,b,equal_nan=True)


def test_flow_features_causal_and_quantized():
    ref,p,pm,q,qm=fixture();pm[40,0]=True
    args=(ref,p,pm,q,qm,np.arange(90),np.array([40]),np.array([0]),np.array([.7]))
    a,names=flow_context_features(*args)
    future_p=p.copy();future_q=q.copy();future_p[41:]+=100;future_q[41:]-=100
    b,_=flow_context_features(ref,future_p,pm,future_q,qm,*args[5:])
    np.testing.assert_array_equal(a,b)
    assert a.shape==(1,44) and len(names)==44 and len(set(names))==44
    cols=[j for j,n in enumerate(names) if not any(s in n for s in ('coverage','available'))]
    finite=a[:,cols][np.isfinite(a[:,cols])]
    np.testing.assert_allclose(finite/.1,np.rint(finite/.1),atol=.001)


def test_missing_timestamps_and_current_query_rejected():
    ref,p,pm,q,qm=fixture();pm[40,0]=True
    keep=np.arange(90)!=39
    a,names=flow_context_features(ref,p[keep],pm[keep],q[keep],qm[keep],np.arange(90)[keep],np.array([40]),np.array([0]),np.ones(1))
    assert a[0,names.index('history_flow_pair_available_1')]==0
    assert np.isnan(a[0,names.index('history_flow_change_error_1')])
    with pytest.raises(ValueError,match='exact timestamp'):
        flow_context_features(ref,p[keep],pm[keep],q[keep],qm[keep],np.arange(90)[keep],np.array([39]),np.array([0]),np.ones(1))


def test_no_received_flows_means_unavailable():
    ref,p,pm,q,qm=fixture();pm[40,0]=True;qm[40]=False
    a,names=flow_context_features(ref,p,pm,q,qm,np.arange(90),np.array([40]),np.array([0]),np.ones(1))
    assert a[0,names.index('history_flow_available')]==0
    assert np.isnan(a[0,names.index('history_flow_residual')])


def test_reliable_combination_normalized_and_label_free():
    rng=np.random.default_rng(13);y=rng.binomial(1,.2,3000)
    p=1/(1+np.exp(-(rng.normal(size=(3000,5))+y[:,None]*2-1)))
    r=rng.dirichlet(np.ones(5),len(y));w=np.ones(len(y))
    model=ReliableCombination().fit(p,r,y,w)
    result=model.transform({'experts':p,'routing':r})
    assert model.calibration.shape==(5,2)
    assert (model.calibration[:,0]>0).all()
    np.testing.assert_allclose(result['routing'].sum(1),1)
    assert np.isfinite(result['mixture']).all()
    assert ((result['mixture']>=0)&(result['mixture']<=1)).all()
    np.testing.assert_allclose(result['mixture'],(result['experts']*result['routing']).sum(1))


def test_nested_exclusion_roles():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'thesis_v2/experiments/early_warning'))
    from experiment_flow_fusion import EXCLUSIONS
    assert len(EXCLUSIONS)==10
    for outer in range(5):
        roles=[]
        for inner in set(range(5))-{outer}:
            excluded=tuple(sorted((outer,inner)))
            assert excluded in EXCLUSIONS
            training=set(range(5))-set(excluded)
            assert outer not in training and inner not in training and len(training)==3
            roles.append(inner)
        assert len(set(roles))==4

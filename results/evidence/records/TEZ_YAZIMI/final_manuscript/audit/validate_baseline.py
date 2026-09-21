"""Independent audit of the frozen comparison and matched endpoint populations."""
from pathlib import Path
import hashlib,json,statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
C=ROOT/'runs/operational/early_warning_multiseed_v1'
B=C/'single_model_baseline_v1'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=read(B/'protocol_frozen.json'); freeze=read(B/'evaluation_frozen.json'); summary=read(B/'summary.json')
assert sha(C/'final_ten_model_seeds_v1/final_results.json')=='950f39e9fe229b25ed7137931d5cc2e2346c932bc81b83b748487e697d513c2b'
assert summary['protocol_sha256']==sha(B/'protocol_frozen.json')
assert summary['evaluation_freeze_sha256']==sha(B/'evaluation_frozen.json')
freeze_time=(B/'evaluation_frozen.json').stat().st_mtime
for path,digest in freeze['model_rule_sha256'].items():
    p=ROOT/path
    assert sha(p)==digest
    assert p.stat().st_mtime<=freeze_time
assert len(freeze['model_rule_sha256'])==120
F=['random','replay','drift','noise','targeted']
rows_checked=0
for n in ['modena','ltown']:
    for source in protocol['sources'][n]:
        path=B/n/f'evaluation_source_{source}.json'; d=read(path)
        assert path.stat().st_mtime>=freeze_time
        assert len(d['rows'])==40
        assert len({(r['model_seed'],r['delay_hours'],r['objective']) for r in d['rows']})==40
        by_key={(r['model_seed'],r['delay_hours'],r['objective']):r for r in d['rows']}
        with np.load(B/n/f'evaluation_predictions_{source}.npz') as scores:
            labels=scores['labels']>0; families=scores['families']
            assert len(labels)==d['endpoint_count']
            for seed in range(701,711):
                for delta in [0,3]:
                    probability=scores[f'seed{seed}_delta{delta}']
                    assert np.isfinite(probability).all()
                    for objective in ['pooled','balanced']:
                        row=by_key[(seed,delta,objective)]
                        predicted=probability>row['threshold']
                        for code,name in [(None,'_overall'),*enumerate(F,1)]:
                            mask=np.ones(len(labels),bool) if code is None else families==code
                            y=labels[mask]; p=predicted[mask]; v=row['metrics'][name]
                            assert v['tp']==int(np.count_nonzero(p[y]))
                            assert v['fp']==int(np.count_nonzero(p[~y]))
                            assert v['fn']==int(np.count_nonzero(~p[y]))
                            assert v['tn']==int(np.count_nonzero(~p[~y]))
        for row in d['rows']:
            seed=row['model_seed']; old=C if seed<706 else C/'final_ten_model_seeds_v1'
            m=row['metrics']; overall=m['_overall']
            if n=='modena':
                h=read(old/'stage_e_modena/seed'/str(seed)/'evaluation_report.json')['per_data_seed'][str(source)]
                assert h['rows']==d['endpoint_count']
                assert overall['tp']+overall['fn']==h['arms']['candidate']['tp']+h['arms']['candidate']['fn']
            else:
                h=read(old/'protected_history_system_v1/seed'/str(seed)/'confirmation.json')
                h=next(x['candidate'] for x in h['pairs'] if x['source']==source)
                for f in F+['_overall']:
                    assert m[f]['tp']+m[f]['fn']==h[f]['tp']+h[f]['fn']
            for f,v in m.items():
                assert abs(v['f1']-2*v['tp']/max(1,2*v['tp']+v['fp']+v['fn']))<1e-12
            assert sum(overall[k] for k in ['tp','fp','fn','tn'])==d['endpoint_count']
            assert abs(overall['clean_fpr']-overall['clean_period_fp']/overall['clean_rows'])<1e-12
            assert abs(overall['family_macro_f1']-statistics.mean(m[f]['f1'] for f in F))<1e-12
            point=read(B/n/str(seed)/f"delta{row['delay_hours']}_selection.json")
            assert row['threshold']==point['objectives'][row['objective']]['threshold']
            rows_checked+=1
assert rows_checked==480
result={'status':'passed','model_fits':40,'operating_points':80,'source_model_operating_point_reports':480,
        'matched_endpoint_populations':True,'counts_recomputed':True,
        'counts_recomputed_from_all_saved_probabilities':True,
        'all_models_and_rules_predate_evaluation_freeze':True,'all_evaluation_reports_follow_freeze':True,
        'final_hybrid_results_unchanged':True,'summary_sha256':sha(B/'summary.json'),
        'locked_test_read':False,'new_inference_in_audit':False}
Path(__file__).with_name('baseline_integrity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

from pathlib import Path
import hashlib,json,statistics
root=Path(__file__).resolve().parent
items=json.loads((root/'manifest.json').read_text())
for item in items:
    assert hashlib.sha256((root/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
base=root/'records/runs/operational/early_warning_multiseed_v1/protected_history_system_v1/seed'
b,h=[],[]
for seed in range(701,706):
    d=json.loads((base/str(seed)/'confirmation.json').read_text())
    for p in d['pairs']:
        for arm,values in [('baseline',b),('candidate',h)]:
            r=p[arm]['replay'];f=2*r['tp']/(2*r['tp']+r['fp']+r['fn'])
            assert abs(f-r['f1'])<1e-12
            values.append(f)
assert len(b)==len(h)==30
print('Verified',len(items),'record hashes.')
print('Original confirmation: baseline',statistics.mean(b),'history',statistics.mean(h))
print('Positive paired differences:',sum(y>x for x,y in zip(b,h)),'/ 30')

# Supplementary baseline and historical latency verification
import math
base=root/'records/runs/operational/early_warning_multiseed_v1/single_model_baseline_v1'
summary=json.loads((base/'summary.json').read_text())
protocol=json.loads((base/'protocol_frozen.json').read_text())
freeze=json.loads((base/'evaluation_frozen.json').read_text())
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert summary['protocol_sha256']==digest(base/'protocol_frozen.json')
assert summary['evaluation_freeze_sha256']==digest(base/'evaluation_frozen.json')
assert freeze['all_40_models_and_80_operating_points_frozen']
families=['random','replay','drift','noise','targeted']
for network in ['modena','ltown']:
    records=[]
    for source in protocol['sources'][network]:
        doc=json.loads((base/network/f'evaluation_source_{source}.json').read_text())
        assert doc['evaluation_freeze_sha256']==digest(base/'evaluation_frozen.json')
        assert len(doc['rows'])==40
        records.extend(doc['rows'])
        for row in doc['rows']:
            metrics=row['metrics']; total=metrics['_overall']
            for metric in metrics.values():
                f1=2*metric['tp']/max(1,2*metric['tp']+metric['fp']+metric['fn'])
                assert abs(f1-metric['f1'])<1e-12
            assert sum(total[k] for k in ['tp','fp','fn','tn'])==doc['endpoint_count']
            assert abs(total['clean_fpr']-total['clean_period_fp']/total['clean_rows'])<1e-12
            assert abs(total['family_macro_f1']-statistics.mean(metrics[f]['f1'] for f in families))<1e-12
            folder=base/network/str(row['model_seed'])
            point=json.loads((folder/f"delta{row['delay_hours']}_selection.json").read_text())
            assert point['objectives'][row['objective']]['threshold']==row['threshold']
    for delay in [0,3]:
        for objective in ['pooled','balanced']:
            chosen=[r for r in records if r['delay_hours']==delay and r['objective']==objective]
            assert len(chosen)==60
            target=summary['networks'][network][f'delta{delay}_{objective}']['mean']
            for name in families+['family_macro_f1','pooled_f1','clean_fpr']:
                def value(row):
                    m=row['metrics']
                    return m[name]['f1'] if name in families else m['_overall']['f1' if name=='pooled_f1' else name]
                assert abs(statistics.mean(map(value,chosen))-target[name])<1e-12
historical=root/'records/runs/operational/latency_deployment_v4'
delay=json.loads((historical/'eval_report.json').read_text())
assert delay['selection_sha256']==digest(historical/'selection_frozen.json')
for arm in ['selected','zero_latency']:
    for name in families:
        m=delay['families'][arm][name]
        assert abs(m['f1']-2*m['tp']/(2*m['tp']+m['fp']+m['fn']))<1e-12
assert delay['rows']==3015103 and len(delay['eval_seeds'])==6
print('Verified all supplementary baseline counts, 80 fixed thresholds and 8 arm means; historical latency counts checked.')

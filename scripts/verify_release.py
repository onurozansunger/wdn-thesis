"""Verify the published snapshot and recompute final means without model fitting."""
from pathlib import Path
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / 'results/final'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    manifest = json.loads((ROOT / 'results/release_manifest.json').read_text())
    for item in manifest:
        require(digest(ROOT / item['path']) == item['sha256'], f"Hash mismatch: {item['path']}")
    report = json.loads((FINAL / 'final_results.json').read_text())
    protocol = json.loads((FINAL / 'protocol_frozen.json').read_text())
    require(report['status'] == 'finalized', 'Finalization incomplete')
    require(report['locked_test_read'] is False, 'Locked test was read')
    require(report['protocol_sha256'] == digest(FINAL / 'protocol_frozen.json'), 'Protocol changed')
    require(report['evaluation_freeze_sha256'] == digest(FINAL / 'evaluation_frozen.json'), 'Evaluation freeze changed')
    families = ['random', 'replay', 'drift', 'noise', 'targeted']
    for network, values in report['networks'].items():
        rows = values['rows']
        expected = {(seed, source) for seed in protocol['model_seeds']
                    for source in protocol['evaluation_source_seeds'][network]}
        require(len(rows) == len(expected) == 60, f'{network}: wrong row count')
        require({(r['model_seed'], r['source_seed']) for r in rows} == expected,
                f'{network}: missing or duplicate cells')
        for metric in families + ['pooled_f1', 'clean_fpr']:
            mean = statistics.mean(r[metric] for r in rows)
            require(math.isclose(mean, values['mean'][metric], rel_tol=0, abs_tol=1e-12),
                    f'{network}: {metric} mean mismatch')
        macro = statistics.mean(values['mean'][f] for f in families)
        print(f'{network}: 60 cells verified; family macro F1={macro:.6f}; '
              f"pooled F1={values['mean']['pooled_f1']:.6f}")
    rows = report['networks']['ltown']['rows']
    require(all(r['replay'] > r['replay_baseline'] for r in rows), 'History pair regression')
    print(f'Verified {len(manifest)} release file hashes and both frozen result summaries.')

if __name__ == '__main__':
    main()

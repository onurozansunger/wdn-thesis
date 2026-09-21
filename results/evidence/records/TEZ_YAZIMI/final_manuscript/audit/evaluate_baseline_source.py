"""Run one predeclared source using the unchanged frozen evaluator.

Only the scheduling list is narrowed. Original protocol/model verification,
feature construction, inference, thresholds and reporting remain unchanged.
The main runner still requires all six sources in each network.
"""
import argparse
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'thesis_v2/experiments/early_warning'))
import single_model_baseline as baseline

parser=argparse.ArgumentParser()
parser.add_argument('--network',choices=baseline.NETWORKS,required=True)
parser.add_argument('--source',type=int,required=True)
args=parser.parse_args()
original_check=baseline.check_evaluation
protocol=original_check()
assert args.source in protocol['sources'][args.network]
record=baseline.OUT/'execution'/f'{args.network}_{args.source}.json'
baseline.frozen(record,{
    'operation':'schedule one already-declared evaluation source',
    'network':args.network,'source':args.source,
    'evaluation_freeze_sha256':baseline.sha(baseline.OUT/'evaluation_frozen.json'),
    'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'scientific_recipe_unchanged':True})
def checked_subset():
    result=original_check()
    return dict(result,sources={**result['sources'],args.network:[args.source]})
baseline.check_evaluation=checked_subset
baseline.evaluate(args.network)

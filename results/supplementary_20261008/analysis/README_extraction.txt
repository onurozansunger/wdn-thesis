Frozen hybrid score extraction, 8 October 2026

Purpose: recover continuous branch outputs from existing trained models on the
same calibration/evaluation sources used in the submitted thesis. No fitting,
new corpus generation, locked-test reading, original result writes or threshold
selection occur in this utility.

Runtime: /opt/miniconda3/bin/python (existing project NumPy/LightGBM/sklearn).
The approved run uses three processes concurrently after memory and equivalence
checks on this 14-core, 48-GB host. Each predicts with four tree threads,
checks accelerated outputs against original-thread predictions on up to 512
rows per call, and falls back to original threads on any bitwise difference.

Example (from workspace root):
  /opt/miniconda3/bin/python output/supervisor_revision_20261008/analysis/extract_hybrid_scores.py --network modena --seed 701 --role evaluation --source 40811 --dry-run
Remove --dry-run for inference. Omit --source to process every original source.
The default --role both processes calibration first, then evaluation.

Output: analysis/hybrid_scores/NETWORK/seedSEED/ROLE_sourceSOURCE.npz and .json.
NPZ fields: original endpoint metadata (labels, families, source, scenario,
event, timestep, node); float64 general/drift/noise; float64 verifier_drift and
verifier_noise; bool veto_drift/noise, allow_drift/noise, original_decision.
Float64 is preserved losslessly. The allow masks combine the original veto and
verifier cutoffs. Sweeping detection thresholds with these fixed masks requires
no subsequent model execution. JSON records original thresholds, cutoff values,
hashes, original confusion counts, alignment checks and prediction provenance.

L-Town General probabilities are reconstructed from retained expert/router
arrays using the exact frozen temperature/shrinkage rule. Its retained Boolean
specialist union must match the recomputed union exactly. Modena General is
scored with its frozen model. Source-wise processing retains complete scenarios
and therefore respects all temporal windows and network-time aggregation.

Every feature input is checked against its manifest hash. Every fitted bundle
and decision rule is checked against the ten-seed frozen provenance. Imported
project Python code is checked against its frozen source hashes. The baseline
metadata must align exactly by labels/families/source/scenario/timestep/node.

Evaluation sources pass historical confusion-count equality before publication.
Modena retains exact pooled TP/FP/FN/clean-FP and rounded family F1, so all those
are checked; it did not retain per-family evaluation counts to compare exactly.
L-Town retains per-family counts; all available counts/ratios are compared.
Calibration source reports are checked where available, then pooled calibration
counts/metrics are checked against the original selection after all pieces exist.
Do not perform a new analysis before all its required calibration_validation.json
files and evaluation source check results have passed.

Re-running identical extraction reuses outputs with matching script/model hashes.
Incomplete or differing output is an error requiring inspection, not overwritten.
The utility does not declare or optimize any new threshold policy; that is a
separate documented supplementary analysis, not an intrinsic hybrid probability.

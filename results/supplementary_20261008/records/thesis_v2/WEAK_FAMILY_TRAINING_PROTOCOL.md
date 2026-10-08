# Weak-family campaign v1 — frozen development protocol

Authorised by the user: train toward F1 >= 0.80 separately for drift and noise,
then diagnose remaining failures if the target is not reached.

- New campaign: `runs/operational/weak_family_v1`; at most FOUR full candidates.
  Old GNN/redesign studies stay complete and unchanged.
- Fixed operational seed811 data, original outer splits, pressure/flow missing
  probabilities 0.50. No regeneration, severity edits, test extraction or test
  scoring. No cloud or paid resources.
- Preserve the previous tree control's general/abrupt/replay experts and their
  scenario-out-of-fold training scores. Refit only drift/noise experts; original
  pressure reference and features remain available as protected inputs.
- Candidate 0: nonlinear drift/noise experts on existing pressure features.
  Candidate 1: add group-excluded, causal multi-sensor pressure evidence.
  Candidate 2: add a TRAIN-only joint pressure/flow reference and its residual
  and multi-sensor evidence. Flow is noisy and attacked, not trusted truth.
  Candidate 3: tune the representation selected from candidates 0–2 by
  calibration only, with Optuna regularisation/tree-size/early-phase weighting.
- Joint reference is refitted inside the original three whole-scenario TRAIN
  folds. All group-conditioned fitting and contextual evidence exclude the
  target group. No target sensor lists or true event boundaries enter features.
- Add causal expert-score memory; reset by scenario, not true attack boundaries.
  Fusion is fitted on scenario-out-of-fold TRAIN expert outputs and constrained
  to nonnegative expert/memory contributions. This is an expert stack, not a
  claimed normalised neural MoE gate. OOF fusion fitting scores are not reported
  as independent validation performance.
- Event-balanced learning already exists; any early-phase weight uses only
  TRAIN annotations, never calibration/validation labels as model inputs.
- One global calibration threshold. Maximise the worst attack-family F1 with
  normal and clean FPR <= 0.005 and replay F1 >= 0.50. Strong-family results are
  checked and any regression is disclosed, not hidden by averages.
- Freeze all candidate models and thresholds and choose the candidate BEFORE
  extracting any new validation features or scoring any validation predictions.
  Previous validation informed this development design; no independence claim.
- Disclose all candidates, independent expert AUPRC, clean false alarms, early
  recall and post-event false alarms. Do not replace the selected candidate with
  a validation winner. If the target fails, diagnose rather than silently add
  trials or change the dataset. Existing models are not replaced by default.

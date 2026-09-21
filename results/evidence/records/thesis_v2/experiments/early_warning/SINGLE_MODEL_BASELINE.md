# Frozen supplementary single-classifier comparison

14 September 2026. Added during thesis review, after the final hybrid results were known. This is a supplementary comparison on the existing evaluation sources, not a fresh independent confirmation or a new model-selection campaign.

## Question and controls

Does a single general-purpose classifier achieve the hybrid's joint attack-family and false-alarm profile when given the retained feature representation and an equal maximum information allowance?

Fit one LightGBM binary classifier per network, seed and information allowance. Controls are 0 hours and 3 hours; both are mandatory and reported. The 3-hour control is the primary complexity comparison. Neither uses expert scores, routing, feedback, early-warning heads, verification or a replay-specific decision rule. The deterministic TRAIN-fitted normal reference remains shared: “single classifier” does not mean an end-to-end raw-sensor model with no learned reference.

The input bank is the retained 116 causal columns (Modena), or those columns plus the frozen 42 received-pressure-history columns (L-Town). At three hours add the existing 59 bounded-window columns, computed on the full observed series before endpoint sampling. The resulting widths are 116/175 and 158/217. Labels, attack-family identities and event boundaries are never input features. Missingness, generator distributions, attack severity, source/scenario splits and the locked test remain unchanged. This matches available observation types and deadlines, but does not isolate routing from every difference in feature construction and training loss.

## Training and calibration

Use all existing TRAIN scenarios, seeds 701–710, and the exact existing calibration corpus for each network. Per seed, retain every positive endpoint and a random sample of at most 60,000 negatives, using seed 600 + model_seed (the retained General sampling convention). Use the existing `training_weights` helper: half the mass to positives, equally across families and events; half to negatives, equally across sampled scenarios. The binary classifier learns all five families jointly. The two time allowances share the same sampled endpoints and weights.

Use the existing full seasonal/delayed LightGBM recipe without architecture or hyperparameter search: 400 trees, 31 leaves, learning rate 0.03, minimum child samples 20, L1 1, L2 20, feature fraction 0.85, deterministic column-wise training; model random seed 5100 + model_seed. Four CPU threads are an execution setting. No early stopping, fitting on calibration, or evaluation-guided revision.

For each fitted model, search every distinct score threshold on calibration under the existing clean-FPR ceiling of 0.005 (0.5%). Freeze two mandatory operating points: (1) maximum pooled F1, ties by worst-family F1 then lower clean FPR; (2) maximum worst-family F1, ties by pooled F1 then lower clean FPR. The first is the primary baseline; the second documents the family-balance trade-off. Thresholds and any unattained 0.80 family targets are reported. Neither operating point is selected using evaluation results. The threshold rules differ from the hybrid's recorded integration/protection search, which is retained unchanged.

## Freeze, evaluation and reporting

Record the protocol, code/dependency hashes, source manifests, final-result hash and feature-input hashes before fitting. Fit and calibrate all 40 classifiers before freezing all model and threshold hashes. Only then evaluate the existing six source datasets per network: Modena 40811–45811 in steps of 1000; L-Town 110811–115811 in steps of 1000. Do not open the locked test, generate new scenarios, or refit the hybrid. Reuse its stored 60 matched model/source results per network.

Report all five family F1 values, family macro F1, pooled F1, clean FPR, per-source/model summaries, ranges and paired differences. Include both time allowances and both calibration objectives, regardless of which method performs best. Distinguish the effect of the information allowance on the single classifier from the historical hybrid's 0-versus-3-hour comparison. Ten seeds share six sources and are not 60 independent datasets. If the single classifier matches or exceeds the hybrid, revise the thesis's complexity claim accordingly.

Implementation failures may be repaired and documented without consulting evaluation outcomes. A completed evaluation is not rerun for tuning. Generated feature matrices are reproducible caches; preserve fitted models, selections, predictions and audit records. No existing project artifact is overwritten.

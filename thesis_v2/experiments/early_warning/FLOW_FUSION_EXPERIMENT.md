# Received flow and internal General combination experiment

Frozen before fitting or scoring any new candidate. Four arms: the independently
confirmed 42-pressure-history design, internal probability calibration plus
reliability routing, received flow context, and both changes. The reference's
confirmed replay F1 is 0.730307; its deployment status is unchanged.

The flow representation adds 44 shared features to the original 158. A rank-16
normal received-flow factorization uses observation masks and robust latent
projection; a ridge-10 regression relates that flow context to received normal
pressure. Features include its pressure discrepancy, uncertainty and support,
16 flow-context coordinates, and change consistency at 1, 3, 6, 12 and 24 hours.
Numerical evidence is quantized to 0.1 standardized units. A missing flow context
is explicitly unavailable. There is no additional inference delay or detector.

The combination arm keeps precisely five internal components. Each component
gets a positive-slope sigmoid calibration fitted to population-weighted binary
labels on inner held-source predictions. A regularized linear correction of
the original internal softmax weights minimizes binary mixture log loss; it
receives only the component scores and original routing probabilities. No
family-specific inference rule or additional branch is introduced.

For every outer TRAIN source holdout, inner leave-one-source-out predictions
come from teachers trained on the other three sources. ALL normal references,
feature scales and detectors exclude BOTH held sources. Ten unordered source
exclusion pairs can share their models across the two legitimate outer roles.
Each teacher uses every positive and 20,000 sampled negatives per fitting source,
with inverse-probability weights. Seeds are 701 through 705. The outer fits use
the exact existing all-positive plus 60,000-negative samples. Full TRAIN fitting,
if authorized by the progression gate, uses component seeds 1301 through 1305.

All 25 outer fold/seed comparisons must complete before selection. Existing even
held-source TRAIN scenarios select thresholds on the unchanged seven-budget
grid; odd scenarios are the development diagnostic. These reused TRAIN sources
are not independent confirmation. Eligibility requires at least 20 replay wins,
mean replay F1 gain of at least 0.01 and positive mean gain in every source. Rank
eligible designs by mean worst-family F1, then pooled F1, then lower clean FPR.
Only one design proceeds. General-only other-family changes are diagnostic;
the complete system must satisfy the protection gates below.

The selected design receives pressure feature-path rounding checks at 0.01 and
0.05 m, five sources, seed 701, frozen models and thresholds. Mean replay gain
must remain positive and every source gain nonnegative. No reselection follows
failure. This does not claim end-to-end flow or pressure quantization robustness.

Full-system calibration, conditional on passing TRAIN and rounding, retains the
original specialized decisions and calibrates operating settings only. Other
family mean F1 losses cannot exceed 0.005, source losses 0.01. Pooled mean F1 and
mean clean FPR must not worsen; source pooled loss <=0.005 and source clean FPR
increase <=0.0001. The existing absolute FPR bound and protected router grid
remain. Confirmation readiness requires every mean family and every source-mean
family >=0.78 and positive source-mean replay improvements. 0.80 is the target;
0.78 to 0.80 is close, not attainment. Failure stops this frozen experiment.

Architecture remains General, Drift and Noise, with General's five internal tree
components and the existing external router/feedback. Drift/Noise delay is three
hours. Pressure and flow missingness remain 0.50. Generator distribution,
severity, scenario splits and locked test remain fixed. No consumed confirmation
cases may inform this design. A fresh confirmation is allowed only after a
qualified design and all operating settings are frozen; no deployment is implied.

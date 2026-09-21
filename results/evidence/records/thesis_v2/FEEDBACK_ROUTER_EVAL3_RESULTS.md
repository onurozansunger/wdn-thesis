# Evidence router + expert feedback: locked EVAL-3 results

Date: 2026-09-03.

## Claim supported by this run

The frozen hybrid detector with the calibration-selected router/feedback veto
achieved a point F1 above 0.80 for every attack family on a new, one-time
EVAL-3 corpus. The result uses a declared three-hour decision latency for the
drift/noise branches. It is an independent fresh-seed simulation confirmation,
not the original locked test result.

## Final architecture

The deployed decision retains three separately calibrated branches:

1. a zero-latency general residual-mixture branch for abrupt, replay and broad
   anomaly evidence;
2. a three-hour seasonal drift branch;
3. a three-hour seasonal noise branch.

The new graph-time router classifies observable evidence into clean, abrupt,
replay, drift or noise. Two row-level feedback heads then assess whether the
candidate drift/noise alarm is locally consistent with that expert's evidence.
The router and feedback may veto an already-positive seasonal specialist only
when replay is preferred by at least 0.10 and the relevant expert feedback is
below 0.10. The general branch is never vetoed, so every rejection falls back
to the general mixture. Neither labels nor event boundaries are inputs at
inference.

## Development sequence

The router and feedback heads were trained from generator-held TRAIN OOF
predictions: 1,298,091 rows and 9,548 graph-time groups from 62 scenarios. The
initial symmetric score re-ranking was rejected because it improved replay but
reduced genuine noise below the calibration gate. This negative result was
recorded before the replacement rule was assessed.

The one-sided veto was then selected on the existing 99-scenario calibration
corpus. It raised calibration replay F1 from 0.8215 to 0.9319 while retaining
drift/noise at 0.8348/0.8172 and reducing clean FPR from 0.004274 to 0.004253.
All thresholds and the veto rule were frozen before EVAL-3 generation.

## Locked EVAL-3

EVAL-3 contains 144 scenarios from six predeclared generator seeds. Pressure
and flow missing probabilities remained 0.50; attack parameters, distribution
and split logic were unchanged. Both the old frozen baseline and the new
candidate were scored on the same 3,015,169 rows in one pass.

| Family | Frozen baseline F1 | Router + feedback F1 | 95% cluster-bootstrap CI |
|---|---:|---:|---:|
| Random | 0.9417 | **0.9413** | 0.9194–0.9582 |
| Replay | 0.7891 | **0.9079** | 0.8857–0.9300 |
| Drift | 0.8405 | **0.8405** | 0.8099–0.8678 |
| Noise | 0.8106 | **0.8106** | 0.7777–0.8421 |
| Targeted | 0.9308 | **0.9310** | 0.9074–0.9514 |

Worst-family F1 increased from 0.7891 to 0.8106. Pooled row-level F1 increased
from 0.6416 to 0.6515, and clean FPR decreased from 0.004492 to 0.004462. The
standalone seasonal expert F1 values were 0.8549 for drift and 0.8097 for noise.

The mechanism removed 775 final alarms. Of these, 664 were replay-family false
alarms and 29 were true positives. Replay precision increased from 0.6906 to
0.9079 while recall changed from 0.9204 to 0.9079. Drift and noise family
decisions were unchanged. This matches the intended mechanism: suppress
seasonal specialists when their apparent evidence is better explained by
replayed values.

## Router audit on EVAL-3

| Router state | Graph-time accuracy |
|---|---:|
| Clean | 0.9951 |
| Abrupt | 0.8966 |
| Replay | 0.9259 |
| Drift | 0.7256 |
| Noise | 0.8430 |

The router is therefore used as a guarded veto signal, not as a hard family
selector. This is especially important for drift, where direct routing remains
weaker than the final detector.

## Interpretation limits

- All five family point estimates exceed 0.80, but noise has the narrowest
  margin and its 95% interval extends below 0.80.
- Drift/noise results use the declared three-hour decision delay; they must not
  be presented as zero-latency performance.
- EVAL-3 was opened once after calibration selection. The original locked test
  remains unopened.
- Three incorrectly generated preliminary directories with 30% missingness
  were quarantined before scoring and excluded. The evaluated manifest contains
  only the six valid 50%-missing corpora.
- The full automated test suite passes (76 tests), and the saved confusion
  counts were independently recalculated from `scores_eval3.npz`.

Primary artifacts: `runs/operational/feedback_router_v2`,
`runs/operational/feedback_router_veto_v1`, and
`runs/operational/feedback_router_eval3_v1`.

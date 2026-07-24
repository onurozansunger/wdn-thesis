# Results package

Figures and tables for the temporal Mixture-of-Experts attack detector, across water, power and traffic networks.

Every table is provided as `.csv` (machine readable) and `.md` (readable inline).

## Contents

- **`figures/01_part1_per_attack.png`** — Part 1 (water/Modena): per-attack F1, mean ± std over 10 seeds.
- **`tables/01_part1_per_attack.csv`** — Part 1 per-attack F1 with seed standard deviation.
- **`figures/02_crossdomain_f1.png`** — Overall detection F1 on three domains with the identical model.
- **`figures/03_replay_vs_signal_speed.png`** — Replay F1 rises with how fast the monitored signal moves — the replay ceiling is a property of the domain, not the model.
- **`figures/04_difficulty_heatmap.png`** — Per-attack F1 for every domain: the hard class inverts between water/power and traffic.
- **`tables/02_crossdomain.csv`** — Cross-domain summary: overall and per-attack F1 for the three networks.
- **`figures/05_replay_weight_pareto.png`** — Forcing replay up in the loss trades away overall F1 along a Pareto front (10 seeds per point).
- **`tables/03_replay_weight_sweep.csv`** — Replay-weight sweep: the overall-vs-replay trade-off.
- **`figures/06_router_diagnostic.png`** — Router confusion. Modena is healthy; Net3 sends 77% of stealthy windows to the replay expert.
- **`tables/04_router_accuracy.csv`** — Per-class routing accuracy on both networks.
- **`figures/07_routing_schemes.png`** — Routing schemes compared on identical checkpoints. The oracle bar is the ceiling any hard-selection scheme can reach.
- **`tables/05_routing_schemes.csv`** — Soft mixture vs hard top-1 vs cascade vs oracle expert.
- **`figures/08_cascade_attempts.png`** — How many experts the cascade has to try per window.
- **`figures/10_normalisation.png`** — Global vs per-node normalisation — testing whether water values sitting close together is what hides replay.
- **`tables/07_normalisation.csv`** — Normalisation comparison (overall and replay F1).

## Not yet generated

- expert-supervision sweep

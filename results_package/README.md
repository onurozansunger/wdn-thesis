# Results package

Figures and tables for the temporal Mixture-of-Experts attack detector on water, power and traffic sensor networks. Every table is provided as `.csv` (machine-readable) and `.md` (readable inline). Read top to bottom — it follows the story.

## Headline numbers

- **Water**: detection F1 0.833
- **Power**: detection F1 0.816
- **Traffic**: detection F1 0.891

- **Final ensembled model (water)**: F1 0.847 (single-seed mean 0.843)

## Figures

- **`figures/01_overview_domain_f1.png`** — The identical detector reaches F1 ~0.83-0.87 on water, power and traffic.
- **`tables/01_overview_domain_f1.csv`** — Overall F1 per domain.
- **`figures/02_water_normalisation.png`** — Normalising each sensor by its own scale lifts replay ranking from below chance (0.46) to AUROC 0.79; overall F1 holds.
- **`tables/02_water_normalisation.csv`** — Global vs per-node normalisation on water.
- **`figures/03_normalisation_generalisation.png`** — Per-node normalisation lifts replay most where it is hardest (water, power); traffic replay is already easy.
- **`tables/03_normalisation_generalisation.csv`** — Replay AUROC, global vs per-node, per domain.
- **`figures/04_difficulty_heatmap.png`** — Water/power find stealthy easy and replay hard; traffic is the mirror image.
- **`figures/05_routing_schemes.png`** — Cascade 0.772 matches the oracle ceiling (0.769), within a point of the soft blend.
- **`tables/05_routing_schemes.csv`** — Routing-scheme F1 (5 seeds).
- **`figures/06_router_diagnostic.png`** — Router is healthy on Modena (96%); on Net3 it sends 77% of stealthy windows to the replay expert.
- **`figures/07_replay_why.png`** — Why replay still underperforms: replayed and genuine scores overlap (AUROC 0.75), so no threshold separates them (best F1 0.41) — a physical limit.


## Conference-grade detail

- **`tables/08_full_results_pernode.csv`** — Per-domain overall and per-attack F1 (per-node, 5 seeds).
- **`tables/09_ablation.csv`** — Contribution of each new component (water).
- **`figures/09_ablation.png`** — Per-node normalisation and threshold calibration together turn replay from ~0 into a usable signal; the ensemble adds a little.
- **`tables/10_threshold_calibration.csv`** — Validation-calibrated threshold vs default 0.5, per domain.
- **`tables/11_model_config.csv`** — Model and training configuration.
- **`tables/12_signal_statistics.csv`** — Per-sensor temporal variation by domain — small variation (water) is why replay hides in the noise.
- **`figures/14_pr_curves_per_attack.png`** — Four attacks are near-perfectly separable; only replay's curve collapses — the concrete signature of the information limit.
- **`figures/15_seed_variance.png`** — Every configuration is tightly clustered across 5 seeds — the results are reproducible.


## Conference-grade detail

- **`tables/08_full_results_pernode.csv`** — Per-domain overall and per-attack F1 (per-node, 5 seeds).
- **`tables/09_ablation.csv`** — Contribution of each new component (water).
- **`figures/09_ablation.png`** — Per-node normalisation and threshold calibration together turn replay from ~0 into a usable signal; the ensemble adds a little.
- **`tables/10_threshold_calibration.csv`** — Validation-calibrated threshold vs default 0.5, per domain.
- **`tables/11_model_config.csv`** — Model and training configuration.
- **`tables/12_signal_statistics.csv`** — Per-sensor temporal variation by domain — small variation (water) is why replay hides in the noise.
- **`figures/13_learning_curves.png`** — The detector converges within ~30 epochs and the 5-seed band is tight — the result is reproducible.
- **`figures/14_pr_curves_per_attack.png`** — Four attacks are near-perfectly separable; only replay's curve collapses — the concrete signature of the information limit.
- **`figures/15_seed_variance.png`** — Every configuration is tightly clustered across 5 seeds — the results are reproducible.


## Conference-grade detail

- **`tables/08_full_results_pernode.csv`** — Per-domain overall and per-attack F1 (per-node, 5 seeds).
- **`tables/09_ablation.csv`** — Contribution of each new component (water).
- **`figures/09_ablation.png`** — Per-node normalisation and threshold calibration together turn replay from ~0 into a usable signal; the ensemble adds a little.
- **`tables/10_threshold_calibration.csv`** — Validation-calibrated threshold vs default 0.5, per domain.
- **`tables/11_model_config.csv`** — Model and training configuration.
- **`tables/12_signal_statistics.csv`** — Per-sensor temporal variation by domain — water/power signals barely move (why replay hides), traffic swings widely.
- **`figures/13_learning_curves.png`** — The detector converges within ~30 epochs and the 5-seed band is tight — the result is reproducible.
- **`figures/14_pr_curves_per_attack.png`** — Four attacks are near-perfectly separable; only replay's curve collapses — the concrete signature of the information limit.
- **`figures/15_seed_variance.png`** — Every configuration is tightly clustered across 5 seeds — the results are reproducible.

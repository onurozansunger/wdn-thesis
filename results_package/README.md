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

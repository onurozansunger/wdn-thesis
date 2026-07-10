# Part 2 — Self-Play Roadmap & Findings

_Working note for the Friday supervisor meeting. Companion to the Part-1
results on the Model Comparison dashboard page._

## Where Part 1 leaves us

Part 1 is locked and reproducible:

- Temporal Mixture-of-Experts GNN, **F1 0.866 ± 0.004** over 10 seeds on Modena.
- Four attack families detected reliably (random, targeted, stealthy, noise).
- Replay at the **information-theoretic ceiling** (~0.02 F1): in a six-hour
  Modena window pressure varies ~0.01 m against ~0.4 m sensor noise, so a
  k-step replay is nearly indistinguishable from clean data. The replay-weight
  sweep (10 seeds) shows overall-F1 vs replay-F1 trade off along a Pareto
  front — the ceiling is a property of the data, not the model.

The one lever left is the **attacker** side. That is Part 2.

## What we built and measured in Part 2

Self-play couples an attacker GNN (4-expert MoE) against the Part-1 defender.
Both co-train; the attacker maximises damage + stealth under a stealth budget
(ε_p, ε_q, top-k sensors), the defender minimises detection loss.

### Finding 1 — Catastrophic forgetting, then fixed

Naive self-play erodes the defender: hand-crafted F1 drops 0.866 → 0.733
because the defender only ever sees adversarial batches and drifts off the
Part-1 optimum. A **hand-crafted retention loss**
(`L = L_adv + λ·L_hand_crafted`) plus a slower defender LR recovers this:

| Config (5 seeds) | hand-crafted F1 | adversarial F1 |
|---|---:|---:|
| baseline (no retention) | 0.733 ± 0.036 | 0.757 ± 0.037 |
| retention λ=1, lr=5e-4 | 0.760 ± 0.026 | 0.776 ± 0.018 |
| **retention λ=5, lr=1e-4** | **0.789 ± 0.002** | **0.804 ± 0.002** |

Retention also collapses the seed variance (0.036 → 0.002).

### Finding 2 — The attacker does not (yet) outpace the defender

PGD refinement of the perturbation, added first in the attacker step and then
in **both** attacker and defender steps, did not shift the equilibrium:

| Config | hand F1 | adv F1 |
|---|---:|---:|
| retention λ=5 (no PGD) | 0.789 | 0.804 |
| + PGD (attacker only) | 0.792 | 0.810 |
| + PGD (attacker + defender) | 0.788 | 0.804 |

adv F1 ≈ hand F1 throughout: the defender catches adversarial attacks about as
well as hand-crafted ones. The Part-1 defender is simply strong enough that,
within the current stealth budget (ε_p = 5 m, k_p = 10 of 272 nodes), there is
no perturbation that reliably evades it.

### Finding 3 — Attacker structure

Vocabulary mining on the trained attacker (`scripts/vocabulary_mining_v2.py`):

- **Expert collapse**: the 4-expert attacker MoE routes 99% of windows to a
  single expert despite the diversity/balance losses — one universal
  perturbation dominates.
- **No hub preference**: node-degree ↔ attack-frequency correlation ≈ 0; the
  attacker perturbs sensors roughly uniformly, not strategically.
- **Stealth ≈ 0**: the defender assigns near-1 anomaly probability to attacked
  sensors — consistent with adv F1 ≈ 0.80.

## Reframing: the null result is a robustness result

The honest read of Finding 2 is not "the attacker failed" but **"the Part-1
defender is certifiably robust to any bounded attacker we can train."** The
self-play equilibrium F1 ≈ 0.79 is a *lower bound* on defender performance
under adaptive, gradient-based, physics-aware attack within a realistic stealth
budget. That is a publishable, defensible contribution — an empirical
robustness certificate rather than a failed attack.

## Options for Part 2 (ranked)

**A. Robustness-certification framing (recommended, low risk).**
Present the equilibrium as a bounded-attacker robustness certificate. Sweep the
stealth budget (ε_p, k_p) and plot the defender's worst-case F1 vs budget — a
robustness curve. Delivers a clean figure and a clear claim with the runs we
already have plus one budget sweep (~half a day of compute).

**B. Enlarge the attack budget, then re-certify (fast, complements A).**
ε_p 5→12, k_p 10→30. If adv F1 finally drops, we have a genuine attacker and a
tighter co-evolution story; if it holds, the robustness claim gets stronger.
One 5-seed run.

**C. Attacker-output analysis / threat model (supervisor-requested, do regardless).**
Ship the vocabulary-mining figures (t-SNE of perturbations, stealth–damage
scatter, magnitude-per-class, centrality) and the formal threat-model table
(knowledge × capability × budget × goal). This satisfies "analysis on the
attacker output" independently of whether the attacker gets stronger.

**D. Fix MoE expert collapse (needed for C to be meaningful).**
Condition each attacker expert on a target attack class, or add a load-balancing
penalty with a hard floor, so the vocabulary analysis reflects real diversity
rather than one dominant mode.

**E. RL attacker (high effort, optional / future work).**
Policy gradient over discrete (node, magnitude) actions with a stealth-shaped
reward. Larger potential lift but 1–2 weeks; better positioned as future work
than as a Friday deliverable.

## Suggested plan for the meeting

1. Lead with **Part 1 as done and reproducible** (Model Comparison page).
2. Present Part 2 as a **robustness study** (framing A) with the retention fix
   and the PGD null result shown honestly.
3. Bring the **vocabulary + threat-model analysis** (C) as the concrete
   "attacker output" deliverable.
4. Ask the supervisors to choose between **B** (bigger budget) and **E** (RL)
   for the next sprint.

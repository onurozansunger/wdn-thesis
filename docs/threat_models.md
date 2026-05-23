# Threat Models

A formal description of the adversaries this system is designed to
detect. Defining these explicitly is one of the supervisor-feedback
items for Part 1 and frames the scope of every result we report.

## Operating environment

- **System under defence.** A water distribution network (WDN)
  instrumented with pressure sensors at every junction and flow
  sensors at every pipe. Readings flow into a SCADA aggregator over an
  IP network on a fixed (e.g. 1-hour) interval.
- **Defender.** A graph neural network co-located with the SCADA
  aggregator. Sees the raw sensor stream. May drop readings as
  observed/unobserved (the missing-data mask) but cannot influence the
  physical network.
- **Trust assumption.** The defender trusts the network topology
  (the `.inp` file), the hydraulic solver used at training time, and
  the historical clean data used to fit normalisers.

## Adversary capabilities

We assume a *partial-control insider* adversary that has gained
read/write access to a fraction of the SCADA telemetry stream:

| Capability | Granted | Justification |
|---|---|---|
| Read current sensor stream | ✅ | Insider has SCADA access. |
| Modify a subset of readings before they reach the defender | ✅ | The attack surface. |
| Modify network topology / pump schedule | ❌ | Out of scope; physical-actuation attacks need a different model. |
| Read clean training data | ❌ | Limits a fully white-box attack. |
| Inspect defender weights | partially | Realistic for an insider that exfiltrates the model; the self-play loop is the worst-case proxy. |
| Sustain access across timesteps | ✅ | Each attack is a *campaign*, not a one-shot event. |

### Attack budget

The adversary cannot move every sensor without being trivially caught
(a network-wide jump violates mass conservation). Two budget knobs:

- `k` — maximum number of sensors compromised in any single snapshot
  (`k ≲ 0.15 · N`, matching the 15% attack-fraction in the corruption
  pipeline).
- `ε` — maximum magnitude of any individual perturbation in metres of
  pressure (default `ε = 3 m`).

Both budgets are enforced explicitly in `apply_stealth_budget` for the
self-play attacker.

## Attack catalogue

| Class | Goal | Footprint | Detection difficulty |
|---|---|---|---|
| **Random falsification** | Coarse disruption | Scaled + biased reading per sensor | Easy — large residual |
| **Replay** | Hide a real change behind a stale reading | `pressure_obs[t] = pressure_true[t − k]`, `k ∈ {3,4,5,6}` | **Information-ceiling** — see `replay_ceiling.md` |
| **Stealthy bias** | Slow operator drift | Linear ramp over the window | Medium — small per-step delta |
| **Noise injection** | Sensor-failure mimicry | High-variance Gaussian on top of reading | Easy — variance spike |
| **Targeted** | Maximise hydraulic damage per touched sensor | Random falsification biased toward high-impact sensors | Easy — large residual |

The five hand-crafted classes are deliberately simple. The self-play
attacker (`AttackerGNN`) generalises them — it learns sparse,
budget-respecting perturbations and an Attacker-MoE auto-discovers
attack families without supervision.

## What the defender is NOT designed for

Stated explicitly so reviewers don't read more into the results than
is there:

- **Physical-layer attacks** that change demand patterns, open valves
  or trigger pumps. The defender only inspects telemetry; a hydraulic
  attack that the SCADA stream faithfully reports is not an "attack"
  by our definition.
- **Adaptive white-box adversaries** with full gradient access to the
  defender at attack time. The self-play loop is the closest proxy
  we offer; a stronger adversarial setting (PGD with model gradients
  at attack time) is future work.
- **Denial of service** at the network layer. We assume the SCADA
  pipeline keeps delivering packets — the attack is in the *content*
  of those packets.

## Why this matters

Defining the threat model:

1. **Bounds the claims.** "GNN beats WLS by 38× on reconstruction" is
   only meaningful inside the budget above.
2. **Justifies the dataset.** k-step replay (k ∈ {3..6}) replaces the
   degenerate 1-step replay — a real attacker replays recorded
   sessions, not a single sample.
3. **Maps to other domains.** The same partial-control insider model
   with a stealth budget shows up in network IDS, smart-grid
   monitoring, and payment-fraud detection — see
   `cross_domain_framing.md`.

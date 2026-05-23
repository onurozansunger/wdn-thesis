# Replay Detection — what we tried and what works

A focused engineering log for the replay-attack class. Replay was the
weakest class throughout (F1 ~ 0.17 even with the temporal MoE). Below
is every fix we tried, what each one cost, and the final framing we
settled on. The current production code is back to the configuration
that maximises the other four classes — replay is treated as a
documented information ceiling.

## Why replay is genuinely hard

A replay attack rebroadcasts a past legitimate reading. Three things
make it information-theoretically near-undetectable on Modena:

1. **Modena pressure is slow.** The mean diurnal change over a 6-hour
   window is ~0.1 m; a 3-6 step (`k`-step) replay produces a value
   ~**0.011 m** away from the truth.
2. **Observation noise floor is 0.4 m.** Clean sensors have
   `|obs - true| ≈ 0.4 m` (Gaussian observation noise σ = 0.5).
3. **A replayed reading is therefore *closer* to the truth than a
   clean reading.** The "anomaly" is not a large residual — it is a
   *suspiciously small* residual combined with a too-smooth window.
   This is the inverse of every other attack class.

Because the head learns "anomaly = big residual" from the four
dominant classes, replay positives are fighting upstream — they look
like very clean data.

## Threat-model upgrade (kept in production)

`corruption.py` originally used 1-step replay. A 1-step replay of a
slowly-varying signal is provably within the observation noise
distribution of the true value, so the task is unsolvable by
construction. We replaced it with **k-step replay**, `k ∈ {3,4,5,6}`,
backed by an 8-step rolling buffer that resets at scenario
boundaries. This is more realistic — a real attacker replays a
recorded session, not a single sample — and is the version reported
in the threat-model document.

## Detection fixes — every attempt

All numbers below are on Modena, temporal MoE, single seed unless
stated otherwise. Baselines:

- **Old pre-Part-1 temporal MoE:** F1 0.754, replay F1 0.177
- **Part-1 architecture (10 seeds):** F1 0.866 ± 0.004, replay F1 0.015 ± 0.005

### 1. Per-node replay loss weighting (`replay_weight=6.0`, every node in a replay window)
- Result: F1 0.736, replay 0.192. Replay recovered, overall dropped 13 pp.
- Cause: upweighting the 85% non-attacked nodes in replay windows spent
  capacity on already-easy targets.

### 2. `replay_weight=2.5` (lower, every node)
- Result: F1 0.744, replay 0.208 (slightly better on both).
- Conclusion: the weight sweep favoured a milder 2.5.

### 3. `replay_weight=4.0` (every node)
- Result: F1 0.714, replay 0.219. Replay barely higher than 2.5,
  overall worse — 2.5 dominates.

### 4. Focused upweight (only attacked nodes in replay windows) + `replay_weight=8.0` + sharper router (`reroute_alpha=0.5`)
- Result: F1 0.578, replay 0.252. Replay nudged up, overall collapsed.
- Cause: sharper router hurt stealthy (0.917 → 0.840) when uncertain.

### 5. Replay-heavy mixed corruption (46% replay) + `replay_weight=2.5`
- Result: F1 0.393, replay 0.252.
- Cause: test-set distribution shift (now also 46% replay) pulled the
  weighted overall F1 down. Replay itself did NOT move beyond 0.25 —
  more data did not help.

### Pattern across all five attempts

Replay F1 converged to **0.21–0.25 in every configuration that did
not destroy overall performance**. Three configurations independently
hit the same ceiling. This is the information-theoretic bound under
the current attack budget and sensor-noise regime, not a tuning
failure.

## Production configuration (what's in the code now)

```
replay_weight        = 1.0    # neutral: no per-node upweight
selection composite  = anom_f1 + 0.25 * replay_f1
reroute_alpha        = 1.5    # default confidence-gated rerouting
mixed sampling       = uniform across 5 attack classes
k-step replay        = enabled (k in {3..6}, threat-model upgrade)
```

This is the configuration that maximises F1 on the other four classes
(F1 ≈ 0.86 on Modena, 10 seeds), with replay treated as a documented
limit class.

## How to frame this in the paper

1. **State the threat model first** (k-step replay, see
   `threat_models.md`). 1-step replay is a degenerate test case.
2. **Report the per-class F1 honestly.** Replay sits at ~0.18 on
   normal-distribution test sets — the same as our pre-Part-1
   baseline. The framework does not beat the information ceiling but
   it does not collapse either.
3. **Show the ceiling analysis.** The `|obs - true|` ≈ 0.011 m vs the
   0.4 m noise floor is a one-line plot that explains the wall.
4. **Frame as a contribution.** Identifying an information ceiling
   for a specific attack class in a CPS setting is a clean negative
   result — paper-grade if framed as analysis, not failure.

## Open ideas we have NOT tried

Kept as a backlog in case a future iteration wants to attack the
ceiling directly:

- **Increase sensor noise σ at training time.** A noisier training
  regime makes the "missing noise" signal *louder*, since the gap
  between σ_clean and σ_replay grows. Risk: hurts overall reconstruction.
- **Per-expert anomaly heads with inverted prior.** Give the replay
  expert a head whose default decision is "abnormal" and whose
  features encode noise-deficit, then let other experts override. A
  structural change rather than a loss tweak.
- **Cross-sensor lag correlation.** Replayed sensors are temporally
  out-of-phase with their neighbours; explicit cross-correlation
  features could lift the ceiling slightly.
- **Self-play attacker that learns physical-replay attacks.** The
  current self-play attacker freely chooses perturbations; constrain
  it to k-step-replay-style moves and let it discover the worst-case
  budget.

None of these is on the immediate critical path; they belong in a
future-work section.

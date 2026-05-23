# Conference shortlist

Candidate venues for the thesis paper, ranked by realistic fit given
the current empirical state.

## Tier A — strongest fit, accept-realistic

### IEEE Transactions on Smart Grid (TSG) — special / workshop track
- **Why it fits.** The methodology (graph state estimation, FDI
  detection, physics-informed ML) maps directly to smart-grid
  literature; the WDN framing is a sibling.
- **Bar.** Empirical: real data is preferred but synthetic with
  rigorous baselines is accepted in workshops. Need 2+ classical
  baselines (WLS ✓, Kalman/EKF would help).
- **Effort to submit.** Low — repurpose the current draft, swap one
  diagram.

### ACM e-Energy (smart-water / smart-infrastructure track)
- **Why it fits.** The conference explicitly covers
  cyber-physical-water systems. Recent papers have used WNTR.
- **Bar.** Moderate. Strong methodology + 3 networks meets it.
- **Effort.** Medium — re-frame around energy/utility infrastructure.

### IEEE PES General Meeting / ISGT — cyber-physical sessions
- **Why it fits.** Power-systems venue but explicitly covers
  state-estimation under FDI; our threat model + MoE story translates.
- **Bar.** Engineering-paper friendly.
- **Effort.** Medium — needs a smart-grid framing layer in the intro.

## Tier B — workshop-tier ML venues, easier accept

### NeurIPS workshops (ML4CPS, ML for Engineering, AdvML)
- **Why it fits.** Self-play + MoE + held-out generalisation is the
  ML half of the story; CPS-flavoured workshops welcome novel
  applications.
- **Bar.** Strong methodology, modest empirical OK. Our 10-seed
  reproducibility + Modena/Net1/Net3 spread fits.
- **Effort.** Low — 4-page workshop format is close to what we have.
- **Deadline.** Typically October.

### AAAI workshops (Safe and Robust AI, AI for Critical Infrastructure)
- Similar to NeurIPS workshop bar. Multiple relevant tracks each year.

### ICASSP — graph signal processing / sensor-network sessions
- Less obvious fit but the spatial-temporal GNN with mass-conservation
  penalty maps to graph signal processing language.

## Tier C — main-conference shots, ambitious

### ICML / NeurIPS main track
- **Why it does NOT fit yet.** Empirical wins are modest (overall F1
  +5.8%); replay hit an information ceiling (~0.25 on Modena); no
  real-data validation. Reviewers reject on "limited datasets".
- **What it would take.** PGD-style stronger attacker, BATADAL
  real-attack data, formal Stackelberg convergence analysis.

### ESORICS / NDSS / S&P (security venues)
- **Why it does NOT fit yet.** Security venues want real-world
  attacker validation. Synthetic-only would not pass.
- **What it would take.** A real CPS testbed (e.g., SWaT) or a CTF
  partnership.

## Domain venues (smart-water specifically)

### Water Research / Journal of Water Resources Planning and Management
- **Why it fits.** Long-form journal. Audience values WNTR-based
  studies; comparable papers exist.
- **Bar.** Methodology clarity + hydraulic plausibility. We tick both.
- **Effort.** Highest — journal-length writing, slow turnaround.

### CCWI (Computing and Control for the Water Industry)
- Small but well-targeted. Engineering-paper friendly. Quick turnaround.

## Recommended sequence

1. **First submission, June–July.** NeurIPS / AAAI workshop. Cheapest
   accept, fastest feedback, gets the framework into the literature.
2. **Second submission, autumn.** IEEE TSG or e-Energy. Use workshop
   reviewer feedback to harden the empirical section.
3. **Third submission, winter.** CCWI or Water Research journal as
   the long-form record.

A main-tier ML conference (ICML/NeurIPS) is realistic *after* a real-
data benchmark is added — currently a separate work item, not a 2026
goal.

## What still blocks a strong submission

- **No real-attack benchmark.** Synthetic-only invites "limited
  evaluation" reviews.
- **Replay ceiling not yet framed as a contribution.** The
  information-theoretic analysis exists in our notes but is not in the
  paper draft — it would be a clean section on its own.
- **No formal Stackelberg analysis.** Current self-play is empirical.
  Adding even a one-shot equilibrium analysis would lift the
  theoretical bar significantly.

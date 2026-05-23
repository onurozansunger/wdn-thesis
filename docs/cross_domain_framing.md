# Cross-Domain Framing

The supervisors asked whether the framework generalises beyond water
distribution networks — specifically to network-traffic intrusion
detection, smart-grid monitoring, and payment-fraud detection. The
short answer is **yes, with one substitution per setting**. Every
domain shares the same ingredients:

1. A **graph** whose nodes carry time-varying observations.
2. A **partial-control adversary** that tampers with a budget-limited
   subset of those observations.
3. A **multi-task** detection requirement: reconstruct the true state
   AND flag which nodes were tampered.

This document maps the WDN abstractions to each target domain so the
self-play / MoE / pattern-feature recipe transfers.

## 1. Network-traffic intrusion detection (NIDS)

| WDN abstraction | NIDS equivalent |
|---|---|
| Junction node | Host / interface |
| Pipe edge | Active TCP / UDP session |
| Pressure / flow | Packet-rate, byte-rate, flow-duration per host |
| Hand-crafted attacks | Port scan, SYN flood, exfiltration, beaconing, slow drift |
| Mass conservation | Byte-rate conservation across an aggregation switch |
| Replay attack | Recorded benign flow re-played to mask exfil |

The k-step replay analysis transfers directly: a flow replayed from
last week is hard to distinguish from a fresh benign flow when both
match the user's profile. Self-play attackers correspond to adaptive
intrusion frameworks (Caldera, Atomic Red Team) that probe the IDS.

**What needs to change:** the encoder. Flow statistics are not
spatially smooth in the same way as pressure — message-passing on the
host-interaction graph behaves differently. GraphSAGE still works as
a starting point because the aggregation is mean-style.

## 2. Smart-grid monitoring

| WDN abstraction | Smart-grid equivalent |
|---|---|
| Junction node | Substation bus |
| Pipe edge | Transmission line |
| Pressure | Bus voltage magnitude |
| Flow | Line active-power flow |
| Mass conservation (B·q = 0) | Power-flow balance at each bus (Kirchhoff) |
| Hand-crafted attacks | False-data injection (FDI), measurement replay, topology forgery |
| Stealth budget | Bad-data-detection threshold of the EMS |

This is the *closest* analogue — smart-grid state estimation
literature has been the implicit inspiration for the physics-loss
term. The "stealthy FDI" class in the FDI literature is *exactly* our
stealthy-drift attack. The MoE-with-pattern-features recipe carries
over with no substitutions.

**What needs to change:** swap the hydraulic incidence matrix for the
electrical admittance matrix (the `Y` matrix). Replay analysis
applies — synchrophasor (PMU) data is also smooth and a replay of an
old PMU sample is information-ceiling-bounded.

## 3. Payment-fraud detection

| WDN abstraction | Payment-fraud equivalent |
|---|---|
| Junction node | Account / merchant |
| Pipe edge | Transaction (directed) |
| Pressure | Account balance / spending rate |
| Flow | Per-transaction amount |
| Hand-crafted attacks | Card-not-present fraud, account takeover, money-mule chain, structuring |
| Mass conservation | Money in − money out − fees = balance change |
| Replay | Recorded legitimate transaction re-submitted (skimmed card) |
| Stealth budget | Daily-cap / unusual-pattern thresholds |

Payment graphs are the largest of the three — directed, dynamic, with
edges (transactions) being the primary objects rather than nodes.
Self-play here corresponds to red-team fraud rings: an attacker GNN
generates plausible transaction sequences against the defender. The
mass-conservation analogue is debit/credit equality — false-data
injection that violates it gets caught trivially, which is exactly
why real fraudsters use mass-conserving schemes.

**What needs to change:** edge-centric representation. Use an
edge-based message-passing variant (or a line graph) rather than
node-centric GraphSAGE.

## What stays the same across all three

- **Stackelberg self-play framing** — the attacker is a leader, the
  defender is a follower, budgets bound both.
- **Mixture-of-Experts** with a small classifier + bigger experts +
  direct expert supervision and confidence-gated rerouting.
- **Information-ceiling diagnostics** — measure
  `|obs − true| / noise_floor` to know whether the task is
  detectable at all before grinding more compute.
- **Held-out novel-attack test** — synthesise attacks that match no
  hand-crafted class; the defender's score on them is the only
  honest generalisation number.

## What this buys the thesis

A single framework that's clearly portable to three well-funded
adjacent problems. The empirical results stay scoped to WDN, but the
discussion section can argue that the methodology — not the specific
network — is the contribution.

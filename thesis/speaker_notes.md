# Speaker Notes — Progress Presentation
_Companion to `thesis/slides.pdf` (20 slides). English lines are what you
say out loud; **[TR]** notes are delivery guidance for you only._

---

## Delivery plan

- **Target 18–20 minutes** + questions. ~1 min/slide, but spend **2 min on
  slide 7 (cascade architecture)** and **2 min on slide 13 (normalisation)**
  — the two things that are new since last time and the two they asked for.
- **[TR]** Bu sunumun iki yeni ana çıktısı: (1) cascade mimarisi — hocaların
  istediği, (2) per-node normalizasyon — replay'i kurtaran fikir, yine
  onların önerisi. Bu ikisinde yavaşla, gerisinde akıcı geç.
- **The one sentence they must remember:** *"We built the interpretable
  cascade you asked for, and your normalisation idea recovered replay from
  below-chance to AUROC 0.79."*
- **[TR]** İki şeyi de "sizin fikriniz / isteğiniz" diye çerçevele — sahiplik
  hissi verir, toplantıyı işbirlikçi kılar.

---

## Slide 1 — Title
> "Thanks. Two headlines today: the new cascade routing architecture, and a
> normalisation fix that recovers the replay attack. Both came directly out
> of our last discussion."

**[TR]** 15 saniye, geç.

## Slide 2 — The problem
> "Quick recap of the setting: a water network is a graph, junctions report
> pressure, pipes report flow. Sensing is sparse — half the readings missing
> — and noisy. An attacker falsifies a subset of sensors. Five attack
> families, from black-box noise to white-box stealthy drift. The detector
> reconstructs the true values and flags which sensors are lying."

**[TR]** Tabloyu göster, "Stealthy" ve "Replay"i vurgula.

## Slide 3 — Pipeline
> "The pipeline: network, simulate, inject attacks, detect, evaluate. Only
> the first stage is domain-specific — the rest is shared code, which
> matters for Part 2."

## Slide 4 — Architecture (base)
> "The detector is a temporal Mixture-of-Experts GNN. Six experts, each a
> GraphSAGE-plus-GRU model specialised on one attack class, with a router
> that decides which experts to trust. Physics loss enforces mass
> conservation. 377K parameters."

**[TR]** Bu eski mimari — hızlı geç, asıl yeni olan bir sonraki slayt.

## Slide 5 — How it was built up
> "Each stage removes a specific weakness: spatial GNN can't see replay,
> so add temporal GRU; one model can't cover all attacks, so add experts;
> and pattern features expose the noise a replayed value lacks."

**[TR]** ~40 saniye.

## Slide 6 — Four refinements
> "Four refinements to the routing: confidence-gated rerouting, a smaller
> router with bigger experts, direct expert supervision so no expert
> starves, and per-expert reconstruction."

**[TR]** Bunlar geçen sefer konuştuğumuz maddeler — hızlı özet, "bunlar tamam"
tonu.

## Slide 7 — NEW ARCHITECTURE: cascade routing ⭐ (2 min)
**[TR]** Burada yavaşla. Hocaların istediği mimari bu. Diyagramı soldan sağa
elinle takip et.

> "This is the new architecture you asked for. Instead of blending all six
> experts opaquely, the router now *ranks* them and we run one at a time.
>
> Graph snapshots go to the router — a very fast, simple classifier — which
> ranks the experts. We run the top-ranked expert. Then a feedback check
> asks: did it do a good job? If the feedback is negative, we go back to the
> router and try the next most likely expert — that's the orange loop — and
> we repeat until an expert is accepted.
>
> The key advantage over the blend: every prediction now carries which
> expert ran and why it was kept or rejected. It's interpretable."

**[TR]** "interpretable / audit trail" kelimelerini vurgula — hocalar tam da
bunu istemişti.

## Slide 8 — Cascade: why this design
> "The feedback is the interesting part — it's label-free, because at
> inference we have no ground truth. Two signals: does the expert's own
> 'clean' set of sensors agree with their readings, and does its
> reconstruction obey mass conservation? A wrong expert mislabels which
> readings are trustworthy and pays for it. We standardise the two terms so
> neither dominates, and we trust a confident router unless the feedback
> clearly disagrees."

**[TR]** Formülü işaret et ama okuma; "consistency + physics, etiketsiz" de.

## Slide 9 — Cascade vs the soft mixture
> "Honest comparison over 5 seeds. Hard selection — top-1, cascade, oracle —
> all cluster near 0.77, within a point of the soft blend at 0.79. The
> experts are similar on Modena, so the interpretable cascade costs almost
> no accuracy. The label-free feedback matches the oracle on most seeds;
> making it reliable on *every* seed is the honest open problem, and one of
> the next steps."

**[TR]** Dürüst ol. "costs almost no accuracy" + "open problem" — olgunluk
gösterir. Cascade barındaki error bar'ı (seed2) sorarlarsa: "one seed the
feedback mis-fires, that's the reliability problem."

## Slide 10 — Part 1 results (per-attack)
> "Detection over 5 seeds with a calibrated threshold. Four of five attacks
> are caught reliably — random and targeted near-perfect, stealthy and noise
> above 0.9. Replay sits near zero. The next slides explain why, and how we
> recovered most of it."

## Slide 11 — Router diagnostic
> "We asked if the router was mis-routing. On Modena it's healthy, 96%. On
> the larger Net3 topology, 77% of stealthy windows get routed to the replay
> expert — they both look smooth, and 97 nodes aren't enough to separate
> them. The cascade's feedback is designed to catch exactly this."

## Slide 12 — The replay ceiling
> "Replay fails for a physical reason: Modena pressure moves about a
> centimetre over six hours, sensor noise is forty centimetres, so a
> replayed value hides in the noise. Forcing it up in the loss only trades
> away overall F1. The loss reaches its ceiling — but the score *ranking*
> still holds information, which the next slide exploits."

**[TR]** Son cümle normalizasyona köprü.

## Slide 13 — NEW: normalisation recovers replay ⭐ (2 min)
**[TR]** İkinci ana çıktı, hocaların "normalize dene" önerisi. Yavaşla.

> "Your hypothesis was that water values sit too close together to tell a
> replay apart. We tested it. Instead of one global scale, we normalise each
> sensor by *its own* temporal scale — per-node — which amplifies the tiny
> per-node variation a replay lacks.
>
> Middle panel is the result: replay ranking, measured by AUROC, jumps from
> 0.459 — below chance, the red line — to 0.794, well above. Overall F1
> holds at 0.833. And a threshold calibrated on validation turns that
> recovered ranking into actual detections. So the normalisation idea
> worked — replay is no longer invisible."

**[TR]** "below chance to 0.79" cümlesini net söyle ve dur. Bu slaytın vurucu
anı.

## Slide 14 — Part 2 divider
> "Second question: does any of this survive outside water?"

## Slide 15 — One detector, three physics
> "We took the same detector — unchanged — to a power grid, IEEE 118-bus, and
> a road-traffic network. Voltage instead of pressure, speed instead of
> pressure. Only a per-domain data adapter changed; model, training and
> corruption pipeline are the same code."

## Slide 16 — Cross-domain results
> "It holds: F1 0.845 to 0.870 across three unrelated physical systems, no
> domain-specific tuning."

## Slide 17 — The key insight ⭐
**[TR]** Cross-domain'in vurucu slaytı. Yavaşla.
> "The most interesting result. Replay was impossible in water — 0.02. In
> the grid the same model gets 0.27. In traffic, 0.87. Same model, same
> attack. Replay is detectable only when the signal moves between readings —
> a property of the domain, not the model. It turns a one-domain negative
> into a general principle."

## Slide 18 — Difficulty inversion
> "And the hard case flips: water and power find stealthy easy, replay
> impossible; traffic is the mirror image."

## Slide 19 — Summary
> "So: Part 1 — the temporal MoE with the new interpretable cascade routing;
> the router failure diagnosed; and per-node normalisation recovering replay
> ranking from below-chance to 0.79. Part 2 — the identical model transfers
> to power and traffic, and the replay behaviour gives a general principle.
> Next steps: make the feedback reliable on every seed, extend the threat
> model, and consolidate toward a venue. I'd value your input on priority."

## Slide 20 — Thank you
> "Thank you — happy to take questions."

---

# Anticipated questions & answers

**[TR]** Bilmediğin şeyi uydurma; "we haven't tested that yet, fair point" de.

## On the cascade (the new architecture)

**Q: Why does the cascade not beat the soft mixture?**
> "Because on Modena the experts are similar — the oracle, the best possible
> single expert, is 0.77, basically the same as top-1. So hard selection
> can't beat the blend here; there's no accuracy left on the table. The
> cascade's value is interpretability at near-zero cost, plus the ability to
> reject a bad expert. On a network where experts differ more, the ranking
> would matter more."

**Q: The cascade has high variance — one seed is much lower.**
> "Yes, and that's the honest limitation. On that seed the label-free
> feedback accepts a wrong expert — the consistency signal can be fooled by
> an expert that under-flags. Four of five seeds are fine. Making the
> feedback reliable everywhere is the concrete next problem."

**Q: What exactly is the feedback signal?**
> "Two label-free terms. One: the expert declares which sensors are clean;
> on those, its reconstruction should match the reading — a wrong expert
> leaves falsified values in its clean set and pays for it. Two: the
> reconstructed flow should conserve mass. We z-score the two across experts
> so neither dominates."

**Q: Is the router the same as before?**
> "Yes — a small, fast spatio-temporal classifier. In the cascade it now
> produces a ranking rather than a soft weight, but it's the same network."

## On normalisation (the replay fix)

**Q: Does per-node normalisation add information, or just help the model use it?**
> "It doesn't add information — replay is still near the information limit.
> What it does is rescale each sensor by its own tiny temporal variance, so
> the small signal that *is* there isn't swamped by the global scale. The
> AUROC going from 0.46 to 0.79 shows the ranking information was there; the
> model just couldn't exploit it under global normalisation."

**Q: Why did overall F1 barely change if replay improved so much?**
> "Replay is one of five classes and its F1 is still low in absolute terms
> (0.14 at the calibrated threshold), so its weight on the overall number is
> small. The honest headline is the *ranking* — AUROC 0.79 — not the F1."

**Q: Why is replay F1 only 0.14 if AUROC is 0.79?**
> "AUROC measures ranking; F1 needs a threshold. Replay scores are better
> ranked but still overlap the clean distribution, so no single threshold
> gives high precision and recall together. The ranking is the real gain;
> turning it into high F1 would need a replay-specific threshold, which the
> cascade's per-class routing could eventually provide."

**Q: Did you calibrate the threshold on test? Isn't that leakage?**
> "No — calibrated on validation, applied to test. We also report the
> uncalibrated 0.5 numbers so the gain is auditable. The fix was that 0.5 is
> mis-calibrated under per-node normalisation, which shifts the score
> distribution."

## On the numbers / protocol

**Q: Last time you said 0.866, now it's 0.833. What changed?**
> "Different, cleaner protocol: these 5-seed runs all share one split and one
> validation-calibrated threshold, so global and per-node are strictly
> comparable. The 0.866 was an earlier best-case number under a different
> selection; I'm now reporting a single controlled protocol so the
> comparison is honest."

**Q: Only 5 seeds?**
> "Five gives a standard deviation around 0.005, so the comparison is tight.
> Happy to extend to ten."

## On cross-domain (if it comes up)

**Q: Is the traffic data real?**
> "Synthetic — a road-sensor graph with rush-hour dynamics. The power grid is
> a standard IEEE-118 benchmark. A real traffic benchmark like METR-LA is the
> natural next step; only the adapter changes."

## On the thesis

**Q: What's the contribution / where would you publish?**
> "An interpretable cascade detector for physics-constrained sensor graphs; a
> diagnosis of when routing and replay fail; a normalisation fix; and the
> cross-domain principle that replay detectability is set by signal dynamics.
> The framing fits CPS-security venues — I'd like to pick one with you."

---

## If you get a stuck question
1. *"That's a fair point — we haven't tested that yet; good candidate for the
   next run."*
2. *"I'd want to check the numbers before answering properly — can I come back
   to you?"*
3. *"I'm not sure — what would you expect to see?"* — **[TR]** hocaları
   konuşturur.

**Never invent a number.** Unsure between 0.79 and 0.80 to say "about 0.79".

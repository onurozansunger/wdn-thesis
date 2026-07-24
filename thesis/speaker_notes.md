# Speaker Notes — Progress Presentation
_Companion to `thesis/slides.pdf` (18 slides). English lines are what you
say out loud; **[TR]** notes are delivery guidance for you only._

---

## Delivery plan
- **Target 16–18 min** + questions. Spend **2 min on slide 4 (cascade
  architecture)**, **2 min on slide 10 (normalisation)** and **2 min on
  slide 11 (why replay still underperforms)** — the three things they
  asked for.
- **[TR]** Üç ana çıktı: (1) cascade mimarisi — hocaların istediği, (2)
  normalizasyon replay'i kurtardı, (3) replay'in neden hâlâ tam çözülmediğini
  somut kanıtla anlatıyoruz. Bu üçünde yavaşla.
- **One sentence to remember:** *"We built the interpretable cascade you
  asked for, the normalisation idea recovered replay ranking, and we can now
  show concretely why the last bit of replay is a physical limit, not a bug."*

## Slide 1 — Title
> "Thanks. Three things today: the new cascade architecture, a normalisation
> fix that recovers replay, and concrete evidence for why replay still isn't
> fully solved."

## Slide 2 — The problem
> "A water network is a graph — junctions report pressure, pipes report flow.
> Sensing is sparse, half the readings missing, and noisy. An attacker
> falsifies a subset of sensors. Five attack families. The detector
> reconstructs the truth and flags which sensors lie."

**[TR]** "Stealthy" ve "Replay"i vurgula.

## Slide 3 — Pipeline
> "Network, simulate, inject attacks, detect, evaluate. Only the first stage
> is domain-specific — the rest is shared code, which matters for Part 2."

## Slide 4 — The architecture: cascade routing (2 min)
**[TR]** Hocaların istediği mimari. Diyagramı soldan sağa elinle takip et.
> "This is the architecture. Instead of blending experts opaquely, the router
> ranks them and we run one at a time. Snapshots go to the router — a fast
> classifier — which ranks the experts. We run the top one, then a feedback
> check asks: did it do a good job? On negative feedback we go back and try
> the next expert — the orange loop — until one is accepted. Every prediction
> carries which expert ran and why it was kept or rejected. It's
> interpretable."

## Slide 5 — The feedback mechanism
> "The feedback is label-free, because at inference we have no ground truth.
> Two signals: do the sensors the expert calls 'clean' agree with their
> readings, and does the reconstruction conserve mass? We standardise the two
> terms so neither dominates, restrict re-routing to the router's top-2 for
> stability, and trust a confident router unless the feedback clearly
> disagrees."

**[TR]** Formülü işaret et, okuma. "consistency + physics, etiketsiz" de.

## Slide 6 — The experts
> "The experts the cascade routes among: six, each a GraphSAGE-plus-GRU model
> over the 6-step window. Two training choices matter — direct expert
> supervision, so every expert learns its own class regardless of routing and
> never starves; and a physics loss that forbids hydraulically impossible
> reconstructions. These same experts become the substrate for power and
> traffic in Part 2, unchanged."

**[TR]** Hızlı geç, ~45 saniye.

## Slide 7 — Cascade vs the alternatives
> "Over 5 seeds the cascade reaches 0.772, matching the oracle ceiling — the
> best any hard routing could do — and within a point of a full soft blend.
> The experts are similar here, so the interpretable cascade costs almost no
> accuracy, and it's stable."

**[TR]** "matches the ceiling, stable, costs almost no accuracy" — üçünü vurgula.

## Slide 8 — Detection results
> "Detection over 5 seeds with a calibrated threshold. Four of five attacks
> caught reliably. Replay near zero — the next three slides are why, how we
> recovered most of it, and why it's still not solved."

## Slide 9 — Router diagnostic
> "Is the router mis-routing? On Modena it's healthy, 96%. On Net3, 77% of
> stealthy windows go to the replay expert — both look smooth, 97 nodes too
> few to separate them. The cascade feedback targets exactly this."

## Slide 10 — Recovering replay: normalisation (2 min)
**[TR]** Hocaların "normalize dene" önerisi. Yavaşla.
> "Your hypothesis: water values sit too close together. We tested it.
> Normalising each sensor by its own temporal scale amplifies the tiny
> per-node variation a replay lacks. Middle panel: replay ranking jumps from
> below chance — 0.46 — to AUROC 0.79. Overall F1 holds at 0.833. The
> normalisation idea worked."

## Slide 11 — Why replay still underperforms (2 min)
**[TR]** Bu yeni slayt. Somut kanıt. Yavaşla, iki paneli tek tek göster.
> "But replay's F1 is still low, and here's concretely why. Left: the
> anomaly-score distributions. Replayed sensors — purple — do score higher
> than genuine ones — green — so the ranking works, AUROC 0.75. But the two
> distributions overlap heavily. Right: because they overlap, no threshold
> gives high precision and recall together — the best F1 is 0.41.
>
> The reason is physical. A replay is a verbatim copy of a past reading, so
> the only thing distinguishing it from a genuine value is the fresh
> observation noise it lacks. That residual signal is real but weak — this is
> an information limit, not a training bug."

**[TR]** "overlap → no threshold separates them → physical limit" zincirini
net kur. Bu, "neden hâlâ çözülmedi" sorusunun kanıtlı cevabı.

## Slide 12 — Part 2 divider
> "Does any of this survive outside water?"

## Slide 13 — One detector, three physics
> "Same detector, unchanged, on a power grid — IEEE 118-bus — and a
> road-traffic network. Voltage and speed instead of pressure. Only a data
> adapter changed."

## Slide 14 — Cross-domain F1
> "It holds — F1 0.82 to 0.89 across three unrelated physical systems, no
> tuning."

## Slide 15 — The normalisation fix generalises
> "Two messages in one figure. The grey bars — global normalisation — already
> track signal speed: replay is hardest in slow water (0.46) and trivial in
> fast traffic (0.97). The purple bars show per-node normalisation lifts it
> most where it's hardest — water 0.46 to 0.79, power 0.66 to 0.90 — and
> isn't needed where it's already easy. A principled cross-domain fix, not a
> water-specific hack."

**[TR]** Grey vs purple ayrımını net anlat — hem signal-speed hem fix tek figürde.

## Slide 16 — Difficulty inversion
> "And the hard case flips: water and power find stealthy easy, replay hard;
> traffic is the mirror image. The domain, not the model, decides."

## Slide 17 — Summary
> "Part 1: the interpretable cascade matches the ceiling; the router failure
> diagnosed; per-node normalisation recovered replay ranking; and we showed
> the residual replay gap is a physical limit. Part 2: the detector transfers
> unchanged, the normalisation fix generalises, and replay difficulty tracks
> signal speed. Next steps: push the cascade where experts differ more, extend
> the threat model, pick a venue. I'd value your input on priority."

## Slide 18 — Thank you
> "Thank you — questions?"

---

# Anticipated questions & answers

## On the cascade
**Q: Why not just use the soft mixture — isn't it slightly higher?**
> "It is, by about a point, but it's an opaque blend of all six experts. The
> cascade gives essentially the same accuracy — it matches the oracle ceiling
> — while telling you which expert ran and why. For a safety-critical
> detector, that interpretability is worth more than a point of F1."

**Q: How stable is the cascade?**
> "0.772 plus or minus 0.017 over 5 seeds. Stability comes from restricting
> re-routing to the router's top-2 experts, so the feedback can't land on a
> wildly wrong one."

**Q: What is the feedback signal exactly?**
> "Two label-free terms: reconstruction agreement on the sensors the expert
> calls clean, and mass-conservation of the reconstructed flow. Z-scored
> across experts so neither dominates."

## On replay (the key evidence)
**Q: Why is replay F1 still low if AUROC is 0.79?**
> "That's exactly slide 11. AUROC measures ranking — replayed sensors do score
> higher. But the two score distributions overlap, so no single threshold
> separates them cleanly; the best F1 is 0.41. It's the classic
> high-AUROC-low-F1 situation."

**Q: Could a better feature fix it?**
> "We tested the most direct one — an exact-lag-match detector, since a replay
> is a verbatim copy. On its own it only reached AUROC 0.62, because with 50%
> missing data and replays that reach outside the 6-step window, the exact
> match often isn't visible. The model already captures most of the available
> signal."

**Q: So is replay unsolvable?**
> "Near its information limit on slow-signal domains, yes — the only cue is the
> missing observation noise of a copy. On fast-signal domains like traffic
> it's already easy (AUROC 0.97). The domain's signal dynamics decide, not the
> model."

## On normalisation / cross-domain
**Q: Does per-node normalisation add information?**
> "No — it rescales each sensor by its own variance so the small signal that
> is there isn't swamped. AUROC going 0.46 to 0.79 shows the information was
> present; the model just couldn't exploit it under a global scale."

**Q: Why did power's overall F1 drop under per-node?**
> "The same replay-recall trade-off as water: recovering replay costs a little
> precision elsewhere. Water and traffic actually gained overall F1; power
> traded about three points for a big replay-ranking gain."

**Q: Is the traffic data real?**
> "Synthetic road-sensor graph; the power grid is a standard IEEE-118
> benchmark. A real traffic benchmark like METR-LA is the natural next step —
> only the adapter changes."

## On the thesis
**Q: Contribution / venue?**
> "An interpretable cascade detector for physics-constrained sensor graphs; a
> diagnosis of the routing and replay limits with concrete evidence; a
> normalisation fix that generalises across domains. Fits CPS-security venues
> — I'd like to choose one with you."

---

## If you get a stuck question
1. *"Fair point — we haven't tested that yet; good candidate for the next run."*
2. *"I'd want to check the numbers before answering properly — can I come back?"*
3. *"I'm not sure — what would you expect to see?"*

**Never invent a number.** Unsure → say "about 0.79".

"""Direct expert objectives, independent of router decisions."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def direct_specialist_loss(out, batch, replay_weight=1., hard_negative_fraction=.25):
    """Balance positive and negative risks within each specialist's domain.

    Expert 0 learns all families; other experts learn their family and clean
    episodes. The highest-loss training negatives are retained, without ever
    consulting validation/test labels. Each expert has equal weight regardless
    of how many clean nodes happen to be present in the current minibatch.
    """
    if not 0 < hard_negative_fraction <= 1:
        raise ValueError("hard_negative_fraction must be in (0, 1]")
    logits = out["expert_pressure_anomaly_logits"]
    family = batch["attack_type"].repeat_interleave(batch["num_nodes"])
    observed = batch["pressure_mask"] > 0
    labels = batch["pressure_anomaly"] > .5
    detection, reconstruction = [], []
    for expert in range(logits.shape[1]):
        domain = torch.ones_like(observed) if expert == 0 else ((family == expert) | (family == 0))
        positives = domain & observed & labels
        negatives = domain & observed & ~labels
        terms = []
        if positives.any():
            weight = 1 + (replay_weight-1.)*(family[positives] == 2).float()
            terms.append((F.softplus(-logits[positives, expert])*weight).mean())
        if negatives.any():
            losses = F.softplus(logits[negatives, expert])
            count = max(1, math.ceil(len(losses)*hard_negative_fraction))
            terms.append(losses.topk(count).values.mean())
        if terms:
            detection.append(torch.stack(terms).mean())
        if domain.any():
            reconstruction.append(F.smooth_l1_loss(
                out["expert_pressure_pred"][domain, expert], batch["y_pressure"][domain]))
        if "expert_flow_pred" in out:
            edge_family = batch["attack_type"].repeat_interleave(out["expert_flow_pred"].shape[0]//len(batch["attack_type"]))
            edge_domain = torch.ones_like(edge_family, dtype=torch.bool) if expert == 0 else ((edge_family == expert) | (edge_family == 0))
            if edge_domain.any():
                reconstruction.append(F.smooth_l1_loss(
                    out["expert_flow_pred"][edge_domain, expert], batch["y_flow"][edge_domain]))
    zero = logits.sum()*0.
    return {"anomaly": torch.stack(detection).mean() if detection else zero,
            "reconstruction": torch.stack(reconstruction).mean() if reconstruction else zero}

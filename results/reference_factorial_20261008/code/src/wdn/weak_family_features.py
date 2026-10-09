"""Group-blind pressure/flow context; no attack metadata enter inference."""
from __future__ import annotations

import numpy as np


def pooled_evidence(residual, observed):
    """Causal sparse-change evidence across streams, not sensor-level labels."""
    observed = np.asarray(observed, dtype=bool)
    z = np.where(observed, np.clip(residual, -10., 10.), 0.)
    if z.ndim != 2 or z.shape != observed.shape or not np.isfinite(z).all():
        raise ValueError("Matching finite observed residuals required")
    names = ["tail_2", "tail_3", "energy"]
    names += [f"{name}_{length}" for length in (3, 6, 12)
              for name in ("mean", "ramp", "variance", "support")]
    rows = []
    for t in range(len(z)):
        valid = observed[t]
        count = max(1, int(valid.sum()))
        row = [float((np.abs(z[t]) > 2).sum()/count),
               float((np.abs(z[t]) > 3).sum()/count), float((z[t]**2).sum()/count)]
        for length in (3, 6, 12):
            start = max(0, t-length+1)
            w, x = observed[start:t+1], z[start:t+1]
            n = w.sum(0)
            age = np.arange(1, len(x)+1)[:, None]
            mean = .5*x.sum(0)**2/np.maximum(n, 1)
            ramp = .5*(x*age).sum(0)**2/np.maximum((w*age**2).sum(0), 1.)
            energy = (x*x).sum(0)/np.maximum(n, 1)
            variance = .5*n*np.where(energy > 1, energy-1-np.log(np.maximum(energy, 1.)), 0.)
            supported = n >= 2
            # .02 is a sparse-mixture assumption, not the benchmark target
            # fraction and not a known list/count of attacked sensors.
            for statistic in (mean, ramp, variance):
                evidence = np.logaddexp(np.log(.98), np.log(.02)+np.minimum(statistic, 20.))
                row.append(float(evidence[supported].mean()) if supported.any() else 0.)
            row.append(float(n.mean()/len(x)) if len(n) else 0.)
        rows.append(row)
    return np.asarray(rows, dtype=np.float32), names


def conditional_features(reference, values, observed, pressure_count):
    """Fit once per target group and pool only its excluded-group context.

    Even residuals used for contextual pooling are fitted without the target
    group. Merely dropping those columns after an ordinary fit is insufficient.
    The optional flow block is treated identically to untrusted pressure inputs.
    """
    values = np.asarray(values, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    ref = reference.reference
    if values.ndim != 2 or values.shape != observed.shape or values.shape[1] != len(ref.mean_):
        raise ValueError("Matching arrays in fitted channel order required")
    if not 0 < pressure_count <= values.shape[1] or not np.isfinite(values[observed]).all():
        raise ValueError("Invalid pressure channel count or observed values")
    sigma = ref.noise_scale_
    design = ref.basis_*ref.scale_[:, None]/sigma[:, None]
    response = np.where(observed, values-ref.mean_, 0.)/sigma
    T = len(values)
    predicted = np.zeros((T, pressure_count))
    spread, support = np.zeros_like(predicted), np.zeros_like(predicted)
    channels = [("pressure", np.arange(pressure_count))]
    if pressure_count < values.shape[1]:
        channels.append(("flow", np.arange(pressure_count, values.shape[1])))
    context = np.zeros((T, pressure_count, len(channels)*15), dtype=np.float32)
    names = None
    for group, subsets in enumerate(reference.subsets_):
        excluded = ref.group_ == group
        target = excluded[:pressure_count]
        if not target.any():
            continue
        views = []
        for subset in subsets:
            visible = observed & subset[None, :]
            weights = visible.astype(float)
            for step in range(reference.steps):
                gram = np.einsum("tn,nr,ns->trs", weights, design, design, optimize=True)
                gram += np.eye(ref.rank)[None]*ref.ridge
                latent = np.linalg.solve(gram, ((weights*response)@design)[..., None])[..., 0]
                fitted = latent@design.T
                if step+1 < reference.steps:
                    inverse = np.linalg.inv(gram)
                    leverage = np.einsum("nr,trs,ns->tn", design, inverse, design, optimize=True)*weights
                    error = np.abs(np.where(visible, response-fitted, 0.))/np.sqrt(np.maximum(1-leverage, .1))
                    robust = np.minimum(1., 2.5/np.maximum(error, 1e-12)) if step < 2 else np.maximum(np.maximum(0., 1-(error/reference.cutoff)**2)**2, .01)
                    weights = visible*robust
            views.append(fitted*sigma+ref.mean_)
        views = np.asarray(views)
        center = np.median(views, axis=0)
        predicted[:, target] = center[:, :pressure_count][:, target]
        spread[:, target] = np.median(np.abs(views[:, :, :pressure_count][:, :, target]-predicted[None, :, target]), axis=0)
        available = observed & ~excluded[None, :]
        support[:, target] = available.sum(1)[:, None]/max(1, int((~excluded).sum()))
        residual = np.where(available, (values-center)/reference.noise_scale_, 0.)
        pieces, found = [], []
        for name, indices in channels:
            indices = indices[~excluded[indices]]
            piece, labels = pooled_evidence(residual[:, indices], available[:, indices])
            pieces.append(piece)
            found.extend(f"context_{name}_{label}" for label in labels)
        context[:, target] = np.concatenate(pieces, axis=1)[:, None, :]
        if names is not None and names != found:
            raise AssertionError("Context schema changed across groups")
        names = found
    return predicted, support, spread, context, names

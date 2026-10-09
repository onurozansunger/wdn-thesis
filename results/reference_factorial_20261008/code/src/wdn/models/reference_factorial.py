"""Target-group exclusion × residual reweighting on a frozen latent reference.

Both scale arrays remain those of the supplied TRAIN-fitted anchor: the inner
reference scale whitens the solve, and the wrapper scale normalizes downstream
features. This isolates the prediction mechanism without changing pressure-
history binning or refitting the latent representation.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from wdn.models.robust_reference import RobustBlindReference


class ReferenceFactorial(RobustBlindReference):
    """Make one independent arm of the fixed-scale reference factorial.

    ``exclude_target_group=False`` admits the target group through the same
    pre-exclusion view masks used by the anchor. Every non-target sensor keeps
    its original view membership. The original blind-mask fallback decision is
    shared across arms, so changing exclusion cannot change non-target masks.

    ``robust_reweighting=False`` performs one noise-whitened ridge solve per
    view. It disables only iterative residual reweighting; three-view medians,
    disagreement MADs, ridge regularization and both TRAIN scales remain fixed.
    Support continues to mean the observed fraction outside the target group.

    The enabled/enabled arm inherits the original prediction implementation
    unchanged. The anchor and its arrays are deep-copied before any arm changes.
    """

    def __init__(self, anchor: RobustBlindReference, *,
                 exclude_target_group: bool, robust_reweighting: bool):
        if not isinstance(anchor, RobustBlindReference):
            raise TypeError("A fitted RobustBlindReference anchor is required")
        if isinstance(anchor, ReferenceFactorial):
            raise ValueError("Use the original frozen anchor, not another factorial arm")
        if not isinstance(exclude_target_group, bool) or not isinstance(robust_reweighting, bool):
            raise TypeError("Factor settings must be booleans")
        if anchor.views != 3 or anchor.steps != 6:
            raise ValueError("The frozen anchor must use three views and six fitting steps")

        # Preserve every fitted attribute, including any future provenance
        # fields, while preventing one arm from mutating another or its anchor.
        self.__dict__.update(deepcopy(anchor.__dict__))
        ref = self.reference
        sensor_count = len(ref.mean_)
        if (ref.group_.shape != (sensor_count,) or
                ref.scale_.shape != (sensor_count,) or
                ref.basis_.shape != (sensor_count, ref.rank) or
                len(self.subsets_) != ref.groups):
            raise ValueError("The anchor has incompatible fitted dimensions or views")
        if (ref.noise_scale_ is None or
                np.shape(ref.noise_scale_) != (sensor_count,) or
                np.shape(self.noise_scale_) != (sensor_count,) or
                not np.isfinite(ref.noise_scale_).all() or
                not np.isfinite(self.noise_scale_).all() or
                np.any(ref.noise_scale_ <= 0) or np.any(self.noise_scale_ <= 0)):
            raise ValueError("The anchor must contain both positive, fitted TRAIN scales")

        rng = np.random.default_rng(self.seed)
        unmasked_subsets = []
        for group in range(ref.groups):
            outside = ref.group_ != group
            selectors = [np.ones(sensor_count, dtype=bool)]
            for _ in range(self.views - 1):
                selector = rng.random(sensor_count) < .8
                # Use the anchor's fallback condition for both exclusion arms.
                if np.count_nonzero(selector & outside) <= ref.rank:
                    selector = np.ones(sensor_count, dtype=bool)
                selectors.append(selector)
            if len(self.subsets_[group]) != self.views:
                raise ValueError("The anchor has an incompatible number of views")
            for view, selector in enumerate(selectors):
                if not np.array_equal(self.subsets_[group][view], selector & outside):
                    raise ValueError("Anchor view masks do not match its recorded seed and recipe")
            unmasked_subsets.append(selectors)

        self.exclude_target_group = exclude_target_group
        self.robust_reweighting = robust_reweighting
        self.anchor_steps = anchor.steps
        if not exclude_target_group:
            self.subsets_ = unmasked_subsets
        if not robust_reweighting:
            self.steps = 1

    def calibrate_scale(self, normal_values, normal_observed):
        """Reject scale changes: all factorial arms use the frozen anchor scale."""
        raise RuntimeError("Factorial scales are frozen; do not recalibrate an arm")

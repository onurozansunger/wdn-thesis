"""Corruption pipeline: apply missing data, noise, and attacks to clean snapshots.

Takes ground-truth pressure/flow values and produces corrupted observations
with masks indicating which values are observed.

Attack types (from WDN cyber-security literature):
    1. Random falsification: scale + bias on random sensors
    2. Replay attack: replace current reading with a past value
    3. Stealthy bias injection: small gradual drift that's hard to detect
    4. Targeted attack: attack sensors with highest impact on the network
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass

from wdn.config import CorruptionConfig


# Attack type integer IDs. 0 = clean, 1..5 = individual attacks.
ATTACK_TYPE_TO_ID = {
    "clean": 0,
    "random": 1,
    "replay": 2,
    "stealthy": 3,
    "noise": 4,
    "targeted": 5,
}
ID_TO_ATTACK_TYPE = {v: k for k, v in ATTACK_TYPE_TO_ID.items()}
NUM_ATTACK_CLASSES = len(ATTACK_TYPE_TO_ID)

# Every attack the "mixed" pipeline can draw from. A config may restrict
# this pool (``CorruptionConfig.attack_pool``) to generate a dataset that
# covers only a subset of the threat model.
ALL_ATTACKS = ["random", "replay", "stealthy", "noise", "targeted"]


def active_attack_classes(corrupted: list) -> list[str]:
    """Attack classes actually present in a corrupted dataset.

    Returned in canonical id order, so "clean" always comes first. Used to
    size the router / expert bank to the threat model the data covers
    rather than to the full five-attack vocabulary.
    """
    present = {int(getattr(c, "attack_type_id", 0)) for c in corrupted}
    return [ID_TO_ATTACK_TYPE[i] for i in sorted(present)]


def compact_id_map(class_names: list[str]) -> dict[int, int]:
    """Map canonical attack ids onto a contiguous 0..K-1 range.

    With a restricted attack pool the canonical ids are sparse (dropping
    replay leaves 0,1,3,4,5). The router is a K-way classifier and each
    expert is indexed by class, so the ids have to be compacted first.
    """
    return {ATTACK_TYPE_TO_ID[name]: k for k, name in enumerate(class_names)}


@dataclass
class CorruptedSnapshot:
    """Corrupted observations for a single snapshot.

    These get combined with the clean Snapshot to form the model input.
    """

    # Observed values (with noise; missing values set to 0)
    pressure_obs: torch.Tensor         # (N,)
    flow_obs: torch.Tensor             # (NE,)

    # Binary masks: 1 = observed, 0 = missing
    pressure_mask: torch.Tensor        # (N,)
    flow_mask: torch.Tensor            # (NE,)

    # Anomaly labels: 1 = attacked, 0 = clean
    pressure_anomaly: torch.Tensor     # (N,)
    flow_anomaly: torch.Tensor         # (NE,)

    # Dominant attack type applied to this snapshot.
    # 0 = clean / no attack, 1..5 = individual attacks.
    # Optional so older pickled datasets still load fine.
    attack_type_id: int = 0


def corrupt_snapshot(
    pressure_true: torch.Tensor,
    flow_true: torch.Tensor,
    cfg: CorruptionConfig,
    rng: np.random.Generator,
    replay_buffer: dict | None = None,
    snapshot_idx: int = 0,
    episode: dict | None = None,
) -> CorruptedSnapshot:
    """Apply corruption to a single snapshot's ground truth values.

    Pipeline:
        1. Generate missing-data masks (Bernoulli)
        2. Add Gaussian noise to observed values
        3. (Optional) Apply adversarial attacks
        4. Zero out missing values

    Args:
        pressure_true: (N,) ground truth pressures.
        flow_true: (NE,) ground truth flows.
        cfg: Corruption parameters.
        rng: NumPy random generator for reproducibility.
        replay_buffer: Dict with past observations for replay attacks.
        snapshot_idx: Current snapshot index (for stealthy drift).

    Returns:
        CorruptedSnapshot with observations, masks, and anomaly labels.
    """
    N = pressure_true.shape[0]
    NE = flow_true.shape[0]

    # ------------------------------------------------------------------
    # Step 1: Missing data masks
    # mask=1 means observed, mask=0 means missing
    # ------------------------------------------------------------------
    p_mask = torch.tensor(
        rng.random(N) >= cfg.missing_rate_pressure,
        dtype=torch.float32,
    )
    q_mask = torch.tensor(
        rng.random(NE) >= cfg.missing_rate_flow,
        dtype=torch.float32,
    )

    # ------------------------------------------------------------------
    # Step 2: Add Gaussian noise to observed values
    # ------------------------------------------------------------------
    p_obs = pressure_true.clone()
    q_obs = flow_true.clone()

    if cfg.noise_sigma_pressure > 0:
        p_noise = torch.tensor(
            rng.normal(0, cfg.noise_sigma_pressure, size=N),
            dtype=torch.float32,
        )
        p_obs = p_obs + p_noise * p_mask  # only add noise to observed values

    if cfg.noise_sigma_flow > 0:
        q_noise = torch.tensor(
            rng.normal(0, cfg.noise_sigma_flow, size=NE),
            dtype=torch.float32,
        )
        q_obs = q_obs + q_noise * q_mask

    # ------------------------------------------------------------------
    # Step 3: Anomaly labels (clean by default)
    # ------------------------------------------------------------------
    p_anomaly = torch.zeros(N, dtype=torch.float32)
    q_anomaly = torch.zeros(NE, dtype=torch.float32)
    attack_type_id = ATTACK_TYPE_TO_ID["clean"]

    if cfg.attack_enabled:
        p_obs, q_obs, p_anomaly, q_anomaly, attack_type_id = _apply_attacks(
            p_obs, q_obs, p_mask, q_mask,
            pressure_true, flow_true,
            cfg, rng,
            replay_buffer=replay_buffer,
            snapshot_idx=snapshot_idx,
            episode=episode,
        )

    # ------------------------------------------------------------------
    # Step 4: Zero out missing values
    # ------------------------------------------------------------------
    p_obs = p_obs * p_mask
    q_obs = q_obs * q_mask

    return CorruptedSnapshot(
        pressure_obs=p_obs,
        flow_obs=q_obs,
        pressure_mask=p_mask,
        flow_mask=q_mask,
        pressure_anomaly=p_anomaly,
        flow_anomaly=q_anomaly,
        attack_type_id=int(attack_type_id),
    )


# -----------------------------------------------------------------------
# Attack implementations
# -----------------------------------------------------------------------

def _select_targets(
    mask: torch.Tensor,
    fraction: float,
    rng: np.random.Generator,
    high_impact_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Select which observed sensors to attack.

    Args:
        mask: (K,) binary observation mask.
        fraction: Fraction of observed sensors to attack.
        rng: Random generator.
        high_impact_indices: If provided, preferentially attack these indices.

    Returns:
        Array of indices to attack.
    """
    observed = torch.where(mask > 0)[0].numpy()
    if len(observed) == 0:
        return np.array([], dtype=int)

    n_attack = max(1, int(len(observed) * fraction))
    n_attack = min(n_attack, len(observed))

    if high_impact_indices is not None:
        # Prefer high-impact sensors (targeted attack)
        candidates = np.intersect1d(observed, high_impact_indices)
        if len(candidates) >= n_attack:
            return rng.choice(candidates, size=n_attack, replace=False)
        # Fill remaining from other observed sensors
        remaining = np.setdiff1d(observed, candidates)
        n_extra = n_attack - len(candidates)
        if len(remaining) > 0 and n_extra > 0:
            extra = rng.choice(remaining, size=min(n_extra, len(remaining)), replace=False)
            return np.concatenate([candidates, extra])
        return candidates

    return rng.choice(observed, size=n_attack, replace=False)


def _attack_random_falsification(
    obs: torch.Tensor,
    targets: np.ndarray,
    scale: float,
    bias: float,
) -> torch.Tensor:
    """Random falsification: obs_new = obs * scale + bias.

    Simple but effective — mimics a compromised sensor sending
    scaled/offset readings.
    """
    out = obs.clone()
    for idx in targets:
        out[idx] = out[idx] * scale + bias
    return out


def _attack_replay(
    obs: torch.Tensor,
    targets: np.ndarray,
    replay_values: torch.Tensor | None,
) -> torch.Tensor:
    """Replay attack: replace current reading with a past value.

    The attacker records legitimate sensor readings and replays them
    later to mask real changes in the network state. This is particularly
    dangerous because individual replayed values look realistic.
    """
    out = obs.clone()
    if replay_values is None:
        return out  # no history yet, skip
    for idx in targets:
        if idx < len(replay_values):
            out[idx] = replay_values[idx]
    return out


def _attack_stealthy_bias(
    obs: torch.Tensor,
    targets: np.ndarray,
    rng: np.random.Generator,
    snapshot_idx: int,
    max_drift: float = 5.0,
    ramp_steps: int = 20,
    directions: dict[int, float] | None = None,
) -> torch.Tensor:
    """Stealthy bias injection: small gradual drift over time.

    Instead of a sudden large change, the attacker slowly shifts
    readings. By the time the drift is large enough to matter,
    operators have adjusted to the "new normal".

    drift(t) = max_drift * min(t / ramp_steps, 1.0) * direction
    """
    out = obs.clone()
    # How far along the ramp are we?
    # The first attacked step must contain a non-zero perturbation; otherwise
    # it would be labelled anomalous while being numerically identical to a
    # clean observation.
    ramp_factor = min((snapshot_idx + 1) / max(ramp_steps, 1), 1.0)
    for idx in targets:
        direction = (
            directions[int(idx)]
            if directions is not None and int(idx) in directions
            else float(rng.choice([-1.0, 1.0]))
        )
        drift = max_drift * ramp_factor * direction
        out[idx] = out[idx] + drift
    return out


def _attack_gaussian_noise_injection(
    obs: torch.Tensor,
    targets: np.ndarray,
    rng: np.random.Generator,
    noise_multiplier: float = 5.0,
) -> torch.Tensor:
    """Noise injection: add large Gaussian noise to readings.

    Simulates a malfunctioning or jammed sensor producing noisy output.
    The readings fluctuate wildly around the true value.
    """
    out = obs.clone()
    for idx in targets:
        noise = rng.normal(0, abs(out[idx].item()) * noise_multiplier * 0.1 + 1.0)
        out[idx] = out[idx] + noise
    return out


def _apply_attacks(
    p_obs: torch.Tensor,
    q_obs: torch.Tensor,
    p_mask: torch.Tensor,
    q_mask: torch.Tensor,
    p_true: torch.Tensor,
    q_true: torch.Tensor,
    cfg: CorruptionConfig,
    rng: np.random.Generator,
    replay_buffer: dict | None = None,
    snapshot_idx: int = 0,
    episode: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Apply adversarial attacks to observed values.

    Only attacks *observed* sensors (can't attack what's not there).

    Supported attack types:
        - "random": Random falsification (scale + bias)
        - "replay": Replay past legitimate readings
        - "stealthy": Gradual bias drift over time
        - "noise": Inject large random noise
        - "mixed": Randomly pick from all attack types per snapshot

    Returns updated (p_obs, q_obs, p_anomaly, q_anomaly, attack_type_id).
    """
    N = p_obs.shape[0]
    NE = q_obs.shape[0]

    p_anomaly = torch.zeros(N, dtype=torch.float32)
    q_anomaly = torch.zeros(NE, dtype=torch.float32)

    # Track whether the "targeted" variant was requested before it gets
    # rewritten to "random" below (targeted = random with a biased
    # target-selection strategy, but we still want its own label).
    was_targeted = (cfg.attack_type == "targeted")

    # Determine high-impact nodes for targeted attacks
    high_impact_p = None
    high_impact_q = None
    if cfg.attack_type == "targeted":
        # Attack nodes with highest pressure variance (most informative)
        p_vals = p_true.numpy()
        top_k = max(1, int(N * 0.3))
        high_impact_p = np.argsort(np.abs(p_vals - np.mean(p_vals)))[-top_k:]
        q_vals = q_true.numpy()
        top_k_q = max(1, int(NE * 0.3))
        high_impact_q = np.argsort(np.abs(q_vals))[-top_k_q:]

    # Select targets. Inside an episode the compromised set was fixed when
    # the episode began; we intersect it with whatever is reporting now,
    # because a compromised sensor stays compromised whether or not it
    # happens to transmit this step.
    if episode is not None:
        obs_p = set(torch.where(p_mask > 0)[0].numpy().tolist())
        obs_q = set(torch.where(q_mask > 0)[0].numpy().tolist())
        targets_p = np.array(sorted(obs_p & episode["targets_p"]), dtype=int)
        targets_q = np.array(sorted(obs_q & episode["targets_q"]), dtype=int)
    else:
        targets_p = _select_targets(p_mask, cfg.attack_fraction, rng, high_impact_p)
        targets_q = _select_targets(q_mask, cfg.attack_fraction, rng, high_impact_q)

    # Resolve attack type (for "mixed", pick one randomly per snapshot).
    # "mixed" spans every variant in cfg.attack_pool (all five by default,
    # including targeted) so the downstream attack router sees every class
    # the dataset is meant to cover during training.
    attack = cfg.attack_type
    if episode is not None:
        attack = episode["family"]
        was_targeted = (attack == "targeted")
        if attack == "clean":
            return p_obs, q_obs, p_anomaly, q_anomaly, ATTACK_TYPE_TO_ID["clean"]
    elif attack == "mixed":
        attack = rng.choice(list(cfg.attack_pool or ALL_ATTACKS))
        # "clean" in the pool means this window is left untouched: normal
        # operation, so the detector is also evaluated on attack-free
        # traffic and the router gets a genuine no-attack class.
        if attack == "clean":
            return p_obs, q_obs, p_anomaly, q_anomaly, ATTACK_TYPE_TO_ID["clean"]
        # If targeted is chosen, re-derive the high-impact target lists so
        # the selection logic uses them for this snapshot.
        if attack == "targeted":
            p_vals = p_true.numpy()
            top_k = max(1, int(N * 0.3))
            high_impact_p = np.argsort(np.abs(p_vals - np.mean(p_vals)))[-top_k:]
            q_vals = q_true.numpy()
            top_k_q = max(1, int(NE * 0.3))
            high_impact_q = np.argsort(np.abs(q_vals))[-top_k_q:]
            targets_p = _select_targets(p_mask, cfg.attack_fraction, rng, high_impact_p)
            targets_q = _select_targets(q_mask, cfg.attack_fraction, rng, high_impact_q)
            was_targeted = True
    if attack == "targeted":
        attack = "random"  # targeted just changes sensor selection, uses random falsification

    # Apply the chosen attack.
    # k-step replay: in an episode, use the fixed lag sampled when the
    # episode started. Outside episode mode the lag is redrawn per snapshot,
    # which is the intended IID negative control. Do not shorten the lag when
    # history is unavailable: no replay has occurred yet, so those early
    # observations must remain clean rather than receive a false attack label.
    replay_p = replay_q = None
    if replay_buffer:
        p_hist = replay_buffer.get("pressure_history", [])
        q_hist = replay_buffer.get("flow_history", [])
        lag_lo = max(1, int(getattr(cfg, "replay_lag_min", 3)))
        lag_hi = max(lag_lo, int(getattr(cfg, "replay_lag_max", 6)))
        lag = (
            int(episode["replay_lag"])
            if episode is not None
            else int(rng.integers(lag_lo, lag_hi + 1))
        )
        if len(p_hist) >= lag:
            replay_p = p_hist[-lag]
            replay_q = q_hist[-lag] if len(q_hist) >= lag else None

    if attack == "replay" and replay_p is None and replay_q is None:
        return p_obs, q_obs, p_anomaly, q_anomaly, ATTACK_TYPE_TO_ID["clean"]

    if attack == "random":
        p_obs = _attack_random_falsification(p_obs, targets_p, cfg.attack_scale, cfg.attack_bias)
        q_obs = _attack_random_falsification(q_obs, targets_q, cfg.attack_scale, cfg.attack_bias)
    elif attack == "replay":
        p_obs = _attack_replay(p_obs, targets_p, replay_p)
        q_obs = _attack_replay(q_obs, targets_q, replay_q)
    elif attack == "stealthy":
        # A drift has to ramp from the moment the attack starts. With
        # episodes off this is the global index, which is why the drift
        # never actually drifted on any one sensor.
        drift_t = snapshot_idx if episode is None else episode["age"]
        ramp_steps = int(getattr(cfg, "stealthy_ramp_steps", 20))
        p_directions = episode.get("directions_p") if episode is not None else None
        q_directions = episode.get("directions_q") if episode is not None else None
        p_obs = _attack_stealthy_bias(
            p_obs, targets_p, rng, drift_t,
            max_drift=cfg.attack_bias,
            ramp_steps=ramp_steps,
            directions=p_directions,
        )
        q_obs = _attack_stealthy_bias(
            q_obs, targets_q, rng, drift_t,
            max_drift=cfg.attack_bias * 0.1,
            ramp_steps=ramp_steps,
            directions=q_directions,
        )
    elif attack == "noise":
        nm = getattr(cfg, "attack_noise_multiplier", 5.0)
        p_obs = _attack_gaussian_noise_injection(p_obs, targets_p, rng, nm)
        q_obs = _attack_gaussian_noise_injection(q_obs, targets_q, rng, nm)

    # Mark attacked sensors
    for idx in targets_p:
        p_anomaly[idx] = 1.0
    for idx in targets_q:
        q_anomaly[idx] = 1.0

    # Resolve which attack label to report. For "mixed" we report the
    # concrete attack that was chosen. For "targeted" we keep its own
    # label because its selection strategy is distinctive.
    if was_targeted:
        label_name = "targeted"
    else:
        label_name = attack

    # If no sensors were actually attacked (empty targets), flag as clean
    # so the router does not learn to associate the empty label with a
    # specific attack type.
    if len(targets_p) == 0 and len(targets_q) == 0:
        label_name = "clean"

    attack_type_id = ATTACK_TYPE_TO_ID.get(label_name, 0)

    return p_obs, q_obs, p_anomaly, q_anomaly, attack_type_id


# -----------------------------------------------------------------------
# Batch corruption
# -----------------------------------------------------------------------

def corrupt_all_snapshots(
    snapshots: list,
    cfg: CorruptionConfig,
    seed: int = 42,
) -> list[CorruptedSnapshot]:
    """Apply corruption to all snapshots.

    Maintains a replay buffer for replay attacks (stores the previous
    snapshot's observations to use as replay values).

    Args:
        snapshots: List of Snapshot objects with .pressure_true and .flow_true.
        cfg: Corruption configuration.
        seed: Random seed.

    Returns:
        List of CorruptedSnapshot objects (same order as input).
    """
    rng = np.random.default_rng(seed)
    # Episode identity/length must not depend on how many node-level random
    # draws a network consumes. A dedicated RNG makes a same-seed Modena and
    # L-Town pair share the same family schedule even though their graph sizes
    # differ. Sensor targets and measurement noise still use ``rng``.
    schedule_rng = np.random.default_rng(seed + 1_000_003)
    corrupted = []
    # Replay buffer holds a short HISTORY of recent true snapshots so a
    # k-step replay can re-broadcast a genuinely stale reading (a
    # 1-step replay of a slow hydraulic signal is within observation
    # noise of the truth and near-undetectable). The history resets at
    # every scenario boundary so a scenario never replays another's data.
    replay_buffer = {"pressure_history": [], "flow_history": []}
    prev_scenario = None

    # Episode scheduling. When enabled, one family and one compromised
    # sensor set are held for a run of consecutive snapshots, so a
    # detector window can sit inside a single coherent attack. The set is
    # drawn over all sensors, not just the ones reporting at the moment it
    # starts, since a compromise does not end when a sensor drops a packet.
    ep_on = getattr(cfg, "attack_episode_min", 0) > 0
    balanced_families = bool(getattr(cfg, "balanced_episode_families", False))
    family_queue: list[str] = []

    def next_episode_family() -> str:
        nonlocal family_queue
        pool = [str(name) for name in (cfg.attack_pool or ALL_ATTACKS)]
        if not balanced_families:
            return str(schedule_rng.choice(pool))
        if not family_queue:
            family_queue = pool.copy()
            schedule_rng.shuffle(family_queue)
        return family_queue.pop()

    episode = None
    for i, snap in enumerate(snapshots):
        scenario = getattr(snap, "scenario_id", None)
        if scenario != prev_scenario:
            replay_buffer = {"pressure_history": [], "flow_history": []}
            prev_scenario = scenario
            episode = None          # an episode never spans two scenarios

        if ep_on:
            if episode is None or episode["age"] >= episode["length"]:
                N, NE = snap.pressure_true.shape[0], snap.flow_true.shape[0]
                fam = next_episode_family()
                lo = int(cfg.attack_episode_min)
                hi = max(lo, int(getattr(cfg, "attack_episode_max", lo)))
                nP = max(1, int(N * cfg.attack_fraction))
                nQ = max(1, int(NE * cfg.attack_fraction))
                if fam == "targeted":
                    pv = snap.pressure_true.numpy()
                    pool_p = np.argsort(np.abs(pv - pv.mean()))[-max(1, int(N * 0.3)):]
                    qv = snap.flow_true.numpy()
                    pool_q = np.argsort(np.abs(qv))[-max(1, int(NE * 0.3)):]
                else:
                    pool_p, pool_q = np.arange(N), np.arange(NE)
                targets_p = set(rng.choice(
                    pool_p, size=min(nP, len(pool_p)), replace=False,
                ).tolist())
                targets_q = set(rng.choice(
                    pool_q, size=min(nQ, len(pool_q)), replace=False,
                ).tolist())
                lag_lo = max(1, int(getattr(cfg, "replay_lag_min", 3)))
                lag_hi = max(lag_lo, int(getattr(cfg, "replay_lag_max", 6)))
                episode = {
                    "family": fam,
                    "length": int(schedule_rng.integers(lo, hi + 1)),
                    "age": 0,
                    "targets_p": targets_p,
                    "targets_q": targets_q,
                    "directions_p": {
                        idx: float(rng.choice([-1.0, 1.0])) for idx in targets_p
                    },
                    "directions_q": {
                        idx: float(rng.choice([-1.0, 1.0])) for idx in targets_q
                    },
                    "replay_lag": int(schedule_rng.integers(lag_lo, lag_hi + 1)),
                }

        c = corrupt_snapshot(
            snap.pressure_true, snap.flow_true, cfg, rng,
            replay_buffer=replay_buffer,
            snapshot_idx=i,
            episode=episode,
        )
        if episode is not None:
            episode["age"] += 1
        corrupted.append(c)

        # Record the current clean reading for future replay; keep only
        # the last 8 steps (enough for a k<=6 lag plus margin).
        replay_reachable = cfg.attack_type == "replay" or (
            cfg.attack_type == "mixed"
            and "replay" in (cfg.attack_pool or ALL_ATTACKS)
        )
        if cfg.attack_enabled and replay_reachable:
            p_rec, q_rec = snap.pressure_true.clone(), snap.flow_true.clone()
            if getattr(cfg, "replay_records_observation", False):
                # Record what the sensor reported, not the ground truth: an
                # attacker tapping the stream captures the noise too. The
                # noise is drawn fresh here because it is independent of the
                # noise on the step the value is later replayed into.
                p_rec = p_rec + torch.tensor(
                    rng.normal(0, cfg.noise_sigma_pressure, p_rec.shape[0]),
                    dtype=torch.float32)
                q_rec = q_rec + torch.tensor(
                    rng.normal(0, cfg.noise_sigma_flow, q_rec.shape[0]),
                    dtype=torch.float32)
            replay_buffer["pressure_history"].append(p_rec)
            replay_buffer["flow_history"].append(q_rec)
            replay_buffer["pressure_history"] = replay_buffer["pressure_history"][-8:]
            replay_buffer["flow_history"] = replay_buffer["flow_history"][-8:]

    return corrupted

from types import SimpleNamespace

import torch

from wdn.config import CorruptionConfig
from wdn.corruption import ATTACK_TYPE_TO_ID, corrupt_all_snapshots

from wdn.operational_data import OperationalConfig, corrupt_operational


def snapshots(values: list[float], n_pressure: int = 4, n_flow: int = 3):
    return [
        SimpleNamespace(
            scenario_id=0,
            timestep=i,
            pressure_true=torch.full((n_pressure,), value, dtype=torch.float32),
            flow_true=torch.full((n_flow,), value, dtype=torch.float32),
        )
        for i, value in enumerate(values)
    ]


def episode_config(family: str, length: int) -> CorruptionConfig:
    return CorruptionConfig(
        missing_rate_pressure=0.0,
        missing_rate_flow=0.0,
        noise_sigma_pressure=0.0,
        noise_sigma_flow=0.0,
        attack_enabled=True,
        attack_fraction=0.5,
        attack_bias=10.0,
        attack_type="mixed",
        attack_pool=[family],
        attack_episode_min=length,
        attack_episode_max=length,
    )


def test_stealthy_episode_has_fixed_targets_direction_and_monotone_drift():
    cfg = episode_config("stealthy", length=6)
    cfg.stealthy_ramp_steps = 10
    clean = snapshots([0.0] * 6)

    corrupted = corrupt_all_snapshots(clean, cfg, seed=7)
    attacked = corrupted[0].pressure_anomaly.bool()

    assert attacked.any()
    for item in corrupted:
        assert torch.equal(item.pressure_anomaly.bool(), attacked)
        assert item.attack_type_id == ATTACK_TYPE_TO_ID["stealthy"]

    deltas = torch.stack([item.pressure_obs[attacked] for item in corrupted])
    signs = torch.sign(deltas)
    assert torch.all(signs == signs[0])
    assert torch.all(deltas.abs()[1:] > deltas.abs()[:-1])


def test_replay_uses_one_fixed_lag_and_does_not_label_missing_history():
    cfg = episode_config("replay", length=8)
    cfg.attack_fraction = 1.0
    cfg.replay_lag_min = 2
    cfg.replay_lag_max = 2
    clean = snapshots([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    corrupted = corrupt_all_snapshots(clean, cfg, seed=11)

    for t in (0, 1):
        assert corrupted[t].attack_type_id == ATTACK_TYPE_TO_ID["clean"]
        assert corrupted[t].pressure_anomaly.sum().item() == 0
        assert torch.allclose(corrupted[t].pressure_obs, clean[t].pressure_true)

    for t in range(2, len(clean)):
        assert corrupted[t].attack_type_id == ATTACK_TYPE_TO_ID["replay"]
        assert torch.all(corrupted[t].pressure_anomaly == 1)
        assert torch.allclose(corrupted[t].pressure_obs, clean[t - 2].pressure_true)


def test_episode_resets_at_scenario_boundary():
    cfg = episode_config("stealthy", length=10)
    first = snapshots([0.0, 0.0])
    second = snapshots([0.0, 0.0])
    for item in second:
        item.scenario_id = 1

    corrupted = corrupt_all_snapshots(first + second, cfg, seed=3)

    first_start = corrupted[0].pressure_obs.abs().max().item()
    first_end = corrupted[1].pressure_obs.abs().max().item()
    second_start = corrupted[2].pressure_obs.abs().max().item()
    second_end = corrupted[3].pressure_obs.abs().max().item()
    assert first_end > first_start
    assert second_end > second_start


def test_balanced_family_schedule_is_independent_of_network_size():
    cfg = episode_config("stealthy", length=2)
    cfg.attack_pool = ["clean", "random", "stealthy", "noise", "targeted"]
    cfg.balanced_episode_families = True
    small = snapshots([0.0] * 20, n_pressure=4, n_flow=3)
    large = snapshots([0.0] * 20, n_pressure=12, n_flow=9)

    small_corrupted = corrupt_all_snapshots(small, cfg, seed=19)
    large_corrupted = corrupt_all_snapshots(large, cfg, seed=19)

    assert [item.attack_type_id for item in small_corrupted] == [
        item.attack_type_id for item in large_corrupted
    ]


def test_operational_units_masks_and_replay_sources():
    import numpy as np
    from wdn.corruption import corrupt_snapshot

    cfg = OperationalConfig(seed=31, clean_gap_hours=(2, 2),
        attack_duration_hours=(4, 4), replay_lag_hours=(2, 2),
        pressure_bias_m=(1., 1.), flow_bias_m3s=(.001, .001),
        attack_fraction=.5)
    clean = snapshots([float(i) for i in range(160)], n_pressure=64, n_flow=64)
    corrupted, ledger = corrupt_operational(clean, cfg)
    rng = np.random.default_rng(cfg.seed)
    base_cfg = CorruptionConfig(missing_rate_pressure=.5, missing_rate_flow=.5,
        noise_sigma_pressure=cfg.pressure_noise_sigma_m,
        noise_sigma_flow=cfg.flow_noise_sigma_m3s)
    base = [corrupt_snapshot(s.pressure_true, s.flow_true, base_cfg, rng) for s in clean]
    for c, b in zip(corrupted, base):
        for channel in ("pressure", "flow"):
            assert torch.equal(getattr(c, f"{channel}_mask"), getattr(b, f"{channel}_mask"))
            assert not (getattr(c, f"{channel}_anomaly").bool() &
                        ~getattr(c, f"{channel}_mask").bool()).any()
        if c.attack_type_id in (1, 5):
            for channel, expected in (("pressure", 1.), ("flow", .001)):
                attacked = getattr(c, f"{channel}_anomaly").bool()
                delta = (getattr(c, f"{channel}_obs")-getattr(b, f"{channel}_obs"))[attacked].abs()
                assert torch.allclose(delta, torch.full_like(delta, expected), atol=1e-5)
    references = [ref for event in ledger for ref in event["replay_sources"]]
    assert references
    for ref in references:
        channel, selected = ref["channel"], ref["sensors"]
        assert ref["source_index"] < ref["target_index"]
        assert getattr(base[ref["source_index"]], f"{channel}_mask")[selected].all()
        assert torch.equal(getattr(corrupted[ref["target_index"]], f"{channel}_obs")[selected],
                           getattr(base[ref["source_index"]], f"{channel}_obs")[selected])


def test_operational_rejects_changed_missing_rate():
    import pytest
    with pytest.raises(ValueError, match="0.50"):
        corrupt_operational(snapshots([1., 2.]), OperationalConfig(missing_rate_flow=.3))


def test_operational_noise_has_no_hidden_one_unit_floor():
    cfg = OperationalConfig(seed=7, clean_gap_hours=(1, 1), attack_duration_hours=(3, 3),
        pressure_noise_sigma_m=0., flow_noise_sigma_m3s=0., attack_fraction=.5)
    clean = snapshots([1.] * 100, n_pressure=32, n_flow=32)
    corrupted, ledger = corrupt_operational(clean, cfg)
    noise_events = [e for e in ledger if e["family"] == "noise"]
    assert noise_events
    for event in noise_events:
        for c in corrupted[event["start_timestep"]:event["start_timestep"]+event["actual_steps"]]:
            assert c.pressure_anomaly.sum() == 0
            assert c.flow_anomaly.sum() == 0

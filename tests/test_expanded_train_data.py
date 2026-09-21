from wdn.expanded_train_data import (
    ALLOWED_CONFIG_CHANGES, distribution_changes, scenario_uid)


def test_expansion_distribution_audit_only_allows_identity_and_size_changes():
    base = {"profile": "base", "output_dir": "a", "seed": 811,
            "num_scenarios": 24, "missing_rate_pressure": .5,
            "attack_duration_hours": [6, 18]}
    candidate = {**base, "profile": "expanded", "output_dir": "b",
                 "seed": 1811, "num_scenarios": 16}
    assert distribution_changes(base, candidate) == ALLOWED_CONFIG_CHANGES
    changed = {**candidate, "attack_duration_hours": [4, 12]}
    assert "attack_duration_hours" in distribution_changes(base, changed)


def test_global_scenario_ids_include_seed_and_local_id():
    assert scenario_uid(1811, 7) == 1_811_007
    assert scenario_uid(1811, 7) != scenario_uid(2811, 7)

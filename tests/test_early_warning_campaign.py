"""Campaign-level invariants for `early_warning_multiseed_v1`.

These check the things that would quietly invalidate a reported number rather
than crash: split identities that overlap, a corpus generated under a different
distribution, a reported F1 that its own recorded counts do not reproduce, a
cache that silently accepts an incompatible configuration, or an L-Town corpus
pointing at Modena's reference.

Tests that need a campaign artifact skip when it is absent, so the suite stays
green on a checkout that has not run the campaign.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "runs/operational/early_warning_multiseed_v1"
EXPERIMENTS = ROOT / "thesis_v2/experiments/early_warning"

if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))


def load(path):
    if not Path(path).exists():
        pytest.skip(f"{Path(path).name} has not been produced yet")
    return json.loads(Path(path).read_text())


def f1_from_counts(tp, fp, fn):
    return 2 * tp / max(1, 2 * tp + fp + fn)


# --------------------------------------------------------------------------
# Seed and split identities
# --------------------------------------------------------------------------

def test_reserved_seeds_do_not_touch_locked_or_training_corpora():
    manifest = load(CAMPAIGN / "seed_manifest.json")
    reserved = {s for seeds in manifest["generator_seeds"].values() for s in seeds}
    forbidden = set(manifest["collision_check"]["forbidden_listed"])
    assert not reserved & forbidden
    assert not reserved & set(manifest["training_seeds"])
    assert len(reserved) == sum(len(v) for v in manifest["generator_seeds"].values())


def test_train_and_calibration_scenario_identities_are_disjoint():
    train = load(CAMPAIGN / "features/modena_train/manifest.json")
    calibration = load(CAMPAIGN / "features/modena_calibration/manifest.json")

    def identities(manifest):
        return {piece["seed"] * 1000 + sid
                for piece in manifest["pieces"] for sid in piece["scenarios"]}

    train_ids, calibration_ids = identities(train), identities(calibration)
    assert not train_ids & calibration_ids
    assert len(train_ids) == train["total_scenarios"]
    assert len(calibration_ids) == calibration["total_scenarios"]


def test_no_campaign_corpus_reuses_a_locked_split_scenario():
    """The original seed-811 test and validation scenarios never enter a corpus."""
    splits = json.loads(
        (ROOT / "runs/operational/blind_reference_probe_rank16/splits.json").read_text())
    protected = set(splits["test"]) | set(splits["validation"])
    for name in ("modena_train", "modena_calibration"):
        manifest = load(CAMPAIGN / f"features/{name}/manifest.json")
        for piece in manifest["pieces"]:
            if piece["seed"] != 811:
                continue
            assert not set(piece["scenarios"]) & protected, name


def test_every_campaign_corpus_fixes_both_missing_probabilities():
    configs = sorted((ROOT / "configs/early_warning").glob("*.yaml"))
    if not configs:
        pytest.skip("no campaign corpus has been generated yet")
    for path in configs:
        config = yaml.safe_load(path.read_text())
        assert config["missing_rate_pressure"] == 0.5, path.name
        assert config["missing_rate_flow"] == 0.5, path.name
        assert config["duration_hours"] == 168 and config["timestep_minutes"] == 60
        assert config["attack_fraction"] == 0.05
        assert config["pressure_noise_sigma_m"] == 0.1


def test_generator_config_check_rejects_a_changed_distribution():
    from build_feature_cache import check_distribution

    class FakeDirectory:
        name = "fake"

    benchmark = yaml.safe_load(
        (ROOT / "data/thesis_v2/operational_modena_seed811"
         / "generate_config.yaml").read_text())
    changed = dict(benchmark, attack_fraction=0.10)
    import build_feature_cache

    original = build_feature_cache.config_of
    try:
        build_feature_cache.config_of = lambda _: changed
        with pytest.raises(ValueError, match="distribution"):
            check_distribution(FakeDirectory(), benchmark)
        # A changed missing rate is caught by the distribution check first.
        build_feature_cache.config_of = lambda _: dict(benchmark, missing_rate_flow=0.3)
        with pytest.raises(ValueError, match="distribution"):
            check_distribution(FakeDirectory(), benchmark)
        # The explicit 0.50 guard is the backstop for a benchmark that itself
        # drifted, which the distribution comparison alone would not catch.
        wrong = dict(benchmark, missing_rate_flow=0.3)
        build_feature_cache.config_of = lambda _: wrong
        with pytest.raises(ValueError, match="missing probabilities"):
            check_distribution(FakeDirectory(), wrong)
    finally:
        build_feature_cache.config_of = original


# --------------------------------------------------------------------------
# Reference provenance
# --------------------------------------------------------------------------

def test_ltown_corpora_never_point_at_modenas_reference():
    import build_feature_cache
    import ltown_setup

    ltown_setup.register()
    modena = str(build_feature_cache.MODENA_REFERENCE)
    for name in ("ltown_train", "ltown_calibration"):
        spec = build_feature_cache.CORPORA[name]
        assert spec["network"] == "ltown"
        assert str(spec["reference"]) != modena
        assert "stage_e_ltown" in str(spec["reference"])


def test_ltown_reference_was_fitted_on_ltown_train():
    summary = load(CAMPAIGN / "stage_e_ltown/reference_summary.json")
    assert summary["network"] == "ltown"
    assert summary["sensors"] == 785
    assert set(summary["train_seeds"]) == {60811, 61811, 62811, 63811, 64811}
    assert sum(summary["group_sizes"]) == summary["sensors"]


def test_ltown_pilot_reference_is_blind_to_its_own_target_group():
    pilot = load(CAMPAIGN / "ltown_pilot_v1/pilot_report.json")
    assert pilot["blindness"]["own_prediction_is_blind"] is True
    assert pilot["blindness"]["max_change_in_own_prediction"] == 0.0
    assert pilot["blindness"]["max_change_in_same_group"] == 0.0
    # The probe must actually have perturbed something, or it proves nothing.
    assert pilot["blindness"]["max_change_in_other_groups"] > 0.0
    assert pilot["roster"]["sensors_below_reference_minimum"] == 0


# --------------------------------------------------------------------------
# Reported numbers reproduce from their own recorded counts
# --------------------------------------------------------------------------

def test_audit_branch_attribution_partitions_the_false_alarms():
    audit = load(CAMPAIGN / "stage_a_audit/false_alarm_audit.json")
    attribution = audit["surfaces"]["calibration"]["branch_attribution"]
    assert sum(attribution["combinations"].values()) == attribution["total_false_alarms"]
    for branch in ("general", "drift", "noise"):
        assert attribution[branch]["raises_alone"] <= attribution[branch]["raises"]


def test_audit_pooled_f1_reproduces_from_its_own_counts():
    audit = load(CAMPAIGN / "stage_a_audit/false_alarm_audit.json")
    for surface in audit["surfaces"].values():
        pooled = surface["pooled"]
        assert f1_from_counts(pooled["tp"], pooled["fp"], pooled["fn"]) == \
            pytest.approx(pooled["f1"], abs=1e-12)


def test_audit_reproduces_the_recorded_deployed_operating_point():
    audit = load(CAMPAIGN / "stage_a_audit/false_alarm_audit.json")
    frozen = load(ROOT / "runs/operational/feedback_router_veto_v1/selection_frozen.json")
    calibration = audit["surfaces"]["calibration"]
    assert calibration["pooled"]["f1"] == pytest.approx(
        frozen["selected"]["report"]["overall"], abs=1e-9)
    for family, value in frozen["selected"]["report"].items():
        if family in calibration["family_f1"]:
            assert calibration["family_f1"][family] == pytest.approx(value, abs=1e-9)


def test_screen_candidates_reproduce_their_f1_and_respect_the_gates():
    screen = load(CAMPAIGN / "verifier_screen_v1/selection_frozen.json")
    gates = screen["gates"]
    baseline = screen["baseline"]
    for name, entry in screen["candidates"].items():
        if not entry.get("passes_gates"):
            continue
        report = entry["report"]
        assert f1_from_counts(report["tp"], report["fp"], report["fn"]) == \
            pytest.approx(report["pooled_f1"], abs=1e-6), name
        assert report["clean_fpr"] <= gates["clean_fpr"], name
        for family, floor in gates["family_floors"].items():
            assert report[family] >= floor, (name, family)
            assert report[family] >= baseline[family] - gates["max_family_deterioration"], \
                (name, family)


def test_selected_candidate_is_the_best_passing_pooled_f1():
    screen = load(CAMPAIGN / "verifier_screen_v1/selection_frozen.json")
    passing = {n: e["report"]["pooled_f1"] for n, e in screen["candidates"].items()
               if e.get("passes_gates")}
    if not passing:
        assert screen["selected"] is None
        return
    assert screen["selected"] == max(passing, key=passing.get)


def test_warning_metrics_use_event_denominators_not_detected_events():
    trace = load(CAMPAIGN / "warning_to_confirmation_v1/warning_to_confirmation.json")
    rows = trace["per_event"]
    assert trace["events_traced"] == len(rows)
    assert trace["summary"]["events_with_a_warning"] <= len(rows)
    assert trace["summary"]["events_with_a_confirmation"] <= len(rows)
    lead = [r["warning_lead_over_confirmation_hours"] for r in rows
            if r["warning_lead_over_confirmation_hours"] is not None]
    if lead:
        assert trace["summary"]["median_warning_lead_hours"] == \
            pytest.approx(float(np.median(lead)))
    for row in rows:
        if row["first_confirmed_decision_hour"] is not None:
            assert row["confirmation_available_hour"] == \
                row["first_confirmed_decision_hour"] + trace["declared_decision_latency_hours"]


def test_early_head_reports_no_result_on_a_locked_corpus():
    for path, keys in (
        (CAMPAIGN / "early_head_v1/summary.json",
         ("calibration_evaluated", "eval_evaluated", "test_evaluated")),
        (CAMPAIGN / "verifier_screen_v1/selection_frozen.json",
         ("eval_evaluated", "test_evaluated")),
        (CAMPAIGN / "stage_a_audit/false_alarm_audit.json",
         ("locked_test_evaluated", "eval1_evaluated", "eval2_evaluated",
          "eval3_evaluated")),
    ):
        report = load(path)
        for key in keys:
            assert report[key] is False, (path.name, key)


# --------------------------------------------------------------------------
# Resume behaviour
# --------------------------------------------------------------------------

def test_feature_cache_refuses_an_incompatible_signature(tmp_path):
    import build_feature_cache

    name = "modena_calibration"
    if not (CAMPAIGN / f"features/{name}/signature.json").exists():
        pytest.skip("no cached corpus to compare against")
    target = tmp_path / name
    target.mkdir(parents=True)
    stored = json.loads((CAMPAIGN / f"features/{name}/signature.json").read_text())
    (target / "signature.json").write_text(
        json.dumps({**stored, "feature_count": stored["feature_count"] + 1}))
    with pytest.raises(SystemExit, match="incompatible"):
        build_feature_cache.build(name, output_root=tmp_path)


def test_campaign_status_never_claims_a_locked_corpus_was_read():
    status = CAMPAIGN / "stage_e_modena/status.json"
    if not status.exists():
        pytest.skip("the Stage E runner has not started")
    report = json.loads(status.read_text())
    assert report["locked_test_evaluated"] is False
    assert report["locked_eval_seeds_read"] is False

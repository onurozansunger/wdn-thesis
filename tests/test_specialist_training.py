import numpy as np
import torch

from wdn.audit_operational_experts import best_f1_threshold
from wdn.models.specialist_loss import direct_specialist_loss
from wdn.train_operational_moe import TrainingConfig, operational_loss


def _physics_fixture():
    from types import SimpleNamespace
    from wdn.models.physics_reference import HazenWilliamsReference

    graph = SimpleNamespace(node_types=np.array([0, 0, 0, 1]),
        node_elevations=np.array([5., 4., 3., 50.]),
        edge_index=np.array([[0, 0, 1, 2], [1, 2, 2, 3]]),
        edge_types=np.zeros(4), edge_lengths=np.full(4, 100.),
        edge_diameters=np.full(4, .3), edge_roughness=np.full(4, 130.))
    head = np.array([51., 50.8, 50.7, 50.])
    pressure = np.tile(head-graph.node_elevations, (4, 1))

    class Baseline:
        reference = SimpleNamespace(group_=np.array([0, 1, 1, 1]))
        noise_scale_ = np.full(4, .4)

        def predict_details(self, values, mask):
            return (np.tile(pressure[0]+.25, (len(values), 1)),
                    np.ones_like(values), np.zeros_like(values))

    model = HazenWilliamsReference(graph, Baseline())
    delta = head[graph.edge_index[0]]-head[graph.edge_index[1]]
    flow = np.tile(np.sign(delta)*(np.abs(delta)/model.resistance)**(1/1.852), (4, 1))
    return graph, model, pressure, flow


def test_physical_reference_blindness_units_and_causal_flow_direction():
    from wdn.models.physics_reference import HazenWilliamsReference

    graph, model, pressure, flow = _physics_fixture()
    pm, fm = np.ones_like(pressure, dtype=bool), np.ones_like(flow, dtype=bool)
    original = model.predict_details(pressure, pm, flow, fm)
    assert np.max(np.abs(original[0][:, 0]-pressure[:, 0])) < .04
    assert np.all(original[3]["physical_weight"][:, 0] > .8)
    changed_pressure = pressure.copy()
    changed_pressure[:, 0] += 1000.
    changed_mask = pm.copy()
    changed_mask[:, 0] = False
    changed = model.predict_details(changed_pressure, changed_mask, flow, fm)
    for key in original[3]:
        np.testing.assert_allclose(original[3][key][:, 0], changed[3][key][:, 0])
    np.testing.assert_allclose(original[0][:, 0], changed[0][:, 0])
    graph.edge_index = graph.edge_index[::-1].copy()
    reverse = HazenWilliamsReference(graph, model.baseline).predict_details(pressure, pm, -flow, fm)
    np.testing.assert_allclose(original[0], reverse[0], atol=1e-8)
    flow[-1] *= 3
    future = model.predict_details(pressure, pm, flow, fm)
    np.testing.assert_allclose(original[0][:-1], future[0][:-1])


def test_physical_reference_missing_disconnected_and_corrupt_flow_fallback():
    _, model, pressure, flow = _physics_fixture()
    pm, fm = np.ones_like(pressure, dtype=bool), np.ones_like(flow, dtype=bool)
    fm[:, 2] = False
    before = model.predict_details(pressure, pm, flow, fm)
    flow[~fm] = np.nan
    after = model.predict_details(pressure, pm, flow, fm)
    np.testing.assert_allclose(before[0], after[0])
    empty = model.predict_details(pressure, pm, flow, np.zeros_like(fm))
    np.testing.assert_allclose(empty[0], model.baseline.predict_details(pressure, pm)[0])
    assert not empty[3]["physical_weight"].any()
    flow[:] = 1.
    corrupt = model.predict_details(pressure, pm, flow, np.ones_like(fm))
    assert np.isfinite(corrupt[0]).all()
    assert np.all(corrupt[3]["physical_weight"][:, 0] == 0)


def test_marked_change_segment_integrals_match_independent_density():
    from scipy.integrate import quad
    from scipy.special import gammainc
    from scipy.stats import invgamma, multivariate_normal
    from wdn.marked_change import ChangeConfig, slope_log_evidence, variance_log_evidence

    cfg = ChangeConfig()
    z, age = np.array([.2, -.3, 2.]), np.array([1., 3., 4.])
    got, _, _ = slope_log_evidence(3, z.sum(), (z*z).sum(), age.sum(),
                                   (age*age).sum(), np.dot(age, z), cfg)
    design = np.column_stack([np.ones(3), age])
    covariance = np.eye(3)+design@np.diag([cfg.offset_sigma**2, cfg.slope_sigma**2])@design.T
    assert np.isclose(got, multivariate_normal.logpdf(z, cov=covariance))
    noise, _ = variance_log_evidence(3, z.sum(), (z*z).sum(), cfg)
    covariance = np.eye(3)+np.ones((3, 3))/cfg.mean_precision
    integral = quad(lambda variance: multivariate_normal.pdf(z, cov=variance*covariance)
        *invgamma.pdf(variance, a=cfg.variance_shape, scale=cfg.variance_scale), 1., np.inf)[0]
    integral /= gammainc(cfg.variance_shape, cfg.variance_scale)
    assert np.isclose(noise, np.log(integral), atol=1e-7)


def test_marked_change_is_causal_missing_safe_and_recovers_after_fault():
    from wdn.marked_change import MarkedChangeFilter

    rng = np.random.default_rng(847)
    values = rng.normal(scale=.3, size=(100, 2))
    values[30:50, 0] += np.arange(1, 21)*.4
    values[30:50, 1] += np.tile([-5., 5.], 10)
    mask = np.ones_like(values, dtype=bool)
    model = MarkedChangeFilter()
    full, names = model.transform(values, mask, np.zeros_like(values), np.ones(2), window=1)
    prefix, _ = model.transform(values[:45], mask[:45], np.zeros_like(values[:45]), np.ones(2), window=1)
    np.testing.assert_allclose(full[:45], prefix)
    assert full[49, 0, names.index("drift_run_probability")] > .9
    assert full[49, 1, names.index("noise_run_probability")] > .9
    assert np.max(full[-1, :, names.index("seq_run_change_probability")]) < .1
    mask[::3] = False
    before, _ = model.transform(values, mask, np.zeros_like(values), np.ones(2))
    values[~mask] = np.nan
    after, _ = model.transform(values, mask, np.zeros_like(values), np.ones(2))
    np.testing.assert_allclose(before, after)
    assert np.isfinite(after).all()


def test_marked_change_missing_emissions_follow_only_transition_prior():
    from wdn.marked_change import MarkedChangeFilter

    model = MarkedChangeFilter()
    values = np.full((12, 2), np.nan)
    out, names = model.transform(values, np.zeros_like(values, dtype=bool),
                                  np.zeros_like(values), np.ones(2), window=1)
    fault = 0.
    for t in range(len(values)):
        fault = fault*model.config.survival_probability+(1-fault)*model.config.entry_probability
        np.testing.assert_allclose(out[t, :, names.index("seq_run_change_probability")], fault, rtol=1e-6)


def test_recovery_screen_forbids_calibration_validation_test_extraction():
    import pytest
    from wdn.screen_recovery import TrainOnlyData

    store = TrainOnlyData.__new__(TrainOnlyData)
    store.allowed = {2, 5}
    # No backing data object: rejection must precede all archive access.
    for sid in (0, 3, 6, 100):
        with pytest.raises(ValueError, match="TRAIN"):
            store.scenario(sid)


def test_recovery_gate_rejects_more_false_alarms_or_failed_reference():
    from wdn.screen_recovery import gate

    control = {"by_scenario_ap": {str(k): .4 for k in range(4)}, "macro_ap": .4,
        "worst_ap": .4, "early_macro_recall": .2, "curve_clean_fpr": .001,
        "curve_post_event_fp": 3}
    candidate = {**control, "by_scenario_ap": {str(k): .5 for k in range(4)},
                 "macro_ap": .5, "worst_ap": .5, "early_macro_recall": .3}
    assert gate(candidate, control)["passed"]
    assert not gate(candidate, control, reference_pass=False)["passed"]
    assert not gate({**candidate, "curve_clean_fpr": .002}, control)["passed"]
    assert not gate({**candidate, "curve_post_event_fp": 4}, control)["passed"]


def test_direct_expert_training_ignores_router_and_missing_labels():
    logits = torch.zeros(12, 6, requires_grad=True)
    family = torch.arange(6)
    labels = torch.tensor([0., 0.]+[1., 0.]*5)
    mask = torch.ones(12)
    mask[-1] = 0.
    output = {"expert_pressure_anomaly_logits": logits,
              "expert_pressure_pred": torch.zeros(12, 6, requires_grad=True)}
    batch = {"attack_type": family, "num_nodes": 2, "pressure_mask": mask,
             "pressure_anomaly": labels, "y_pressure": torch.zeros(12)}
    loss = direct_specialist_loss(output, batch)["anomaly"]
    loss.backward()
    for expert in range(1, 6):
        assert logits.grad[expert*2, expert] < 0
    assert torch.all(logits.grad[-1] == 0)
    # Even a catastrophically wrong router cannot starve the owner expert.
    output["router_logits"] = torch.tensor([[100., 0., 0., 0., 0., 0.]]*6)
    assert torch.allclose(loss, direct_specialist_loss(output, batch)["anomaly"])


def test_warmup_has_expert_gradients_but_no_router_gradient():
    logits = torch.zeros(12, 6, requires_grad=True)
    router = torch.randn(6, 6, requires_grad=True)
    output = {"expert_pressure_anomaly_logits": logits,
              "expert_pressure_pred": torch.zeros(12, 6, requires_grad=True),
              "router_logits": router}
    batch = {"attack_type": torch.arange(6), "num_nodes": 2,
             "pressure_mask": torch.ones(12),
             "pressure_anomaly": torch.tensor([0., 0.]+[1., 0.]*5),
             "y_pressure": torch.zeros(12)}
    cfg = TrainingConfig(expert_objective="balanced", expert_warmup_epochs=2)
    operational_loss(output, batch, cfg, expert_only=True).backward()
    assert router.grad is None
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_diagnostic_threshold_uses_strict_greater_than_for_float32():
    scores = np.array([.1, .2, .3, .4], dtype=np.float32)
    labels = np.array([0., 0., 1., 1.])
    threshold = best_f1_threshold(scores, labels)
    assert np.array_equal(scores > threshold, labels > .5)


def test_legacy_objective_still_trains_specialists_after_balanced_extension():
    logits = torch.zeros(12, 6, requires_grad=True)
    output = {"expert_pressure_anomaly_logits": logits,
              "expert_pressure_pred": torch.zeros(12, 6, requires_grad=True),
              "pressure_anomaly_logits": torch.zeros(12, requires_grad=True),
              "pressure_pred": torch.zeros(12, requires_grad=True),
              "flow_pred": torch.zeros(6, requires_grad=True),
              "router_logits": torch.zeros(6, 6, requires_grad=True)}
    batch = {"attack_type": torch.arange(6), "num_nodes": 2,
             "pressure_mask": torch.ones(12), "flow_mask": torch.ones(6),
             "pressure_anomaly": torch.tensor([0., 0.]+[1., 0.]*5),
             "y_pressure": torch.zeros(12), "y_flow": torch.zeros(6)}
    loss = operational_loss(output, batch, TrainingConfig(expert_objective="legacy"))
    loss.backward()
    assert torch.isfinite(loss)
    for expert in range(1, 6):
        assert logits.grad[expert*2, expert] < 0


def test_optuna_enqueue_preserves_recipe_and_trial_budget_on_resume():
    import optuna
    from wdn.tune_operational_moe import base_parameters, suggest_config, remaining_trials

    cfg = TrainingConfig(hidden_dim=16, replay_hidden_dim=24, drift_hidden_dim=16,
                         router_hidden_dim=16, num_layers=1, batch_size=16,
                         feature_version=2, expert_objective="balanced",
                         expert_warmup_epochs=2, lambda_expert=1., epochs=24)
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial(base_parameters(cfg))
    assert remaining_trials(study, 4) == 4
    trial = study.ask()
    candidate = suggest_config(trial, cfg, local=True)
    # Real sampler consumption must reproduce the complete recipe, including
    # the previously unsupported 24-wide replay expert and fixed split seed.
    assert candidate == cfg
    study.tell(trial, .4)
    assert remaining_trials(study, 4) == 3
    failed = study.ask()
    study.tell(failed, state=optuna.trial.TrialState.FAIL)
    assert remaining_trials(study, 4) == 2
    assert remaining_trials(study, 1) == 0


def test_blind_reference_excludes_target_group_and_missing_placeholders():
    from wdn.models.blind_reference import BlindPressureReference

    rng = np.random.default_rng(16)
    latent = rng.normal(size=(80, 2))
    values = latent@rng.normal(size=(2, 12)) + rng.normal(scale=.05, size=(80, 12))
    mask = rng.random(values.shape) > .5
    reference = BlindPressureReference(rank=2, groups=4, iterations=4).fit(values, mask)
    observed = np.ones((5, 12), dtype=bool)
    probe = values[:5].copy()
    prediction, support = reference.predict(probe, observed)
    target = reference.group_ == 0
    probe[:, target] += 1000.
    changed, _ = reference.predict(probe, observed)
    # Robust weights must not indirectly reintroduce the held-out readings.
    assert np.allclose(prediction[:, target], changed[:, target], atol=1e-10)
    observed[:, 1] = False
    before, _ = reference.predict(probe, observed)
    probe[:, 1] = np.nan
    after, _ = reference.predict(probe, observed)
    assert np.allclose(before, after)
    empty, support = reference.predict(probe, np.zeros_like(observed))
    assert np.isfinite(empty).all() and not support.any()


def test_blind_residual_features_are_causal_and_ignore_missing_values():
    from wdn.probe_blind_reference import residual_features

    rng = np.random.default_rng(6)
    values = rng.normal(size=(24, 3))
    observed = rng.random(values.shape) > .5
    predicted = np.zeros_like(values)
    support = np.ones_like(values)*.5
    full, names = residual_features(values, observed, predicted, support, np.ones(3))
    prefix, _ = residual_features(values[:20], observed[:20], predicted[:20], support[:20], np.ones(3))
    assert np.allclose(full[:len(prefix)], prefix)
    values[~observed] = np.nan
    changed, _ = residual_features(values, observed, predicted, support, np.ones(3))
    assert np.isfinite(changed).all() and np.allclose(full, changed)
    assert changed.shape[-1] == len(names)


def test_dynamic_drift_features_are_causal_and_accumulate_weak_changes():
    from wdn.dynamic_residual_features import dynamic_residual_features

    values = np.full((60, 3), 5.)
    mask = np.ones_like(values, dtype=bool)
    predicted = np.zeros_like(values)
    normal, names = dynamic_residual_features(values, mask, predicted, np.ones(3))
    values[30:, 0] += np.arange(30)*.1
    drift, _ = dynamic_residual_features(values, mask, predicted, np.ones(3))
    assert drift[-1, 0, names.index("dynamic_cusum_positive")] > normal[-1, 0, names.index("dynamic_cusum_positive")]+5
    prefix, _ = dynamic_residual_features(values[:45], mask[:45], predicted[:45], np.ones(3))
    assert np.allclose(prefix, drift[:len(prefix)])
    mask[::2, 1] = False
    before, _ = dynamic_residual_features(values, mask, predicted, np.ones(3))
    values[~mask] = np.nan
    after, _ = dynamic_residual_features(values, mask, predicted, np.ones(3))
    assert np.isfinite(after).all() and np.allclose(before, after)


def test_expert_tail_evidence_is_monotone_conservative_at_ties_and_frozen():
    import pytest
    from wdn.probe_expert_calibration import NormalTailEvidence

    normal = np.array([[.1, .2], [.2, .2], [.2, .4], [.5, .6]])
    fitted = NormalTailEvidence().fit(normal)
    before = fitted.normal_.copy()
    scores = np.array([[0., .1], [.2, .2], [.3, .5], [.9, .9]])
    evidence = fitted.transform(scores)
    # Three normal values >= .2, plus a pseudocount: tail = 4/5.
    assert evidence[1, 0] == pytest.approx(-np.log(4/5))
    assert np.all(np.diff(evidence, axis=0) >= 0)
    assert np.isfinite(evidence).all()
    assert evidence[-1, 0] == pytest.approx(np.log(5))
    assert np.array_equal(fitted.normal_, before)
    with pytest.raises(ValueError):
        fitted.transform(np.array([[np.nan, 0.]]))
    with pytest.raises(ValueError):
        fitted.transform(np.ones((3, 1)))


def test_robust_reference_preserves_blindness_and_missing_invariance():
    from wdn.models.blind_reference import BlindPressureReference
    from wdn.models.robust_reference import RobustBlindReference

    rng = np.random.default_rng(821)
    values = rng.normal(size=(80, 2))@rng.normal(size=(2, 16))
    values += rng.normal(scale=.1, size=values.shape)
    observed = rng.random(values.shape) > .5
    base = BlindPressureReference(rank=2, iterations=3).fit(values, observed)
    robust = RobustBlindReference(base, views=3, steps=4).calibrate_scale(values, observed)
    probe, mask = values[:5].copy(), observed[:5].copy()
    initial = robust.predict_details(probe, mask)
    group = base.group_ == 0
    probe[:, group] += 1000
    changed = robust.predict_details(probe, mask)
    for left, right in zip(initial, changed):
        np.testing.assert_allclose(left[:, group], right[:, group], atol=1e-9)
    before = robust.predict_details(probe, mask)
    probe[~mask] = np.nan
    for left, right in zip(before, robust.predict_details(probe, mask)):
        np.testing.assert_allclose(left, right)
    empty = robust.predict_details(probe, np.zeros_like(mask))
    assert all(np.isfinite(item).all() for item in empty)
    assert not empty[1].any()


def test_sequential_evidence_is_causal_and_distinguishes_persistent_variance():
    from wdn.sequential_evidence import sequential_evidence

    values = np.zeros((50, 2))
    mask = np.ones_like(values, dtype=bool)
    values[30:, 0] = np.arange(20)*.2
    values[30:, 1] = np.tile([-4., 4.], 10)
    predicted = np.zeros_like(values)
    features, names = sequential_evidence(values, mask, predicted, np.ones(2))
    prefix, _ = sequential_evidence(values[:40], mask[:40], predicted[:40], np.ones(2))
    np.testing.assert_allclose(prefix, features[:len(prefix)])
    assert features[-1, 0, names.index("drift_cusum_positive_0.5")] > 5
    assert features[-1, 1, names.index("noise_state_9.0")] > .9
    mask[::3] = False
    before, _ = sequential_evidence(values, mask, predicted, np.ones(2))
    values[~mask] = np.nan
    after, _ = sequential_evidence(values, mask, predicted, np.ones(2))
    np.testing.assert_allclose(before, after)
    assert np.isfinite(after).all() and after.shape[-1] == len(names)


def test_hybrid_training_weights_balance_events_and_normal_scenarios():
    from wdn.models.residual_hybrid import training_weights

    labels = np.array([1, 1, 1, 1, 0, 0, 0])
    families = np.array([3, 3, 3, 4, 0, 0, 0])
    events = np.array([1, 1, 2, 3, -1, -1, -1])
    scenarios = np.array([1, 1, 2, 3, 1, 1, 2])
    w = training_weights(labels, families, events, scenarios, np.arange(7))
    assert np.isclose(w[events == 1].sum(), w[events == 2].sum())
    assert np.isclose(w[families == 3].sum(), w[families == 4].sum())
    assert np.isclose(w[labels == 1].sum(), w[labels == 0].sum())


def test_redesign_inner_folds_keep_scenarios_whole_and_all_families():
    from wdn.run_expert_redesign import group_folds

    families = ("random", "replay", "stealthy", "noise", "targeted")
    events = [{"scenario_id": i, "family": families[i % 5]} for i in range(15)]
    folds = group_folds(list(range(15)), events)
    assert sorted(sid for fold in folds for sid in fold) == list(range(15))
    for fold in folds:
        assert {e["family"] for e in events if e["scenario_id"] in fold} == set(families)


def test_joint_context_excludes_target_group_and_ignores_future_missing_values():
    from wdn.models.blind_reference import BlindPressureReference
    from wdn.models.robust_reference import RobustBlindReference
    from wdn.weak_family_features import conditional_features

    rng = np.random.default_rng(831)
    values = rng.normal(size=(60, 2))@rng.normal(size=(2, 16))
    values += rng.normal(scale=.1, size=values.shape)
    mask = rng.random(values.shape) > .5
    base = BlindPressureReference(rank=2, iterations=3).fit(values, mask)
    reference = RobustBlindReference(base, views=2, steps=3).calibrate_scale(values, mask)
    original = conditional_features(reference, values[:24], mask[:24], 8)
    ordinary = reference.predict_details(values[:24], mask[:24])
    # Context extraction must not accidentally change the ordinary reference.
    for extra_path, ordinary_path in zip(original[:3], ordinary):
        np.testing.assert_allclose(extra_path, ordinary_path[:, :8], atol=1e-8)
    changed = values[:24].copy()
    changed[:, base.group_ == 0] += 1000.
    modified = conditional_features(reference, changed, mask[:24], 8)
    target = base.group_[:8] == 0
    for before, after in zip(original[:4], modified[:4]):
        np.testing.assert_allclose(before[:, target], after[:, target], atol=1e-8)
    changed = values[:24].copy()
    changed[~mask[:24]] = np.nan
    modified = conditional_features(reference, changed, mask[:24], 8)
    for before, after in zip(original[:4], modified[:4]):
        np.testing.assert_allclose(before, after)
    prefix = conditional_features(reference, values[:20], mask[:20], 8)
    for before, after in zip(original[:4], prefix[:4]):
        np.testing.assert_allclose(before[:20], after, atol=1e-7)
    assert original[3].shape[-1] == len(original[4]) == 30


def test_expert_memory_is_causal_resets_scenarios_and_decays_missing_hours():
    import pytest
    from wdn.models.weak_family import score_memory

    scores = np.array([[.9], [.1], [.1], [.1]])
    sid, time, node = np.array([0, 0, 0, 1]), np.array([1, 2, 12, 1]), np.zeros(4, dtype=int)
    result = score_memory(scores, sid, time, node)
    prefix = score_memory(scores[:2], sid[:2], time[:2], node[:2])
    np.testing.assert_allclose(result[:2], prefix)
    assert result[1, 1] > result[2, 1] > result[3, 1]
    np.testing.assert_allclose(result[3], [.1, .1, .1])
    with pytest.raises(ValueError):
        score_memory(scores[:2], sid[:2], np.ones(2, dtype=int), node[:2])


def test_early_weighting_preserves_event_contributions():
    from wdn.models.weak_family import phase_weights

    a = {"labels": np.array([1, 1, 1, 1, 0, 0]), "families": np.array([3, 3, 3, 3, 0, 0]),
         "event": np.array([1, 1, 2, 2, -1, -1]), "scenario": np.array([1, 1, 2, 2, 1, 2]),
         "early": np.array([True, False, True, False, False, False])}
    weights = phase_weights(a, np.arange(6), 3.)
    assert np.isclose(weights[0], 3*weights[1])
    assert np.isclose(weights[:2].sum(), weights[2:4].sum())
    assert np.isclose(weights[:4].sum(), weights[4:].sum())


def test_monotone_fusion_cannot_invert_expert_evidence():
    from wdn.models.weak_family import MonotoneFusion

    labels = np.tile([0., 1.], 30)
    a = {"X": np.ones((60, 1)), "labels": labels, "families": labels.astype(int)*3,
         "event": np.where(labels > 0, 1, -1), "scenario": np.zeros(60, dtype=int),
         "timestep": np.arange(60), "node": np.zeros(60, dtype=int)}
    scores = np.tile((.1+.7*labels)[:, None], (1, 5))
    model = MonotoneFusion(["reference_support"]).fit(scores, a)
    before = model.predict(scores, a)
    scores[-1, 3] = .99
    after = model.predict(scores, a)
    assert after[-1] >= before[-1]
    np.testing.assert_allclose(after[:-1], before[:-1])
    assert (model.coef_[:model.monotone_count_] >= 0).all()


def test_normal_context_is_causal_and_target_group_blind():
    from wdn.models.blind_reference import BlindPressureReference
    from wdn.models.robust_reference import RobustBlindReference
    from wdn.normal_context import normal_context_features

    rng = np.random.default_rng(841)
    latent = rng.normal(size=(90, 3))
    values = latent@rng.normal(size=(3, 16))+rng.normal(scale=.08, size=(90, 16))
    mask = rng.random(values.shape) > .4
    base = BlindPressureReference(rank=3, iterations=3).fit(values, mask)
    reference = RobustBlindReference(base, views=2, steps=3).calibrate_scale(values, mask)
    prediction, support, spread = reference.predict_details(values[:30], mask[:30])
    context, names = normal_context_features(
        prediction, support, spread, reference.noise_scale_, np.arange(30))
    prefix = reference.predict_details(values[:24], mask[:24])
    prefix_context, prefix_names = normal_context_features(
        *prefix, reference.noise_scale_, np.arange(24))
    np.testing.assert_allclose(context[:24], prefix_context)
    assert names == prefix_names and context.shape[-1] == len(names)

    target = base.group_ == 0
    changed, changed_mask = values[:30].copy(), mask[:30].copy()
    changed[:, target] += 1000
    changed_mask[:, target] = ~changed_mask[:, target]
    modified = reference.predict_details(changed, changed_mask)
    modified_context, _ = normal_context_features(
        *modified, reference.noise_scale_, np.arange(30))
    np.testing.assert_allclose(context[:, target], modified_context[:, target], atol=1e-8)


def test_conditional_normal_is_pooled_safe_and_joblib_stable(tmp_path):
    import joblib
    import pytest
    from wdn.models.conditional_normal import ConditionalNormalConfig, ConditionalNormalError
    from wdn.normal_context import NAMES

    rng = np.random.default_rng(842)
    n = 900
    X = rng.normal(size=(n, len(NAMES)))
    node = np.tile(np.arange(9), 100)
    scenario = np.repeat(np.arange(6), 150)
    residual = .20*X[:, NAMES.index("normal_reference_level_m")]
    residual += .03*(node-4)+rng.normal(scale=.12, size=n)
    config = ConditionalNormalConfig(minimum_scale=.1, conditional_scale=False)
    model = ConditionalNormalError(NAMES, config).fit(
        X, residual, node=node, scenario=scenario)
    mean, scale = model.predict(X, node=node)
    assert np.mean(np.abs(residual-mean)) < np.mean(np.abs(residual))
    assert np.isfinite(mean).all() and np.all(np.isfinite(scale) & (scale >= .1))
    assert np.max(np.abs(mean)) <= model.mean_clip_m_
    scale_only = ConditionalNormalError(
        NAMES, ConditionalNormalConfig(minimum_scale=.1, fit_mean=False,
                                       conditional_scale=False)).fit(
            X, residual, node=node, scenario=scenario)
    zero, scale_control = scale_only.predict(X, node=node)
    np.testing.assert_array_equal(zero, np.zeros(n))
    assert np.all(scale_control >= .1)
    path = tmp_path/"normal.joblib"
    joblib.dump(model, path)
    for before, after in zip(model.predict(X[:50], node=node[:50]),
                             joblib.load(path).predict(X[:50], node=node[:50])):
        np.testing.assert_array_equal(before, after)
    with pytest.raises(ValueError, match="Unreviewed"):
        ConditionalNormalError(["residual"], ConditionalNormalConfig(covariates=("residual",)))

    from wdn.screen_normal_nuisance import TrainOnlyCampaign
    campaign = TrainOnlyCampaign.__new__(TrainOnlyCampaign)
    campaign.allowed, campaign.accessed = {1}, set()
    with pytest.raises(ValueError, match="TRAIN scenarios only"):
        campaign.scenario(2)
    assert campaign.accessed == set()


def test_conditional_change_filter_is_causal_and_mechanism_specific():
    from wdn.conditional_change import ConditionalChangeFilter

    T, N = 70, 3
    values = np.zeros((T, N))
    values[35:, 0] = np.arange(1, T-34)*.10
    values[35:, 1] = np.tile([-.8, .8], (T-35+1)//2)[:T-35]
    observed = np.ones_like(values, dtype=bool)
    zeros = np.zeros_like(values)
    scale = np.full_like(values, .20)
    prior_scale = np.full(N, .20)
    model = ConditionalChangeFilter()
    full, names = model.transform(values, observed, zeros, zeros, scale, prior_scale)
    prefix, prefix_names = model.transform(
        values[:55], observed[:55], zeros[:55], zeros[:55], scale[:55], prior_scale)
    np.testing.assert_allclose(full[:len(prefix)], prefix, atol=1e-7)
    assert names == prefix_names and full.shape[-1] == len(names)
    assert full[-1, 0, names.index("conditional_drift_probability")] > .8
    assert full[-1, 1, names.index("conditional_noise_probability")] > .8
    assert full[-1, 2, names.index("conditional_change_probability")] < .1
    missing = observed.copy()
    missing[::3] = False
    placeholders = values.copy()
    placeholders[~missing] = np.nan
    changed, _ = model.transform(placeholders, missing, zeros, zeros, scale, prior_scale)
    assert np.isfinite(changed).all()


def test_latent_ar_normal_error_is_cross_scenario_causal_and_serialisable(tmp_path):
    import joblib
    from wdn.models.latent_ar import LatentARNormalError

    rng = np.random.default_rng(41)
    phi, innovation_std = .72, .11
    sequences = []
    for scenario in range(4):
        values = np.zeros((180, 3))
        values[0] = rng.normal(0, innovation_std/np.sqrt(1-phi**2), 3)
        for t in range(1, len(values)):
            values[t] = phi*values[t-1]+rng.normal(0, innovation_std, 3)
        sequences.append(values)
    residual = np.concatenate([value.ravel() for value in sequences])
    scenario = np.concatenate([np.full(value.size, sid) for sid, value in enumerate(sequences)])
    timestep = np.concatenate([np.repeat(np.arange(len(value)), value.shape[1]) for value in sequences])
    node = np.tile(np.arange(3), sum(len(value) for value in sequences))

    model = LatentARNormalError().fit(
        residual, scenario=scenario, timestep=timestep, node=node)
    assert .62 < model.phi_ < .82
    observed = np.ones_like(sequences[0], dtype=bool)
    mean, scale = model.predict_sequence(sequences[0], observed)
    raw_r = np.corrcoef(sequences[0][:-1].ravel(), sequences[0][1:].ravel())[0, 1]
    innovation = sequences[0]-mean
    innovation_r = np.corrcoef(innovation[:-1].ravel(), innovation[1:].ravel())[0, 1]
    assert abs(innovation_r) < abs(raw_r)-.4
    assert np.all(scale > 0)

    changed = sequences[0].copy()
    changed[120:] += 100
    changed_mean, _ = model.predict_sequence(changed, observed)
    np.testing.assert_array_equal(mean[:120], changed_mean[:120])
    missing = observed.copy()
    missing[50:53, 0] = False
    placeholders = sequences[0].copy()
    placeholders[~missing] = np.nan
    missing_mean, missing_scale = model.predict_sequence(placeholders, missing)
    assert np.isfinite(missing_mean).all() and np.isfinite(missing_scale).all()
    assert missing_scale[53, 0] > scale[53, 0]

    path = tmp_path/"latent_ar.joblib"
    joblib.dump(model, path)
    replay_mean, replay_scale = joblib.load(path).predict_sequence(sequences[0], observed)
    np.testing.assert_array_equal(mean, replay_mean)
    np.testing.assert_array_equal(scale, replay_scale)


def test_family_state_features_are_causal_and_gap_aware():
    from wdn.family_state_features import family_state_features, feature_names

    names = ["dynamic_innovation", "unrelated"]
    arrays = {"X": np.array([[.2, 1.], [1.2, 2.], [-.4, 3.], [2.1, 4.], [.5, 5.]]),
        "scenario": np.array([1, 1, 1, 1, 2]),
        "timestep": np.array([10, 11, 14, 15, 4]),
        "node": np.array([0, 0, 0, 0, 0])}
    full, found = family_state_features(arrays, names)
    assert found == feature_names() and full.shape == (5, len(found))
    prefix = {key: value[:3] for key, value in arrays.items()}
    prefix_result, _ = family_state_features(prefix, names)
    np.testing.assert_array_equal(full[:3], prefix_result)
    altered = {key: value.copy() for key, value in arrays.items()}
    altered["X"][3:, 0] += 100
    altered_result, _ = family_state_features(altered, names)
    np.testing.assert_array_equal(full[:3], altered_result[:3])
    # The three-hour observation gap decays a peak instead of treating missing
    # hours as zero-valued observations.
    peak = found.index("state_peak_abs_4")
    assert full[2, peak] < full[1, peak] and full[2, peak] >= abs(arrays["X"][2, 0])
    # A scenario boundary resets all state.
    signed = found.index("state_signed_ewm_2")
    assert full[4, signed] == arrays["X"][4, 0]


def test_family_specific_experts_use_hard_negative_mass_and_monotone_stack():
    from wdn.models.family_specific import (_balanced_weights, FamilySpecificExpert,
                                             MonotoneFamilyStacker)

    rng = np.random.default_rng(52)
    names = ["residual", "abs_residual", "dynamic_innovation", "dynamic_abs_innovation",
        "state_signed_ewm_2", "state_peak_abs_4", "mean_4", "drift_ramp_3",
        "state_mean_strength_3", "std_4", "noise_energy_3", "noise_state_2.0",
        "state_energy_3", "state_support_3"]
    n = 720
    families = np.tile(np.repeat([0, 1, 3, 4], 60), 3)
    scenario = np.repeat(np.arange(3), 240)
    event = np.where(families == 0, -1, scenario*10+families)
    labels = np.zeros(n)
    labels[(families == 3) & (np.arange(n) % 4 == 0)] = 1
    labels[(families == 4) & (np.arange(n) % 3 == 0)] = 1
    early = (np.arange(n) % 5) == 0
    X = rng.normal(size=(n, len(names)))
    X[:, names.index("dynamic_abs_innovation")] = np.abs(X[:, names.index("dynamic_innovation")])
    arrays = {"X": X, "labels": labels, "families": families, "event": event,
              "scenario": scenario, "early": early}
    selected, weights = _balanced_weights(arrays, 3, 4.0)
    mass = weights/weights.sum()
    positive = (families[selected] == 3) & (labels[selected] > 0)
    hard = (families[selected] == 3) & (labels[selected] == 0)
    clean = families[selected] == 0
    other = (families[selected] != 0) & (families[selected] != 3)
    np.testing.assert_allclose([mass[positive].sum(), mass[hard].sum(),
                                mass[clean].sum(), mass[other].sum()], [.45, .30, .15, .10])

    drift = FamilySpecificExpert(names, "drift").fit(arrays)
    scores = drift.predict_heads(X)
    assert scores.shape == (n, 2) and np.all((scores >= 0) & (scores <= 1))
    stacker = MonotoneFamilyStacker("drift").fit(scores, arrays)
    combined = stacker.predict(scores)
    assert combined.shape == (n,) and np.all((combined >= 0) & (combined <= 1))
    assert np.all(stacker.coef_ >= 0)


def test_weak_campaign_rejects_test_and_premature_validation(tmp_path):
    import pytest
    from wdn.train_weak_families import FeatureStore

    store = FeatureStore.__new__(FeatureStore)
    store.output = tmp_path
    store.allowed = {1, 2, 3}
    store.splits = {"test": [3]}
    # These checks must happen before touching a dataset/reference on disk.
    with pytest.raises(ValueError, match="before selection"):
        store.arrays("local", "full", "validation")
    with pytest.raises(ValueError, match="Test/unknown"):
        store.observations(3, joint=True)


def test_incident_window_aggregation_is_grouped_and_retrospective():
    from wdn.models.incident_window import aggregate_incident_scores

    scores = np.array([.1, .4, .9, .2, .8, .7])
    incident = np.array([3, 3, 3, 3, 3, 8])
    node = np.array([1, 1, 1, 2, 2, 1])
    timestep = np.array([7, 8, 9, 7, 9, 2])
    selected = np.array([True, True, True, True, True, False])
    mean = aggregate_incident_scores(scores, incident, node, timestep, selected, "mean")
    top2 = aggregate_incident_scores(scores, incident, node, timestep, selected, "top2")
    np.testing.assert_allclose(mean[:3], np.mean(scores[:3]))
    np.testing.assert_allclose(top2[:3], np.mean([.9, .4]))
    np.testing.assert_allclose(mean[3:5], .5)
    assert mean[5] == top2[5] == 0

    logit = aggregate_incident_scores(scores, incident, node, timestep, selected, "logit")
    expected = np.log(scores[:3]/(1-scores[:3])).sum()/np.sqrt(3)
    np.testing.assert_allclose(logit[:3], expected)


def test_incident_window_aggregation_rejects_selected_rows_without_window():
    import pytest
    from wdn.models.incident_window import aggregate_incident_scores

    with pytest.raises(ValueError, match="window id"):
        aggregate_incident_scores(np.array([.5]), np.array([-1]), np.array([0]),
                                   np.array([1]), np.array([True]), "mean")


def test_budgeted_noise_rank_fusion_and_top_k_do_not_use_labels():
    from wdn.models.budgeted_incident import budgeted_noise_rank_score, top_k_sensor_decision

    names = ["dynamic_innovation", "dynamic_sigma"]
    arrays = {"X": np.array([[0., 1.], [4., 1.], [0., 1.], [.2, 1.], [.1, 1.], [.2, 1.]]),
        "labels": np.array([1, 1, 0, 0, 0, 0]), "families": np.full(6, 4),
        "event": np.full(6, 7), "node": np.array([0, 0, 1, 1, 2, 2]),
        "timestep": np.array([10, 11, 10, 11, 10, 11])}
    learned = np.array([.2, .2, .8, .8, .5, .5])
    score, rows, groups = budgeted_noise_rank_score(arrays, learned, names)
    assert score.shape == (6,) and np.all((score > 0) & (score < 1))
    decision = top_k_sensor_decision(rows, groups, 1)
    assert decision.sum() == 2
    altered = {**arrays, "labels": 1-arrays["labels"]}
    changed, changed_rows, changed_groups = budgeted_noise_rank_score(altered, learned, names)
    np.testing.assert_array_equal(score, changed)
    np.testing.assert_array_equal(rows, changed_rows)
    np.testing.assert_array_equal(decision, top_k_sensor_decision(changed_rows, changed_groups, 1))

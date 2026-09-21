import torch

from wdn.metrics import compute_anomaly_metrics


def test_anomaly_metrics_include_auprc_from_continuous_scores():
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
    scores = torch.tensor([0.1, 0.4, 0.35, 0.8], dtype=torch.float32)
    predictions = (scores >= 0.5).float()

    metrics = compute_anomaly_metrics(predictions, labels, scores=scores)

    assert abs(metrics.auroc - 0.75) < 1e-8
    assert abs(metrics.auprc - (5.0 / 6.0)) < 1e-7


def test_auprc_edge_cases_are_finite():
    no_positives = compute_anomaly_metrics(
        torch.zeros(3), torch.zeros(3), scores=torch.tensor([0.1, 0.2, 0.3])
    )
    all_positives = compute_anomaly_metrics(
        torch.ones(3), torch.ones(3), scores=torch.tensor([0.1, 0.2, 0.3])
    )

    assert no_positives.auprc == 0.0
    assert all_positives.auprc == 1.0


def test_operational_target_score_and_single_global_threshold():
    import numpy as np
    from wdn.train_operational_moe import target_score, select_threshold, summarise

    assert target_score(.9, .2) < target_score(.8, .51)
    assert target_score(.8, .51) == .8
    scores = np.array([.8, .7, .1, .2, .6, .55, .15, .25])
    labels = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    families = np.array([1]*4+[2]*4)
    threshold = select_threshold(scores, labels, families)
    report = summarise(scores, labels, families, threshold)
    assert report["overall"]["f1"] == 1.
    assert report["per_family"]["replay"]["f1"] == 1.
    assert report["replay_target_met"]


def test_operational_splits_are_disjoint_and_optuna_cannot_request_test(tmp_path):
    from types import SimpleNamespace
    import pytest
    from wdn.train_operational_moe import make_splits, train, TrainingConfig

    snaps, corrupted = [], []
    for sid in range(16):
        for family in range(6):
            snaps.append(SimpleNamespace(scenario_id=sid))
            corrupted.append(SimpleNamespace(attack_type_id=family, pressure_anomaly=torch.tensor([1.])))
    splits = make_splits(snaps, corrupted, 417)
    assert splits == make_splits(snaps, corrupted, 417)
    flattened = [sid for values in splits.values() for sid in values]
    assert len(flattened) == len(set(flattened)) == 16
    with pytest.raises(ValueError, match="never evaluate test"):
        train(TrainingConfig(output_dir=str(tmp_path/"blocked"), evaluate_test=True), trial=object())
    assert not (tmp_path/"blocked").exists()


def test_exact_family_calibration_matches_brute_force_with_tied_float32_scores():
    import numpy as np
    from wdn.operational_calibration import calibrate_threshold, family_summary
    from wdn.train_operational_moe import summarise

    scores = np.array([.3, .2, .1, .1, .9, .6, .1, .8, .7, .2,
                       .7, .5, .1, .6, .4, .1, .6, .3, .2], dtype=np.float32)
    families = np.array([0]*4+[1]*3+[2]*3+[3]*3+[4]*3+[5]*3)
    labels = np.array([0]*4+[1, 1, 0]*5)
    for objective in ("macro_f1", "worst_f1", "legacy"):
        point = calibrate_threshold(scores, labels, families, objective=objective, max_fpr=.5)
        candidates = []
        for s in np.unique(scores):
            threshold = float(np.nextafter(s, np.float32(-np.inf)))
            r = summarise(scores, labels, families, threshold)
            f = family_summary(r)
            if max(r["overall"]["fpr"], r["per_family"]["clean"]["fpr"]) > .5 or r["per_family"]["replay"]["f1"] < .5:
                continue
            score = r["target_score"] if objective == "legacy" else f["macro_f1" if objective == "macro_f1" else "worst_family_f1"]
            candidates.append((score, f["macro_f1"], r["overall"]["f1"], threshold))
        assert point["threshold"] == max(candidates)[-1]
        checked = summarise(scores, labels, families, point["threshold"])
        assert point["calibration"]["fpr"] == checked["overall"]["fpr"]


def test_family_calibration_does_not_silently_relax_infeasible_alarm_budget():
    import numpy as np
    import pytest
    from wdn.operational_calibration import calibrate_threshold

    with pytest.raises(ValueError, match="No feasible"):
        calibrate_threshold(np.ones(6), np.array([0, 1, 1, 1, 1, 1]), np.arange(6), max_fpr=0.)

import numpy as np

from wdn.models.causal_deadline import causal_deadline_groups, sensor_event_metrics


def test_deadline_uses_prefix_and_does_not_backfill():
    arrays = {"X": np.array([[1., 1.], [0., 1.], [9., 1.], [8., 1.]]),
        "labels": np.array([1, 0, 1, 0]), "families": np.full(4, 3),
        "event": np.ones(4, int), "timestep": np.array([0, 0, 1, 1]),
        "node": np.array([1, 2, 1, 2])}
    groups = causal_deadline_groups(arrays, np.array([.9, .1, .01, .99]),
        ["dynamic_innovation", "dynamic_sigma"], "drift", 1, "mean", 1.)
    sensors = groups[1]["sensors"]
    assert groups[1]["checkpoint"] == 0
    assert [sensor["rows"].tolist() for sensor in sensors] == [[0], [1]]
    assert sensor_event_metrics(groups, 1) == {
        "f1": 1., "precision": 1., "recall": 1., "tp": 1, "fp": 0, "fn": 0}


def test_unseen_sensor_is_not_selected_ahead_of_seen_sensor():
    arrays = {"X": np.array([[1., 1.], [5., 1.]]), "labels": np.array([1, 0]),
        "families": np.full(2, 4), "event": np.ones(2, int),
        "timestep": np.array([0, 1]), "node": np.array([1, 2])}
    groups = causal_deadline_groups(arrays, np.array([.8, .99]),
        ["dynamic_innovation", "dynamic_sigma"], "noise", 1, "mean", 1.)
    assert sensor_event_metrics(groups, 1)["f1"] == 1.

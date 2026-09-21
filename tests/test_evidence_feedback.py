import numpy as np
import pytest

from wdn.evidence_feedback import (ROUTER_CLASSES, aggregate_router_evidence,
                                   contiguous_groups, guarded_specialist_scores,
                                   group_values, router_targets,
                                   specialist_veto_masks)


def test_graph_time_aggregation_is_sensor_order_preserving():
    arrays = {
        "source": np.array([1, 1, 1, 1, 1]),
        "scenario": np.array([2, 2, 2, 2, 2]),
        "timestep": np.array([4, 4, 5, 5, 5]),
    }
    local = np.array([[1., 2.], [3., 4.], [2., 1.], [4., 3.], [6., 5.]])
    group, starts = contiguous_groups(arrays)
    aggregate = aggregate_router_evidence(local, group, starts)
    assert group.tolist() == [0, 0, 1, 1, 1]
    assert starts.tolist() == [0, 2]
    np.testing.assert_allclose(aggregate[0, :2], [2., 3.])
    np.testing.assert_allclose(aggregate[0, -2:], [3., 4.])
    np.testing.assert_allclose(group_values(np.array([4, 4, 5, 5, 5]), starts), [4, 5])


def test_noncontiguous_graph_time_group_is_rejected():
    arrays = {
        "source": np.array([1, 1, 1]),
        "scenario": np.array([2, 2, 2]),
        "timestep": np.array([4, 5, 4]),
    }
    with pytest.raises(ValueError, match="not contiguous"):
        contiguous_groups(arrays)


def test_router_target_merges_random_and_targeted_as_abrupt():
    assert router_targets(np.arange(6)).tolist() == [0, 1, 2, 3, 4, 1]


def test_router_and_feedback_both_change_specialist_ordering():
    base = np.full((2, 2), .8)
    router = np.full((2, len(ROUTER_CLASSES)), .05)
    router[0, [2, 4]] = [.8, .1]  # replay beats noise
    router[1, [2, 4]] = [.1, .8]  # noise beats replay
    router[:, 3] = .2
    router /= router.sum(axis=1, keepdims=True)
    feedback = np.array([[.5, .1], [.5, .9]])
    adjusted = guarded_specialist_scores(base, router, feedback, .5, .5)
    assert adjusted[1, 1] > adjusted[0, 1]
    no_feedback = guarded_specialist_scores(base, router, np.full((2, 2), .5), .5, 0.)
    assert adjusted[1, 1] > no_feedback[1, 1]


def test_feedback_veto_requires_replay_route_and_feedback_rejection():
    router = np.array([
        [.0, .0, .8, .1, .1],
        [.0, .0, .8, .1, .1],
        [.0, .0, .1, .1, .8],
    ])
    feedback = np.array([[.2, .2], [.8, .8], [.2, .2]])
    veto = specialist_veto_masks(router, feedback, margin=.1, cutoff=.5)
    assert veto.tolist() == [[True, True], [False, False], [False, False]]

import torch

from wdn.models.temporal_multitask import masked_temporal_statistics
from wdn.models.mechanism_moe import mechanism_features, specialist_features, MechanismMoE


def test_masked_statistics_ignore_missing_placeholders():
    values = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [0.0, 30.0],
            [4.0, 40.0],
        ]
    )
    masks = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    changed_placeholder = values.clone()
    changed_placeholder[2, 0] = 9999.0

    first = masked_temporal_statistics(values, masks)
    second = masked_temporal_statistics(changed_placeholder, masks)

    for name in first:
        assert torch.allclose(first[name], second[name], atol=1e-6), name


def test_masked_statistics_use_only_valid_temporal_pairs():
    values = torch.tensor([[1.0], [2.0], [0.0], [4.0]])
    masks = torch.tensor([[1.0], [1.0], [0.0], [1.0]])

    stats = masked_temporal_statistics(values, masks)

    expected_std = torch.tensor([1.2472191])  # population std of [1, 2, 4]
    assert torch.allclose(stats["window_std"], expected_std, atol=1e-6)
    assert torch.allclose(stats["window_range"], torch.tensor([3.0]))
    assert torch.allclose(stats["n_changes"], torch.tensor([1.0]))
    assert torch.allclose(stats["temporal_delta"], torch.tensor([0.0]))
    assert torch.allclose(stats["adj_diff_std"], torch.tensor([0.0]))


def test_insufficient_history_returns_finite_neutral_statistics():
    values = torch.tensor([[7.0], [0.0], [0.0]])
    masks = torch.tensor([[1.0], [0.0], [0.0]])

    stats = masked_temporal_statistics(values, masks)

    for value in stats.values():
        assert torch.isfinite(value).all()
    assert stats["window_std"].item() == 0.0
    assert stats["log_std"].item() == 0.0
    assert stats["halves_diff"].item() == 0.0


def test_mechanism_features_ignore_missing_values_and_handle_empty_history():
    x_seq = [torch.randn(4, 7) for _ in range(8)]
    for t, x in enumerate(x_seq):
        x[:, -1] = torch.tensor([1., float(t % 2), 0., 1.])
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    changed = [x.clone() for x in x_seq]
    for x in changed:
        x[x[:, -1] == 0, -2] = 123456.
    first, second = mechanism_features(x_seq, edges), mechanism_features(changed, edges)
    for name in first:
        assert torch.isfinite(first[name]).all(), name
        assert torch.allclose(first[name], second[name]), name
    for x in changed:
        x[:, -1] = 0.
    assert all(torch.isfinite(v).all() for v in mechanism_features(changed, edges).values())


def test_mechanism_moe_forward_backward_and_uniform_control():
    torch.manual_seed(4)
    x_seq = [torch.randn(4, 7) for _ in range(8)]
    for x in x_seq:
        x[:, -1] = 1.
    edges = torch.tensor([[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]])
    model = MechanismMoE(7, 8, hidden_dim=8, router_hidden_dim=8,
                         num_layers=1, short_window=2, medium_window=4, long_window=8,
                         uniform=True)
    out = model(x_seq=x_seq, edge_index=edges, edge_attr=torch.randn(6, 8),
        is_original_edge=torch.tensor([True]*3+[False]*3), batch_size=1,
        num_nodes_per_graph=4, pressure_obs=x_seq[-1][:, -2], pressure_mask=torch.ones(4),
        flow_obs=torch.randn(3), flow_mask=torch.ones(3))
    assert out["expert_pressure_anomaly_logits"].shape == (4, 6)
    assert torch.allclose(out["pressure_anomaly_logits"], out["expert_pressure_anomaly_logits"].mean(-1))
    out["pressure_anomaly_logits"].square().mean().backward()
    for expert in model.experts:
        grad = expert.specialist_head[0].weight.grad
        assert grad is not None and torch.isfinite(grad).all()


def test_specialist_rates_bridge_missing_observations():
    xs = [torch.zeros(3, 7) for _ in range(3)]
    for t, x in enumerate(xs):
        x[:, -2] = [1., 999., 5.][t]
        x[:, -1] = [1., 0., 1.][t]
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    features = specialist_features(xs, edges)
    assert torch.allclose(features["last_observed_gap"], torch.full((3,), 2.))
    assert torch.allclose(features["last_change_rate"], torch.full((3,), 2.))


def test_specialist_residual_separates_shared_demand_from_local_drift():
    xs = [torch.zeros(3, 7) for _ in range(12)]
    edges = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    for t, x in enumerate(xs):
        x[:, -2] = t*.1
        x[:, -1] = 1.
    common = specialist_features(xs, edges)
    assert torch.allclose(common["residual_slope"], torch.zeros(3), atol=1e-6)
    for t, x in enumerate(xs):
        x[0, -2] += t*.2
    drift = specialist_features(xs, edges)
    assert torch.allclose(drift["residual_slope"][0], torch.tensor(.2), atol=1e-6)


def test_sequence_lag_features_detect_delayed_stream_with_missing_samples():
    xs = [torch.zeros(3, 7) for _ in range(12)]
    edges = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    for t, x in enumerate(xs):
        x[:, -2] = t*t*.05
        x[:, -1] = 1.
    normal = specialist_features(xs, edges)
    for t, x in enumerate(xs):
        x[0, -2] = max(0, t-2)**2*.05
        x[0, -1] = float(t >= 2 and t % 3 != 0)
    replay = specialist_features(xs, edges)
    assert replay["sequence_lag_coverage"][0] > 0
    assert replay["sequence_lag_advantage"][0] > normal["sequence_lag_advantage"][0]+.5


def test_v2_feature_and_model_finite_with_missing_data_and_no_edges():
    xs = [torch.randn(4, 7) for _ in range(8)]
    for t, x in enumerate(xs):
        x[:, -1] = torch.tensor([float(t % 2), 1., 0., 1.])
    empty = torch.zeros((2, 0), dtype=torch.long)
    values = specialist_features(xs, empty)
    assert all(torch.isfinite(v).all() for v in values.values())
    changed = [x.clone() for x in xs]
    for x in changed:
        x[x[:, -1] == 0, -2] = 1e8
    assert all(torch.allclose(v, specialist_features(changed, empty)[k]) for k, v in values.items())
    edges = torch.tensor([[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]])
    model = MechanismMoE(7, 8, hidden_dim=8, router_hidden_dim=8, feature_version=2,
                        num_layers=1, short_window=2, medium_window=4, long_window=8)
    out = model(xs, edges, torch.randn(6, 8), torch.tensor([True]*3+[False]*3),
        batch_size=1, num_nodes_per_graph=4, pressure_obs=xs[-1][:, -2],
        pressure_mask=xs[-1][:, -1], flow_obs=torch.randn(3), flow_mask=torch.ones(3))
    out["pressure_anomaly_logits"].square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

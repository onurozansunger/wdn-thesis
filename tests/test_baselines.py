import numpy as np

from wdn.baselines import analytical_reconstruct_flows, incidence_matrix


def test_analytical_reconstruction():
    # 2 nodes, 2 parallel edges: 0-1, 0-1
    edge_index = np.array([[0, 0], [1, 1]])
    B = incidence_matrix(2, edge_index)
    q_true = np.array([5.0, 5.0], dtype=np.float32)
    q_obs = np.array([5.0, 0.0], dtype=np.float32)
    q_mask = np.array([True, False])

    q_hat = analytical_reconstruct_flows(q_obs, q_mask, B, rcond=1e-4)
    assert np.allclose(q_hat, q_true, atol=1e-3)

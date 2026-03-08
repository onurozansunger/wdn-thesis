import torch

from wdn.models.recon import ReconGNN
from wdn.models.multitask import MultiTaskGNN


def test_recon_forward():
    N, E = 4, 3
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    node_static = torch.randn(N, 5)
    edge_static = torch.randn(E, 4)
    p_obs = torch.randn(N)
    q_obs = torch.randn(E)
    p_mask = torch.ones(N, dtype=torch.bool)
    q_mask = torch.ones(E, dtype=torch.bool)

    model = ReconGNN(node_in_dim=5 + 2, edge_in_dim=4 + 2, hidden_dim=16, num_layers=2, dropout=0.1)
    p_hat, q_hat = model(edge_index, node_static, edge_static, p_obs, q_obs, p_mask, q_mask)
    assert p_hat.shape == (N,)
    assert q_hat.shape == (E,)


def test_multitask_forward():
    N, E = 4, 3
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    node_static = torch.randn(N, 5)
    edge_static = torch.randn(E, 4)
    p_obs = torch.randn(N)
    q_obs = torch.randn(E)
    p_mask = torch.ones(N, dtype=torch.bool)
    q_mask = torch.ones(E, dtype=torch.bool)

    model = MultiTaskGNN(node_in_dim=5 + 2, edge_in_dim=4 + 2, hidden_dim=16, num_layers=2, dropout=0.1)
    outputs = model(edge_index, node_static, edge_static, p_obs, q_obs, p_mask, q_mask)
    assert len(outputs) == 4

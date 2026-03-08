import numpy as np

from wdn.dataset import WDNDataset


def test_dataset_shapes():
    data = {
        "edge_index": np.array([[0, 1], [1, 2]]),
        "node_static": np.random.randn(3, 4).astype(np.float32),
        "edge_static": np.random.randn(2, 3).astype(np.float32),
        "P_true": np.random.randn(5, 3).astype(np.float32),
        "Q_true": np.random.randn(5, 2).astype(np.float32),
        "P_obs": np.random.randn(5, 3).astype(np.float32),
        "Q_obs": np.random.randn(5, 2).astype(np.float32),
        "P_mask": np.ones((5, 3), dtype=bool),
        "Q_mask": np.ones((5, 2), dtype=bool),
        "P_anom": np.zeros((5, 3), dtype=bool),
        "Q_anom": np.zeros((5, 2), dtype=bool),
    }
    ds = WDNDataset(data, np.arange(5))
    sample = ds[0]
    assert sample["P_obs"].shape[0] == 3
    assert sample["Q_obs"].shape[0] == 2

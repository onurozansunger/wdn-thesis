from wdn.config import load_generate_config


def test_load_generate_config():
    cfg = load_generate_config("configs/generate.yaml")
    assert cfg.inp_path.endswith("small_net.inp")
    assert cfg.corruption.missing_p >= 0.0

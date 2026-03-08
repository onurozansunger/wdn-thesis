from __future__ import annotations

from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class CorruptionConfig:
    missing_p: float = 0.3
    missing_q: float = 0.3
    noise_sigma_p: float = 0.5
    noise_sigma_q: float = 0.2
    attack_enabled: bool = False
    attack_fraction: float = 0.0
    attack_bias_p: float = 2.0
    attack_bias_q: float = 0.5
    attack_scale_p: float = 1.0
    attack_scale_q: float = 1.0
    targeted: bool = False
    target_nodes: List[str] = field(default_factory=list)
    target_edges: List[str] = field(default_factory=list)
    time_window: Optional[List[int]] = None


@dataclass
class GenerateConfig:
    seed: int = 42
    inp_path: str = "data/networks/small_net.inp"
    output_path: str = "data/datasets"
    output_name: str = "small_net_snapshots.npz"
    time_start: int = 0
    time_end: int = 24
    time_step: int = 1
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)


@dataclass
class SplitConfig:
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15
    shuffle: bool = True


@dataclass
class DataConfig:
    dataset_path: str = "data/datasets/small_net_snapshots.npz"
    split: SplitConfig = field(default_factory=SplitConfig)
    batch_size: int = 32
    num_workers: int = 0


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class OptimizationConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class ReconTrainConfig:
    seed: int = 42
    run_dir: str = "runs"
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizationConfig = field(default_factory=OptimizationConfig)
    recon_loss_on_all: bool = True


@dataclass
class MultiTaskTrainConfig:
    seed: int = 42
    run_dir: str = "runs"
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizationConfig = field(default_factory=OptimizationConfig)
    lambda_anom: float = 1.0
    recon_loss_on_all: bool = True


@dataclass
class EvalConfig:
    seed: int = 42
    run_dir: str = "runs"
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    recon_model_path: str = ""
    multitask_model_path: str = ""
    baseline_enabled: bool = True
    baseline: Dict[str, float] = field(
        default_factory=lambda: {
            "pinv_rcond": 1e-4,
            "wls_alpha": 1e-2,
            "wls_beta": 1e-2,
            "diag_eps": 1e-6,
            "mad_scale": 3.5,
        }
    )


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge_dataclass(cls, data: Dict[str, Any]):
    if data is None:
        data = {}
    field_values = {}
    for field_name, field_def in cls.__dataclass_fields__.items():
        if field_name in data:
            field_type = field_def.type
            value = data[field_name]
            if hasattr(field_type, "__dataclass_fields__"):
                field_values[field_name] = _merge_dataclass(field_type, value)
            else:
                field_values[field_name] = value
        else:
            if field_def.default is not MISSING:
                field_values[field_name] = field_def.default
            else:
                field_values[field_name] = field_def.default_factory()
    return cls(**field_values)


def load_generate_config(path: str) -> GenerateConfig:
    data = _load_yaml(path)
    return _merge_dataclass(GenerateConfig, data)


def load_recon_train_config(path: str) -> ReconTrainConfig:
    data = _load_yaml(path)
    return _merge_dataclass(ReconTrainConfig, data)


def load_multitask_train_config(path: str) -> MultiTaskTrainConfig:
    data = _load_yaml(path)
    return _merge_dataclass(MultiTaskTrainConfig, data)


def load_eval_config(path: str) -> EvalConfig:
    data = _load_yaml(path)
    return _merge_dataclass(EvalConfig, data)


def save_config(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

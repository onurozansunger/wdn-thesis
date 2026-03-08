from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_recon_errors(errors: Dict[str, np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in errors.items():
        plt.figure()
        plt.hist(values.flatten(), bins=40, alpha=0.8)
        plt.title(f"Reconstruction Error: {name}")
        plt.xlabel("Error")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / f"error_hist_{name}.png", dpi=150)
        plt.close()

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from wdn.config import load_generate_config
from wdn.data_generation import generate_dataset, save_dataset
from wdn.utils import setup_logging


logger = logging.getLogger("wdn.generate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_generate_config(args.config)
    output_dir = Path(cfg.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "generate.log")

    dataset = generate_dataset(cfg)
    output_path = save_dataset(dataset, cfg)
    logger.info("Saved dataset to %s", output_path)


if __name__ == "__main__":
    main()

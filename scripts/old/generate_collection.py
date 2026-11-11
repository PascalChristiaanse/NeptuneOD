"""generate_collection.py

This script ingests a dataset_instruction and generates:
- A collection configuration
- A set of dataset configurations, one per observation set

This collection set can then be used with the observations module to create
tudatpy ObservationCollections
"""

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from orbitdet.data import NSDBManager
from orbitdet.reproducibility import initialize

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../conf/generator",
    config_name="triton_nsdb",
)
def main(cfg: DictConfig):
    initialize(cfg)

    logger.info(f"Generating collection with {len(cfg.datasets)} component datasets")
    manager = NSDBManager(cfg.data_folder)
    config_folder = Path(cfg.config_folder).resolve()

    for name, source in cfg.datasets.items():
        match source:
            case "nsdb":
                logger.info(f"Generating dataset {name} from NSDB")
                # The downloader reuses any dataset already in data_folder, so
                # this both fetches missing data and skips present data.
                record = manager.load(name)
                manager.write_config(record, config_folder)

            case _:
                logger.error(f"Unknown source type {source} for dataset {name}")
                raise ValueError(f"Unknown source type {source} for dataset {name}")

    # Generate a collection config that references the generated dataset configs
    try:
        repo_root = Path(__file__).resolve().parents[1]
        coll_folder = repo_root / "conf" / "collection"
        coll_folder.mkdir(parents=True, exist_ok=True)

        lines = ["# @package _global_"]
        lines.append("defaults:")

        for name in cfg.datasets.keys():
            lines.append(
                f"  - /{cfg.config_folder.replace('conf/', '')}/{name}.yaml@datasets.{name}"
            )
        lines.append("")
        lines.append('name: "NSDB Triton Data"')

        config_name = HydraConfig.get().job.config_name
        out_path = coll_folder / Path(config_name + "_generated" + ".yaml")
        out_path.write_text("\n".join(lines) + "\n")
        logger.info(f"Wrote collection config to {out_path}")
    except Exception:
        logger.exception("Failed to write collection config")


if __name__ == "__main__":
    main()

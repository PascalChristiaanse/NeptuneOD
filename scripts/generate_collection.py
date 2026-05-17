"""generate_collection.py

This script ingests a dataset_instruction and generates:
- A collection configuration
- A set of dataset configurations, one per observation set

This collection set can then be used with the observations module to create tudatpy ObservationCollections
"""

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from orbitdet.data import NSDBManager
from orbitdet.reproducibility import initialize

logger = logging.getLogger(__name__)


def _read_utf8_with_question_marks(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="replace").replace("\ufffd", "?")


@hydra.main(
    version_base=None,
    config_path="../conf/dataset_instructions",
    config_name="triton_nsdb",
)
def main(cfg: DictConfig):
    initialize(cfg)

    logger.info(f"Generating collection with {len(cfg.datasets)} component datasets")
    for name, source in cfg.datasets.items():
        match source:
            case "nsdb":
                logger.info(f"Generating dataset {name} from NSDB")
                folder = Path(f"{cfg.data_folder}/{name}/").resolve()
                config_folder = Path(cfg.config_folder).resolve()

                manager = NSDBManager()
                if folder.exists():
                    logger.warning(f"Data folder {folder} already exists, skipping download")
                    data = Path(str(folder / name) + ".txt")
                    content = Path(str(folder / name) + ".html")
                else:
                    data, content = manager.download_dataset(name, folder)

                logger.info(f"Parsing content metadata for dataset {name}")
                parsed_content = manager._parse_contents_metadata(
                    _read_utf8_with_question_marks(content),
                    name,
                )
                manager.generate_hydra_configs(parsed_content, config_folder, data)
                # Download content and data files into data/name if not already created

                # Generate config that points to data file

            case _:
                logger.error(f"Unknown source type {source} for dataset {name}")
                raise ValueError(f"Unknown source type {source} for dataset {name}")

    # Generate a collection config that references the generated dataset configs
    try:
        repo_root = Path(__file__).resolve().parents[1]
        coll_folder = repo_root / "conf" / "collections"
        coll_folder.mkdir(parents=True, exist_ok=True)

        lines = ["defaults:"]
        for name in cfg.datasets.keys():
            lines.append(f"  - /{cfg.config_folder.replace("conf/", "")}/{name}.yaml@datasets.{name}")
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

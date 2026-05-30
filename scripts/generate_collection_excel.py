"""Generate an Excel workbook summarizing a collection configuration.

This utility loads the composed collection config and exports two sheets:
- ``datasets``: one row per dataset with nested configuration flattened into columns.
- ``attributes``: one row per attribute path so nested values remain easy to inspect.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from orbitdet.reproducibility import initialize

logger = logging.getLogger(__name__)


def _to_plain_value(value: Any) -> Any:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)

    if isinstance(value, dict):
        return {str(key): _to_plain_value(nested) for key, nested in value.items()}

    if isinstance(value, list):
        return [_to_plain_value(item) for item in value]

    if isinstance(value, tuple):
        return [_to_plain_value(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    return value


def _stringify_value(value: Any) -> str:
    plain_value = _to_plain_value(value)
    if isinstance(plain_value, (dict, list)):
        return json.dumps(plain_value, ensure_ascii=False, sort_keys=True, default=str)
    return "" if plain_value is None else str(plain_value)


def _iter_attributes(value: Any, prefix: str = ""):
    plain_value = _to_plain_value(value)

    if isinstance(plain_value, dict):
        if not plain_value:
            yield prefix, plain_value
            return

        for key, nested in plain_value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_attributes(nested, next_prefix)
        return

    if isinstance(plain_value, list):
        if not plain_value:
            yield prefix, plain_value
            return

        for index, nested in enumerate(plain_value):
            next_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from _iter_attributes(nested, next_prefix)
        return

    yield prefix, plain_value


def _collection_output_name(collection_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", collection_name).strip("_")
    return safe_name.lower() if safe_name else "collection"


def _build_workbook_tables(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets = OmegaConf.select(cfg, "datasets")
    if datasets is None:
        raise ValueError("Collection config must contain a 'datasets' mapping")

    wide_rows: list[dict[str, Any]] = []
    attribute_rows: list[dict[str, Any]] = []

    for dataset_key, dataset_cfg in datasets.items():
        dataset_record = _to_plain_value(dataset_cfg)
        if not isinstance(dataset_record, dict):
            dataset_record = {"value": dataset_record}

        wide_rows.append({"dataset_key": dataset_key, **dataset_record})

        dataset_identifier = dataset_record.get("identifier", "")
        dataset_type = dataset_record.get("type", "")

        for attribute_path, attribute_value in _iter_attributes(dataset_cfg):
            attribute_rows.append(
                {
                    "dataset_key": dataset_key,
                    "identifier": dataset_identifier,
                    "type": dataset_type,
                    "attribute_path": attribute_path,
                    "value": _stringify_value(attribute_value),
                    "value_type": type(attribute_value).__name__,
                }
            )

    wide_df = pd.json_normalize(wide_rows, sep=".")
    preferred_columns = [
        column for column in ["dataset_key", "identifier", "type"] if column in wide_df.columns
    ]
    remaining_columns = [column for column in wide_df.columns if column not in preferred_columns]
    wide_df = wide_df[preferred_columns + sorted(remaining_columns)]

    attribute_df = pd.DataFrame(attribute_rows)
    if not attribute_df.empty:
        attribute_df = attribute_df[
            ["dataset_key", "identifier", "type", "attribute_path", "value", "value_type"]
        ]

    return wide_df, attribute_df


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="experiments/generate_collection_excel",
)
def main(cfg: DictConfig):
    initialize(cfg)

    wide_df, attribute_df = _build_workbook_tables(cfg)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collection_name = str(OmegaConf.select(cfg, "name") or "collection")
    workbook_name = f"{_collection_output_name(collection_name)}_attributes.xlsx"
    output_path = output_dir / workbook_name

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        wide_df.to_excel(writer, sheet_name="datasets", index=False)
        attribute_df.to_excel(writer, sheet_name="attributes", index=False)

    logger.info("Wrote collection workbook to %s", output_path)


if __name__ == "__main__":
    main()

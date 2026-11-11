"""Generate an Excel workbook summarizing a collection configuration.

This utility reads the dataset YAML files named by the selected collection and
exports two sheets:
- ``datasets``: one row per dataset with nested configuration flattened into columns.
- ``attributes``: one row per attribute path so nested values remain easy to inspect.

Hydra is used only to resolve *which* collection is selected and to provide the
output directory and logging. The dataset values themselves are read straight
from the YAML files on disk, so what lands in the workbook is exactly what each
version-controlled config contains -- nothing is reconstructed by the config
composer.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from orbitdet.reproducibility import initialize

logger = logging.getLogger(__name__)


def _to_plain_value(value: Any) -> Any:
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


_NUMBERED_CHUNK = re.compile(r"(\d+)")


def _column_sort_key(column: str) -> tuple:
    """Order flattened column names naturally instead of lexicographically.

    Plain ``sorted()`` places ``format_columns.10`` directly after
    ``format_columns.1``, so the header of a wide sheet reads ``1, 10, 11, ...,
    19, 2, 20, 3``. Every column is present, but the format block looks
    scrambled and column 1 reads as an orphan. Splitting each name into text and
    integer chunks keeps numbered families in numeric order.
    """
    chunks = _NUMBERED_CHUNK.split(column)
    return tuple((0, int(chunk)) if chunk.isdigit() else (1, chunk) for chunk in chunks)


# ``- /datasets/generated/nm0081.yaml@datasets.nm0081`` inside a collection's
# ``defaults:`` list. Captures the config path and the key it is placed under.
_COLLECTION_MEMBER = re.compile(
    r"^\s*-\s*/(?P<path>[^\s@#]+\.yaml)@(?P<key>[\w.\-]+)", re.MULTILINE
)


def _conf_root() -> Path:
    """Return the Hydra config root (``conf/``), derived from this script's location."""
    return Path(__file__).resolve().parent.parent / "conf"


def _selected_collection_config(conf_root: Path) -> Path:
    """Resolve the collection YAML that Hydra selected for this run.

    Hydra records the choice made for each config cluster in
    ``runtime.choices``. Using that keeps the caller-facing interface the same
    as before, while the file read below stays an ordinary YAML document.

    Args:
        conf_root: The Hydra config root.

    Returns:
        Path to the selected collection config.
    """
    selection = HydraConfig.get().runtime.choices.get("collection")
    if selection is None:
        raise ValueError("No '/collection' entry found in the Hydra defaults list")

    path = conf_root / "collection" / f"{selection}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Collection config not found: {path}")
    return path


def _collection_members(collection_path: Path, conf_root: Path) -> list[tuple[str, Path]]:
    """Return ``(dataset_key, path)`` for every dataset listed in a collection.

    Args:
        collection_path: The collection config to read.
        conf_root: The Hydra config root, used to resolve the entry paths.

    Returns:
        Dataset key and the config file it names, in file order.
    """
    text = collection_path.read_text(encoding="utf-8")
    members: list[tuple[str, Path]] = []
    for match in _COLLECTION_MEMBER.finditer(text):
        dataset_path = conf_root / match.group("path")
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"{collection_path.name} references a missing config: {dataset_path}"
            )
        members.append((match.group("key").rsplit(".", 1)[-1], dataset_path))

    if not members:
        raise ValueError(f"{collection_path} lists no dataset entries")
    return members


def _load_dataset_configs(members: list[tuple[str, Path]]) -> list[tuple[str, dict]]:
    """Read each dataset config from disk.

    Args:
        members: Dataset key and config path, as returned by
            :func:`_collection_members`.

    Returns:
        Dataset key and the parsed config mapping.
    """
    loaded: list[tuple[str, dict]] = []
    for dataset_key, path in members:
        with path.open(encoding="utf-8") as handle:
            record = yaml.safe_load(handle)
        if not isinstance(record, dict):
            raise ValueError(f"{path} did not parse to a mapping")
        loaded.append((dataset_key, record))
    return loaded


def _build_workbook_tables(
    datasets: list[tuple[str, dict]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten loaded dataset configs into the two workbook tables.

    Args:
        datasets: Dataset key and its parsed config, in collection order.

    Returns:
        The wide ``datasets`` table and the long-form ``attributes`` table.
    """
    wide_rows: list[dict[str, Any]] = []
    attribute_rows: list[dict[str, Any]] = []

    for dataset_key, dataset_record in datasets:
        wide_rows.append({"dataset_key": dataset_key, **dataset_record})

        dataset_identifier = dataset_record.get("identifier", "")
        dataset_type = dataset_record.get("type", "")

        for attribute_path, attribute_value in _iter_attributes(dataset_record):
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
    wide_df = wide_df[preferred_columns + sorted(remaining_columns, key=_column_sort_key)]

    attribute_df = pd.DataFrame(attribute_rows)
    if not attribute_df.empty:
        attribute_df = attribute_df[
            ["dataset_key", "identifier", "type", "attribute_path", "value", "value_type"]
        ]

    return wide_df, attribute_df


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="generator/generate_collection_excel",
)
def main(cfg: DictConfig):
    initialize(cfg)

    conf_root = _conf_root()
    collection_path = _selected_collection_config(conf_root)
    members = _collection_members(collection_path, conf_root)
    datasets = _load_dataset_configs(members)
    logger.info("Read %d dataset configs listed by %s", len(datasets), collection_path.name)

    wide_df, attribute_df = _build_workbook_tables(datasets)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collection_name = collection_path.stem
    workbook_name = f"{_collection_output_name(collection_name)}_attributes.xlsx"
    output_path = output_dir / workbook_name

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        wide_df.to_excel(writer, sheet_name="datasets", index=False)
        attribute_df.to_excel(writer, sheet_name="attributes", index=False)

    logger.info("Wrote collection workbook to %s", output_path)


if __name__ == "__main__":
    main()

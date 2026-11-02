"""Hydra config and Excel output for NSDB dataset metadata.

``NSDBManager`` is the output layer. It downloads and parses through
:class:`~orbitdet.data.nsdb_downloader.NSDBDownloader` and
:class:`~orbitdet.data.nsdb_parser.NSDBParser`, then writes the results in a
form a human or a later pipeline stage can consume:

* :meth:`NSDBManager.write_config` — the dataset YAML that
  ``scripts/generate_collection.py`` feeds to Hydra, and that the observation
  factories are registered against.
* :meth:`NSDBManager.write_excel` — a workbook for inspecting many datasets at
  once.
* :meth:`NSDBManager.write_collection_config` — the collection YAML that
  references the individual dataset configs.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from orbitdet.data.nsdb_downloader import (
    CATALOGUES,
    NSDBDatasetFiles,
    NSDBDownloader,
)
from orbitdet.data.nsdb_parser import NSDBParser, NSDBRecord

logger = logging.getLogger(__name__)

__all__ = ["NSDBManager"]


class NSDBManager:
    """Download, parse and export NSDB dataset metadata.

    Args:
        data_folder: Folder holding NSDB dataset subdirectories. Defaults to
            ``data/raw`` relative to the working directory.
        timeout_seconds: Per-request timeout, passed to the downloader.
        downloader: A pre-built downloader. Supplying one allows a custom
            session to be injected; mainly useful in tests.
        parser: A pre-built parser.

    Examples:
        >>> manager = NSDBManager("data/raw")
        >>> record = manager.load("nm0001")             # doctest: +SKIP
        >>> record.planet["name"]                       # doctest: +SKIP
        'Neptune'
    """

    def __init__(
        self,
        data_folder: str | Path = "data/raw",
        *,
        timeout_seconds: float = 30.0,
        downloader: NSDBDownloader | None = None,
        parser: NSDBParser | None = None,
    ) -> None:
        self.data_folder = Path(data_folder)
        self.downloader = (
            downloader
            if downloader is not None
            else NSDBDownloader(self.data_folder, timeout_seconds=timeout_seconds)
        )
        self.parser = parser if parser is not None else NSDBParser()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, identifier: str, *, force_download: bool = False) -> NSDBRecord:
        """Ensure a dataset is available locally and parse its descriptor.

        Args:
            identifier: Dataset name, e.g. ``nm0001``.
            force_download: Re-download even when the files already exist.

        Returns:
            The parsed record.
        """
        files = self.downloader.fetch(identifier, force=force_download)
        return self.parser.parse(files.read_descriptor(), identifier=identifier)

    def load_many(
        self, identifiers: Iterable[str], *, force_download: bool = False
    ) -> dict[str, NSDBRecord]:
        """Parse several datasets.

        Args:
            identifiers: Dataset names.
            force_download: Re-download even when the files already exist.

        Returns:
            Dataset name to its record.
        """
        return {
            identifier: self.load(identifier, force_download=force_download)
            for identifier in identifiers
        }

    def files_for(self, identifier: str, *, force_download: bool = False) -> NSDBDatasetFiles:
        """Return the on-disk file pair for a dataset, downloading if needed."""
        return self.downloader.fetch(identifier, force=force_download)

    # ------------------------------------------------------------------
    # YAML output
    # ------------------------------------------------------------------

    def write_config(
        self,
        record: NSDBRecord | str,
        output_dir: str | Path,
        data_file: str | Path | None = None,
    ) -> Path:
        """Write a dataset YAML config for one record.

        Args:
            record: A parsed record, or a dataset name to load.
            output_dir: Directory to write into. Created if absent.
            data_file: Path to the dataset's ``.txt``. Defaults to the path the
                downloader manages. Stored as an absolute path, because the
                factories resolve it from wherever Hydra is run.

        Returns:
            The path written.
        """
        if isinstance(record, str):
            identifier = record
            record = self.load(record)
            if data_file is None:
                data_file = self.downloader.dataset_folder(identifier) / f"{identifier}.txt"
        elif data_file is None:
            data_file = (
                self.downloader.dataset_folder(record.identifier) / f"{record.identifier}.txt"
            )

        config = record.to_config(data_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{record.identifier}.yaml"
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        logger.info("Wrote dataset config %s", destination)
        return destination

    def write_collection_config(
        self,
        identifiers: Iterable[str],
        output_dir: str | Path,
        config_folder: str | Path,
        *,
        name: str = "NSDB",
    ) -> Path:
        """Write a collection YAML referencing individual dataset configs.

        Hydra needs the nested configs addressed by their path relative to the
        config search path, which is why ``config_folder`` is passed in rather
        than derived.

        Args:
            identifiers: Dataset names, in the order they should appear.
            output_dir: Directory to write the collection config into.
            config_folder: Directory holding the dataset configs, as it appears
                in the Hydra search path (e.g. ``conf/datasets/generated``).
            name: Collection name written into the config.

        Returns:
            The path written.
        """
        config_folder = Path(config_folder)
        lines = ["# @package _global_", "defaults:"]
        for identifier in identifiers:
            relative = config_folder.as_posix().removeprefix("conf/")
            lines.append(f"  - /{relative}/{identifier}.yaml@datasets.{identifier}")
        lines += ["", f'name: "{name}"', ""]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / "collection.yaml"
        destination.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote collection config %s", destination)
        return destination

    def write_all(
        self,
        identifiers: Iterable[str],
        config_dir: str | Path,
        collection_dir: str | Path,
        *,
        force_download: bool = False,
        name: str = "NSDB",
    ) -> dict[str, Path]:
        """Load several datasets, write each config, then the collection config.

        Args:
            identifiers: Dataset names, in order.
            config_dir: Directory for the per-dataset configs.
            collection_dir: Directory for the collection config.
            force_download: Re-download even when files already exist.
            name: Collection name.

        Returns:
            Dataset name to the config path written.
        """
        identifiers = list(identifiers)
        written: dict[str, Path] = {}
        for identifier in identifiers:
            record = self.load(identifier, force_download=force_download)
            written[identifier] = self.write_config(record, config_dir)

        self.write_collection_config(identifiers, collection_dir, config_dir, name=name)
        return written

    # ------------------------------------------------------------------
    # Excel output
    # ------------------------------------------------------------------

    def records_to_dataframe(self, records: Iterable[NSDBRecord]) -> pd.DataFrame:
        """Flatten records into one inspection-friendly row per dataset.

        Nested values are rendered as compact strings rather than being
        exploded, because the point of this view is to compare many datasets at
        a glance and spot which fields are missing.

        Args:
            records: The records to tabulate.

        Returns:
            One row per dataset.
        """
        rows: list[dict[str, Any]] = []
        for record in records:
            rows.append(
                {
                    "identifier": record.identifier,
                    "type": record.observables,
                    "planet": (record.planet or {}).get("name"),
                    "satellites": ", ".join(record.satellites),
                    "total_number": record.total_number,
                    "start_year": (record.dates or {}).get("start_year"),
                    "end_year": (record.dates or {}).get("end_year"),
                    "observatory_code": record.primary_observatory_code,
                    "observatory_name": (
                        record.observatories[0].get("name") if record.observatories else None
                    ),
                    "n_observatories": len(record.observatories),
                    "multi_observatory": len(record.observatories) > 1,
                    "reference_frame": record.parameters.get("reference_frame"),
                    "epoch_of_equinox": record.parameters.get("epoch_of_equinox"),
                    "centre_of_frame": record.parameters.get("centre_of_frame"),
                    "time_scale": record.parameters.get("time_scale"),
                    "coordinates": record.parameters.get("coordinates"),
                    "receptor": record.parameters.get("receptor"),
                    "reduction": record.parameters.get("reduction"),
                    "relative_to": (record.parameters.get("relative_to") or {}).get("designation"),
                    "n_format_columns": len(record.format_columns),
                    "telescope": (
                        ", ".join(str(t.get("name")) for t in record.telescopes)
                        if isinstance(record.telescopes, list)
                        else record.telescopes
                    ),
                    "has_reference": record.reference is not None,
                    "has_comments": record.comments is not None,
                    "unparsed_observatory_lines": len(record.unparsed_observatory_lines),
                }
            )
        return pd.DataFrame(rows)

    def write_excel(
        self,
        records: Iterable[NSDBRecord],
        output_path: str | Path,
        *,
        attribute_sheet: bool = True,
    ) -> Path:
        """Write a workbook summarising records.

        Two sheets are produced: ``datasets``, one row per dataset with the
        fields most useful for comparison, and ``attributes``, one row per
        (dataset, field) pair so long values such as the bibliography stay
        readable without widening the first sheet.

        Args:
            records: The records to write.
            output_path: Destination ``.xlsx``.
            attribute_sheet: Whether to include the long-form sheet.

        Returns:
            The path written.
        """
        records = list(records)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = self.records_to_dataframe(records)
        # A value may legitimately be a list, which Excel cannot store; render
        # those as text rather than letting openpyxl raise.
        for column in summary.columns:
            if summary[column].map(lambda value: isinstance(value, (list, dict))).any():
                summary[column] = summary[column].map(_stringify)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="datasets", index=False)
            if attribute_sheet:
                self._write_attribute_sheet(writer, records)

        logger.info("Wrote %d datasets to %s", len(records), output_path)
        return output_path

    @staticmethod
    def _write_attribute_sheet(writer: pd.ExcelWriter, records: list[NSDBRecord]) -> None:
        """Write the long-form (dataset, field, value) sheet."""
        rows: list[dict[str, Any]] = []
        for record in records:
            for field_name, value in record.to_config().items():
                if isinstance(value, dict):
                    for key, nested in _flatten(value).items():
                        rows.append(
                            {
                                "identifier": record.identifier,
                                "field": f"{field_name}.{key}",
                                "value": _stringify(nested),
                            }
                        )
                elif isinstance(value, list):
                    for position, item in enumerate(value):
                        if isinstance(item, dict):
                            for key, nested in _flatten(item).items():
                                rows.append(
                                    {
                                        "identifier": record.identifier,
                                        "field": f"{field_name}[{position}].{key}",
                                        "value": _stringify(nested),
                                    }
                                )
                        else:
                            rows.append(
                                {
                                    "identifier": record.identifier,
                                    "field": f"{field_name}[{position}]",
                                    "value": _stringify(item),
                                }
                            )
                else:
                    rows.append(
                        {
                            "identifier": record.identifier,
                            "field": field_name,
                            "value": _stringify(value),
                        }
                    )

        pd.DataFrame(rows).to_excel(writer, sheet_name="attributes", index=False)

    def write_excel_for(
        self,
        identifiers: Iterable[str],
        output_path: str | Path,
        *,
        force_download: bool = False,
    ) -> Path:
        """Load several datasets and write them to a workbook.

        Args:
            identifiers: Dataset names.
            output_path: Destination ``.xlsx``.
            force_download: Re-download even when files already exist.

        Returns:
            The path written.
        """
        records = self.load_many(identifiers, force_download=force_download)
        return self.write_excel(list(records.values()), output_path)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def available_datasets(self, planet: str) -> pd.DataFrame:
        """Return the datasets NSDB publishes for a planet.

        Args:
            planet: Planet name, e.g. ``"Neptune"``.

        Returns:
            A dataframe with one row per dataset.
        """
        identifiers = self.downloader.fetch_available_datasets(planet)
        return pd.DataFrame({"identifier": identifiers})

    @staticmethod
    def known_planets() -> list[str]:
        """Return the planet names NSDB catalogues are known for."""
        return sorted(CATALOGUES)


def _flatten(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into dotted keys."""
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        else:
            flat[name] = value
    return flat


def _stringify(value: Any) -> Any:
    """Render a nested value as a string, leaving scalars untouched."""
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, (dict, list)):
        import json

        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value

"""Produce a standard observation frame from a dataset config.

This is the seam between reading data and building models.  ``orbitdet.data``
owns it because it needs to know where NSDB keeps its files; the factories in
``orbitdet.observations`` never see a raw file, only the frame this returns.

The pipeline it completes:

.. code-block:: text

    descriptor + data file
        -> NSDBParser        (text to record)
        -> NSDBFormatter     (record to standard frame)
        -> this module       (config to standard frame)
        -> observation factory
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .nsdb_formatter import NSDBFormatter
from .nsdb_parser import NSDBParser

logger = logging.getLogger(__name__)

__all__ = ["NSDBSource"]

#: Where the downloaded NSDB tree lives by default.
DEFAULT_DATA_FOLDER = "data/raw"


class NSDBSource:
    """Read one or more NSDB datasets as standard observation frames.

    A source is stateless beyond its data folder, so one instance serves every
    dataset in a collection and reuses its parser and formatter.

    Attributes:
        data_folder: Root of the downloaded ``nmXXXX`` tree.

    Examples:
        >>> source = NSDBSource("data/raw")          # doctest: +SKIP
        >>> frame = source.frame("nm0001")           # doctest: +SKIP
    """

    def __init__(self, data_folder: str | Path = DEFAULT_DATA_FOLDER) -> None:
        self.data_folder = Path(data_folder)
        self.parser = NSDBParser()
        self.formatter = NSDBFormatter(data_folder=self.data_folder)

    def descriptor_for(self, identifier: str, *, data_file: str | Path | None = None) -> Path:
        """Path to a dataset's ``.html`` descriptor.

        The downloaded tree keeps the descriptor and the data file as siblings
        (``data/raw/nm0008/nm0008.html`` beside ``nm0008.txt``), so when the
        config names the data file the descriptor is looked for next to it.

        That matters because the NSDB root cannot be read off ``cfg.data_folder``:
        every experiment config already uses that key for the *kernel* folder
        (``data/other``), while the NSDB tree lives under ``data/raw``.  Keying off
        the dataset's own file keeps the two independent.

        Args:
            identifier: Dataset name, e.g. ``nm0008``.
            data_file: The dataset's data file, when the config names one.

        Returns:
            The descriptor path, preferring one that exists.
        """
        candidates: list[Path] = []
        if data_file is not None:
            candidates.append(Path(data_file).parent / f"{identifier}.html")
        candidates.append(self.data_folder / identifier / f"{identifier}.html")

        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]

    def frame(self, identifier: str, *, data_file: str | Path | None = None) -> pd.DataFrame:
        """Build the standard frame for one dataset.

        Args:
            identifier: Dataset name, e.g. ``nm0001``.
            data_file: Explicit path to the ``.txt`` data file, overriding the one
                recorded in the descriptor.

        Returns:
            A frame satisfying :mod:`orbitdet.preprocessing.schema`.

        Raises:
            FileNotFoundError: If the descriptor is missing.
        """
        descriptor = self.descriptor_for(identifier, data_file=data_file)
        if not descriptor.is_file():
            raise FileNotFoundError(
                f"Descriptor for {identifier} not found at {descriptor}. "
                f"Download it first, or point the NSDB source at the right tree."
            )
        record = self.parser.parse_file(descriptor)
        return self.formatter.format(record, data_file=data_file)

    def frame_from_config(self, dataset_cfg) -> pd.DataFrame:
        """Build the standard frame described by a dataset config.

        The config carries everything needed: ``identifier`` locates the
        descriptor, and ``file`` (when present) overrides the data path so a
        relocatable config still works.

        Args:
            dataset_cfg: A dataset config, normally a Hydra ``DictConfig``.

        Returns:
            The standard frame.

        Raises:
            ValueError: If the config names no dataset identifier.
        """
        identifier = dataset_cfg.get("identifier")
        if not identifier:
            raise ValueError(
                "Dataset config has no 'identifier', so its descriptor cannot be located. "
                "The identifier is what the NSDB reader keys on."
            )
        data_file = dataset_cfg.get("file")
        return self.frame(str(identifier), data_file=data_file)

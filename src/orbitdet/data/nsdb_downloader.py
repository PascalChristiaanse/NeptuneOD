"""Downloading and caching of NSDB dataset files.

Mirrors :class:`~orbitdet.data.kernel.KernelManager`: files are fetched once
into a local folder and reused on later runs, so repeated invocations of the
collection generator do not hit the network.

NSDB publishes each dataset as two files that live side by side:

* ``{name}.html`` — the descriptor, parsed by
  :class:`~orbitdet.data.nsdb_parser.NSDBParser`.
* ``{name}.txt`` — the fixed-width observation data.

Both are stored in ``{data_folder}/{name}/``, which is where the existing
``data/raw/{name}/`` trees live.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

__all__ = [
    "NSDBDownloader",
    "NSDBDatasetFiles",
    "CATALOGUES",
    "CATALOGUE_ABBREVIATIONS",
]


#: Planet name to the base URL of its dataset catalogue. The letter in the path
#: is NSDB's own primary-body abbreviation.
CATALOGUES: dict[str, str] = {
    "Mars": "https://nsdb.imcce.fr/obspos/OBS_COLL/M/",
    "Jupiter": "https://nsdb.imcce.fr/obspos/OBS_COLL/J/",
    "Saturn": "https://nsdb.imcce.fr/obspos/OBS_COLL/S/",
    "Uranus": "https://nsdb.imcce.fr/obspos/OBS_COLL/U/",
    "Neptune": "https://nsdb.imcce.fr/obspos/OBS_COLL/N/",
    "Pluto": "https://nsdb.imcce.fr/obspos/OBS_COLL/P/",
    "Asteroids": "https://nsdb.imcce.fr/obspos/OBS_COLL/abin/",
}

#: The first character of a dataset name identifies its catalogue. Besides the
#: single letters, asteroid datasets use the ``abin`` prefix.
CATALOGUE_ABBREVIATIONS: dict[str, str] = {
    "M": "Mars",
    "J": "Jupiter",
    "S": "Saturn",
    "U": "Uranus",
    "N": "Neptune",
    "P": "Pluto",
    "A": "Asteroids",
}


@dataclass(frozen=True)
class NSDBDatasetFiles:
    """The pair of files that make up one downloaded dataset.

    Attributes:
        identifier: Dataset name, e.g. ``nm0001``.
        descriptor: Path to the ``.html`` descriptor.
        data: Path to the ``.txt`` observation data.
        downloaded: Whether this call fetched the files, or they were already
            present.
    """

    identifier: str
    descriptor: Path
    data: Path
    downloaded: bool

    def read_descriptor(self) -> str:
        """Return the descriptor HTML as text.

        Decoded as UTF-8 with ``errors="replace"`` and replacement characters
        turned into question marks, because the descriptors carry legacy
        Latin-1 author names.
        """
        return self.descriptor.read_bytes().decode("utf-8", errors="replace").replace("\ufffd", "?")


class NSDBDownloader:
    """Fetch NSDB dataset files into a local folder, once.

    Args:
        data_folder: Root folder holding one subdirectory per dataset. Matches
            the ``cfg.data_folder`` used elsewhere in the project; the existing
            trees live under ``data/raw``.
        timeout_seconds: Per-request timeout. NSDB is a small academic server,
            so a timeout is set rather than relying on the socket default.
        session: Optional pre-configured :class:`requests.Session`. Supplying
            one lets a caller control retries, proxies and headers; it is also
            how the tests inject a fake transport.

    Examples:
        >>> downloader = NSDBDownloader("data/raw")
        >>> files = downloader.fetch("nm0001")          # doctest: +SKIP
        >>> files.descriptor.name
        'nm0001.html'
    """

    def __init__(
        self,
        data_folder: str | Path,
        *,
        timeout_seconds: float = 30.0,
        session: Any | None = None,
    ) -> None:
        self.data_folder = Path(data_folder)
        self.timeout_seconds = timeout_seconds
        self._session = session

    @property
    def session(self) -> Any:
        """The requests session, created on first use."""
        if self._session is None:
            self._session = requests.Session()
        return self._session

    # ------------------------------------------------------------------
    # URL construction
    # ------------------------------------------------------------------

    @staticmethod
    def catalogue_for(identifier: str) -> str:
        """Return the catalogue URL for a dataset name.

        Args:
            identifier: Dataset name such as ``nm0001``.

        Returns:
            The catalogue's base URL.

        Raises:
            ValueError: If the name has no recognised catalogue prefix, or is
                too short to carry one.
        """
        if not identifier:
            raise ValueError("Dataset name must not be empty.")

        prefix = identifier[0].upper()
        if prefix == "A" and identifier[:4].lower() == "abin":
            return CATALOGUES["Asteroids"]

        planet = CATALOGUE_ABBREVIATIONS.get(prefix)
        if planet is None:
            raise ValueError(
                f"Unknown catalogue prefix {identifier[0]!r} in dataset "
                f"{identifier!r}. Known prefixes: "
                f"{sorted(CATALOGUE_ABBREVIATIONS)}."
            )
        return CATALOGUES[planet]

    @classmethod
    def url_for(cls, identifier: str, suffix: str) -> str:
        """Return the URL of one of a dataset's files.

        Args:
            identifier: Dataset name.
            suffix: ``"html"`` for the descriptor or ``"txt"`` for the data.

        Returns:
            The absolute URL.
        """
        # The catalogue URLs are written with a trailing slash for readability;
        # joining naively would produce ``N//nm0001.html``.
        base = cls.catalogue_for(identifier).rstrip("/")
        return f"{base}/{identifier}.{suffix}"

    def dataset_folder(self, identifier: str) -> Path:
        """Return the folder a dataset's files are stored in."""
        return self.data_folder / identifier

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def is_available_locally(self, identifier: str) -> bool:
        """Whether both of a dataset's files are already on disk."""
        folder = self.dataset_folder(identifier)
        return (folder / f"{identifier}.html").exists() and (folder / f"{identifier}.txt").exists()

    def fetch(self, identifier: str, *, force: bool = False) -> NSDBDatasetFiles:
        """Ensure a dataset's files are on disk, downloading them if needed.

        Both files are fetched together: the descriptor is needed to interpret
        the data, so having only one of them is not a usable state.

        Args:
            identifier: Dataset name.
            force: Re-download even when the files already exist.

        Returns:
            The paths to both files, and whether this call performed a download.

        Raises:
            FileNotFoundError: If NSDB returns 404 for either file.
            requests.HTTPError: For any other HTTP failure.
        """
        folder = self.dataset_folder(identifier)
        descriptor_path = folder / f"{identifier}.html"
        data_path = folder / f"{identifier}.txt"

        if not force and descriptor_path.exists() and data_path.exists():
            logger.info("Dataset %s already present in %s, skipping download", identifier, folder)
            return NSDBDatasetFiles(identifier, descriptor_path, data_path, downloaded=False)

        folder.mkdir(parents=True, exist_ok=True)

        for suffix, destination in (("html", descriptor_path), ("txt", data_path)):
            self._download(self.url_for(identifier, suffix), destination, identifier)

        return NSDBDatasetFiles(identifier, descriptor_path, data_path, downloaded=True)

    def fetch_many(
        self, identifiers: list[str], *, force: bool = False
    ) -> dict[str, NSDBDatasetFiles]:
        """Fetch several datasets, in order.

        Args:
            identifiers: Dataset names.
            force: Re-download even when files already exist.

        Returns:
            Dataset name to its files.
        """
        return {identifier: self.fetch(identifier, force=force) for identifier in identifiers}

    def fetch_available_datasets(self, planet: str) -> list[str]:
        """List the datasets NSDB publishes for a planet.

        Args:
            planet: Planet name, e.g. ``"Neptune"``.

        Returns:
            Dataset names in the order NSDB lists them.

        Raises:
            ValueError: If the planet is not in :data:`CATALOGUES`, or NSDB
                serves a page without the expected table.
        """
        import bs4

        if planet not in CATALOGUES:
            raise ValueError(f"Unknown planet {planet!r}. Known planets: {sorted(CATALOGUES)}.")

        response = self.session.get(CATALOGUES[planet], timeout=self.timeout_seconds)
        response.raise_for_status()

        soup = bs4.BeautifulSoup(response.content, "html.parser")
        table = soup.find("table")
        if table is None:
            raise ValueError(
                f"No dataset table found for {planet}. It may not be available on NSDB."
            )

        names: list[str] = []
        for row in table.find_all("tr"):
            for cell in row.find_all("td"):
                text = cell.get_text(strip=True)
                # Rows list each dataset twice, once as .html and once as .txt.
                if text.endswith(".html"):
                    names.append(text[: -len(".html")])
        return names

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _download(self, url: str, destination: Path, identifier: str) -> None:
        """Stream a single file to disk.

        Written to a temporary name first, then moved into place, so an
        interrupted download cannot leave a truncated file that a later run
        would treat as already fetched. That matters here because the descriptor
        is parsed on the assumption that it is complete.

        Args:
            url: Absolute URL to fetch.
            destination: Final path.
            identifier: Dataset name, for logging.

        Raises:
            FileNotFoundError: If the server reports 404.
        """
        logger.info("Downloading %s -> %s", url, destination)
        response = self.session.get(url, timeout=self.timeout_seconds, stream=True)
        if response.status_code == 404:
            raise FileNotFoundError(f"NSDB has no file at {url} (dataset {identifier}).")
        response.raise_for_status()

        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

        logger.info("Downloaded %s (%d bytes)", destination.name, destination.stat().st_size)

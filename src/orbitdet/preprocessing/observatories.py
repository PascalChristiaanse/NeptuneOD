"""Observatory reference data and identifier normalisation.

The MPC ``observatories.txt`` catalogue (mirrored from Project Pluto) provides
the geodetic position of every observatory code.  Three things in the pipeline
need it:

* station longitude, for the local mean time meridian offsets in
  :mod:`orbitdet.preprocessing.time_scales`;
* the geodetic triple, to add a ground station to a ``SystemOfBodies``;
* a human-readable name, for plot labels and log messages.

The file path is not passed through every call.  Instead it is configured once,
either explicitly with :func:`set_observatories_path` or implicitly from a Hydra
config, and :func:`observatories_file_path` resolves it thereafter.  This keeps
the pure preprocessing functions free of config plumbing while still working
with the existing ``cfg.data_folder`` layout.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "OBSERVATORY_INFO_FILE",
    "ObservatoryInfo",
    "set_observatories_path",
    "observatories_file_path",
    "normalize_observatory_code",
    "is_numeric_observatory_code",
    "load_observatories",
    "get_observatory_info",
    "get_observatory_location",
    "get_observatory_longitude",
]

#: Name of the MPC observatory catalogue, also mirrored at
#: https://www.projectpluto.com/mpc_stat.txt
OBSERVATORY_INFO_FILE = "observatories.txt"

#: Module-level override set by :func:`set_observatories_path`.
_observatories_path: Path | None = None


@dataclass(frozen=True)
class ObservatoryInfo:
    """A single observatory record from ``observatories.txt``.

    Attributes:
        code: MPC code, zero-padded to three digits.
        longitude_degrees: Longitude east of Greenwich, in degrees.
        latitude_degrees: Geodetic latitude, in degrees north.
        altitude_m: Altitude above the reference ellipsoid, in metres.
        rho_cos_phi: MPC ``rho cos phi'`` geocentric coordinate term.
        rho_sin_phi: MPC ``rho sin phi'`` geocentric coordinate term.
        region: Region or country, when the catalogue records one.
        name: Observatory name.
    """

    code: str
    longitude_degrees: float
    latitude_degrees: float
    altitude_m: float
    rho_cos_phi: float
    rho_sin_phi: float
    region: str
    name: str

    @property
    def longitude_radians(self) -> float:
        """Longitude east of Greenwich, in radians."""
        return float(np.deg2rad(self.longitude_degrees))

    @property
    def latitude_radians(self) -> float:
        """Geodetic latitude, in radians."""
        return float(np.deg2rad(self.latitude_degrees))

    @property
    def geodetic_position(self) -> list[float]:
        """Position as ``[altitude_m, latitude_rad, longitude_rad]``.

        This is the argument order expected by
        ``tudatpy.dynamics.environment_setup.add_ground_station`` with
        ``element_conversion.geodetic_position_type``.
        """
        return [self.altitude_m, self.latitude_radians, self.longitude_radians]


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------


def set_observatories_path(path: str | Path | None) -> None:
    """Configure the location of ``observatories.txt``.

    Call once at start-up, or pass ``cfg.data_folder`` to :func:`configure_from_config`.
    Clearing the cache is handled automatically.

    Args:
        path: Path to the file, or a directory containing it.  ``None`` clears
            the override.
    """
    global _observatories_path

    if path is None:
        _observatories_path = None
    else:
        candidate = Path(path)
        if candidate.is_dir():
            candidate = candidate / OBSERVATORY_INFO_FILE
        _observatories_path = candidate

    # The cache lives on the loader that actually carries the decorator.  Asking
    # the wrong function for its cache is a no-op, so a path change would leave
    # the old catalogue cached and the new path silently ignored.
    _load_observatories_cached.cache_clear()
    logger.debug("Observatory catalogue path set to %s", _observatories_path)


def configure_from_config(cfg) -> None:
    """Configure the observatory catalogue path from a Hydra config.

    Reads ``cfg.data_folder`` and appends :data:`OBSERVATORY_INFO_FILE`.  Does
    nothing when the key is absent, or when the resolved file does not exist, so
    that pure preprocessing use does not depend on project configuration.

    Args:
        cfg: A Hydra ``DictConfig`` (or any mapping-like object) carrying
            ``data_folder``.
    """
    data_folder = getattr(cfg, "data_folder", None)
    if data_folder is None:
        return

    candidate = Path(str(data_folder)) / OBSERVATORY_INFO_FILE
    if candidate.exists():
        set_observatories_path(candidate)
    else:
        logger.debug("No observatory catalogue found at %s", candidate)


def observatories_file_path() -> Path:
    """Return the configured path to ``observatories.txt``.

    Returns:
        The resolved path.

    Raises:
        FileNotFoundError: If no path has been configured, or the configured
            file does not exist.
    """
    if _observatories_path is None:
        raise FileNotFoundError(
            "No observatory catalogue configured. Call "
            "orbitdet.preprocessing.observatories.set_observatories_path(...) "
            "or configure_from_config(cfg) before using timescales that need a "
            "station meridian."
        )
    if not _observatories_path.exists():
        raise FileNotFoundError(f"Observatory catalogue not found at {_observatories_path}")
    return _observatories_path


# ---------------------------------------------------------------------------
# Identifier handling
# ---------------------------------------------------------------------------


def normalize_observatory_code(observatory_code: int | str) -> str:
    """Normalise an observatory identifier for use as a station name.

    MPC codes are zero-padded to three digits.  Negative codes identify
    spacecraft rather than ground stations (``-31`` is Voyager 2), and are
    returned unchanged apart from whitespace stripping, because their sign is
    meaningful and padding would corrupt them.

    Args:
        observatory_code: The code as an integer or string.

    Returns:
        The normalised code.

    Examples:
        >>> normalize_observatory_code(83)
        '083'
        >>> normalize_observatory_code(-31)
        '-31'
        >>> normalize_observatory_code(" 673 ")
        '673'
    """
    text = str(observatory_code).strip()
    if text.startswith("-"):
        return text
    return text.zfill(3)


def is_numeric_observatory_code(observatory_code: int | str) -> bool:
    """Whether an identifier denotes a ground station rather than a spacecraft.

    Args:
        observatory_code: The code to test.

    Returns:
        ``True`` for non-negative integer codes, ``False`` for spacecraft
        identifiers such as ``-31`` and for free-text names.

    Examples:
        >>> is_numeric_observatory_code(673)
        True
        >>> is_numeric_observatory_code(-31)
        False
    """
    try:
        return int(str(observatory_code).strip()) >= 0
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_observatories_cached(path: str) -> tuple[ObservatoryInfo, ...]:
    """Parse ``observatories.txt`` once, caching on the resolved path."""
    records: list[ObservatoryInfo] = []

    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("Pl "):
                continue

            # Columns: Pl Code Longitude Latitude Altitude rho_cos rho_sin_phi
            # then a free-text "region  name" tail separated by two or more spaces.
            parts = line.split(maxsplit=7)
            if len(parts) < 8:
                continue

            tail = parts[7]
            region = ""
            name = tail
            region_and_name = re.split(r"\s{2,}", tail, maxsplit=1)
            if len(region_and_name) == 2:
                region, name = region_and_name

            try:
                records.append(
                    ObservatoryInfo(
                        code=str(parts[1]).zfill(3),
                        longitude_degrees=float(parts[2]),
                        latitude_degrees=float(parts[3]),
                        altitude_m=float(parts[4]),
                        rho_cos_phi=float(parts[5]),
                        rho_sin_phi=float(parts[6]),
                        region=region,
                        name=name,
                    )
                )
            except ValueError:
                logger.debug("Skipping unparseable observatory line: %r", line)
                continue

    return tuple(records)


def load_observatories(path: str | Path | None = None) -> pd.DataFrame:
    """Load the full observatory catalogue as a dataframe.

    Args:
        path: Explicit path to ``observatories.txt``.  Defaults to the
            configured location.

    Returns:
        DataFrame indexed by observatory code, with columns ``longitude``,
        ``latitude``, ``altitude``, ``rho_cos``, ``rho_sin_phi``, ``region``
        and ``name``.

    Raises:
        FileNotFoundError: If no catalogue is configured or the file is missing.
    """
    resolved = Path(path) if path is not None else observatories_file_path()
    records = _load_observatories_cached(str(resolved))

    frame = pd.DataFrame(
        [
            {
                "code": record.code,
                "longitude": record.longitude_degrees,
                "latitude": record.latitude_degrees,
                "altitude": record.altitude_m,
                "rho_cos": record.rho_cos_phi,
                "rho_sin": record.rho_sin_phi,
                "region": record.region,
                "name": record.name,
            }
            for record in records
        ]
    )
    if frame.empty:
        return frame.set_index(pd.Index([], name="code"))
    return frame.set_index("code")


def get_observatory_info(
    observatory_code: int | str,
    path: str | Path | None = None,
) -> ObservatoryInfo:
    """Look up a single observatory by MPC code.

    Args:
        observatory_code: The MPC code.
        path: Explicit path to ``observatories.txt``.  Defaults to the
            configured location.

    Returns:
        The matching :class:`ObservatoryInfo`.

    Raises:
        FileNotFoundError: If no catalogue is configured or the file is missing.
        KeyError: If the code is not present in the catalogue.
    """
    resolved = Path(path) if path is not None else observatories_file_path()
    code = normalize_observatory_code(observatory_code)

    for record in _load_observatories_cached(str(resolved)):
        if record.code == code:
            return record

    raise KeyError(f"No observatory with code {code!r} in {resolved}")


def get_observatory_location(
    observatory_code: int | str,
    path: str | Path | None = None,
) -> tuple[float, float, float]:
    """Return the geodetic position of an observatory.

    Args:
        observatory_code: The MPC code.
        path: Explicit path to ``observatories.txt``.

    Returns:
        Tuple ``(longitude_radians, latitude_radians, altitude_m)``, with
        longitude positive to the east and latitude positive to the north.

    Raises:
        KeyError: If the code is not in the catalogue.
    """
    info = get_observatory_info(observatory_code, path=path)
    return (info.longitude_radians, info.latitude_radians, info.altitude_m)


def get_observatory_longitude(
    observatory_code: int | str,
    path: str | Path | None = None,
) -> float:
    """Return the longitude of an observatory, in radians east of Greenwich.

    Args:
        observatory_code: The MPC code.
        path: Explicit path to ``observatories.txt``.

    Returns:
        Longitude in radians, positive to the east.

    Raises:
        KeyError: If the code is not in the catalogue.
    """
    return get_observatory_info(observatory_code, path=path).longitude_radians

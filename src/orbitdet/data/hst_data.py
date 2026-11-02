"""Hubble Space Telescope (HST) astrometry retrieval for Neptune's moons.

Provides :class:`HSTQuery`, which combines the two public HST astrometry sources
used by NeptuneOD into a single normalised observation table:

* the **MAST archive** (via ``astroquery``'s ``MastMissions``), holding the
  pointing RA/Dec of the target and the exposure start time (UTC); and
* the **Showalter et al. (2019) Springer supplement** (an Excel workbook),
  holding relative ``ΔRA``/``Δdec`` offsets of a moon from the navigated centre
  of Neptune.

Both sources are normalised to the same schema --- absolute RA/Dec in degrees of
the observed moon plus an epoch in seconds since J2000 TDB --- and tagged with a
``source`` column so a dataset factory can filter or weight them separately.

No HST-specific tabulated ephemeris is built: the standard HST SPICE kernel is
used directly downstream (unlike Voyager/Gaia, which need ancillary state
history).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tudatpy.astro import time_representation as time_rep

logger = logging.getLogger(__name__)

# Public URL of the Showalter et al. (2019) astrometry supplement (Excel workbook).
SPRINGER_XLSX_URL = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fs41586-019-0909-9/MediaObjects/41586_2019_909_MOESM1_ESM.xlsx"
)

# Sheet inside the Showalter workbook that holds the astrometry measurements.
SPRINGER_SHEET = "Astrometry"

# Columns requested from the MAST ``MastMissions`` HST query.
MAST_SELECT_COLUMNS = [
    "sci_data_set_name",
    "sci_targname",
    "sci_instrume",
    "sci_spec_1234",
    "sci_start_time",
    "sci_stop_time",
    "sci_actual_duration",
    "sci_ra",
    "sci_dec",
    "ang_sep",
    "sci_pi_last_name",
]

# Canonical (normalised) output columns produced by every retrieval method.
OUTPUT_COLUMNS = [
    "source",
    "moon",
    "epoch_TDB",
    "ra",
    "dec",
    "ra_error_arcsec",
    "dec_error_arcsec",
    "weight",
    "instrument",
    "filter",
    "exposure_seconds",
    "file_name",
]

# Column labels seen in the Showalter sheet, mapped to canonical names.  The
# archive uses unicode headers (``ΔRA``, ``σ(RA,dec)``), so a few ASCII aliases
# are accepted too in case the sheet is re-exported.
_SPRINGER_ALIASES: dict[str, tuple[str, ...]] = {
    "moon": ("Moon",),
    "file_name": ("File name", "File"),
    "start_time": ("Start Time",),
    "exposure": ("Exposure",),
    "tdb": ("TDB",),
    "delta_ra": ("ΔRA", "Delta RA", "dRA"),
    "delta_dec": ("Δdec", "Delta dec", "ddec"),
    "sigma_radec": ("σ(RA,dec)", "sigma(RA,dec)", "s(RA,dec)"),
    "neptune_ra": ("Neptune RA",),
    "neptune_dec": ("Neptune dec",),
    "weight": ("Weight",),
    "instrument": ("Instrument",),
    "filter": ("Filter",),
}


def _utc_to_tdb_seconds(iso_utc: object, exposure_seconds: float = 0.0) -> float:
    """Convert a UTC timestamp at mid-exposure to seconds since J2000 TDB.

    Args:
        iso_utc: Exposure start time in UTC (ISO-8601 string or any value
            accepted by :class:`pandas.Timestamp`).
        exposure_seconds: Exposure duration in seconds; half of it is added to
            the start time so the epoch refers to the midpoint of the exposure.

    Returns:
        Seconds since J2000 in the TDB time scale, or ``NaN`` if *iso_utc* is
        missing.
    """
    if pd.isna(iso_utc):
        return np.nan

    epoch_utc = time_rep.iso_string_to_epoch(pd.Timestamp(iso_utc).isoformat())
    epoch_utc += 0.5 * float(exposure_seconds or 0.0)

    converter = time_rep.default_time_scale_converter()
    epoch_tdb = converter.convert_time(
        time_rep.TimeScales.utc_scale,
        time_rep.TimeScales.tdb_scale,
        epoch_utc,
        np.zeros((3, 1)),
    )
    return float(epoch_tdb)


def _resolve_springer_columns(data: pd.DataFrame, required: tuple[str, ...]) -> dict[str, str]:
    """Map canonical column keys to the actual labels present in *data*."""
    resolved: dict[str, str] = {}
    for key, candidates in _SPRINGER_ALIASES.items():
        for candidate in candidates:
            if candidate in data.columns:
                resolved[key] = candidate
                break

    missing = [key for key in required if key not in resolved]
    if missing:
        raise ValueError(
            "Showalter astrometry sheet is missing required column(s) "
            f"{missing}. Available columns: {list(data.columns)}"
        )
    return resolved


def _optional_column(data: pd.DataFrame, column: str | None) -> pd.Series:
    """Return *column* as a Series, or an all-NaN Series when it is absent."""
    if column is None:
        return pd.Series(np.nan, index=data.index)
    return data[column]


def _finalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Reindex *frame* to the canonical output columns."""
    return frame.reindex(columns=OUTPUT_COLUMNS)


def normalise_mast_table(data: pd.DataFrame, target: str | None = "Triton") -> pd.DataFrame:
    """Normalise a raw MAST HST query table into the canonical observation schema.

    The MAST metadata reports the exposure *start* time in UTC; the epoch is
    therefore taken as ``start + exposure / 2`` and converted to TDB.

    Args:
        data: Raw table returned by ``MastMissions.query_criteria``.
        target: Target name to keep (case-insensitive).  If ``None``, all rows
            returned by the query are kept.

    Returns:
        DataFrame with the :data:`OUTPUT_COLUMNS` schema and ``source == "MAST"``.
    """
    if target is not None:
        names = data["sci_targname"].astype(str).str.strip().str.upper()
        data = data.loc[names == str(target).strip().upper()]

    if data.empty:
        logger.warning("MAST HST query returned no rows for target %s.", target)

    exposure = pd.to_numeric(data["sci_actual_duration"], errors="coerce").fillna(0.0)
    epochs = [
        _utc_to_tdb_seconds(start, duration)
        for start, duration in zip(data["sci_start_time"], exposure, strict=False)
    ]

    moon = data["sci_targname"].astype(str).str.strip() if target is None else target
    frame = pd.DataFrame(
        {
            "source": "MAST",
            "moon": moon,
            "epoch_TDB": epochs,
            "ra": pd.to_numeric(data["sci_ra"], errors="coerce"),
            "dec": pd.to_numeric(data["sci_dec"], errors="coerce"),
            "ra_error_arcsec": np.nan,
            "dec_error_arcsec": np.nan,
            "weight": 1,
            "instrument": data["sci_instrume"],
            "filter": data["sci_spec_1234"],
            "exposure_seconds": exposure,
            "file_name": data["sci_data_set_name"],
        }
    )
    return _finalise(frame)


def normalise_springer_table(data: pd.DataFrame, moon: str | None = "Triton") -> pd.DataFrame:
    """Normalise the Showalter et al. (2019) astrometry sheet into the canonical schema.

    The sheet stores the moon position as a relative offset (``ΔRA``/``Δdec``,
    arcsec) from the navigated centre of Neptune.  The absolute RA/Dec of the
    moon is reconstructed by adding this offset to the tabulated Neptune
    RA/Dec::

        ra  = Neptune RA + ΔRA / (3600 * cos(Neptune dec))
        dec = Neptune dec + Δdec / 3600

    Args:
        data: Raw sheet as read from the Excel workbook (units row skipped).
        moon: Moon name to keep (case-insensitive).  If ``None``, all moons in
            the sheet are kept.

    Returns:
        DataFrame with the :data:`OUTPUT_COLUMNS` schema and
        ``source == "Springer"``.
    """
    columns = _resolve_springer_columns(
        data, required=("moon", "delta_ra", "delta_dec", "neptune_ra", "neptune_dec")
    )
    moon_column = columns["moon"]

    if moon is not None:
        names = data[moon_column].astype(str).str.strip().str.lower()
        mask = names == str(moon).strip().lower()
        if not mask.any():
            available = sorted(data[moon_column].astype(str).str.strip().unique())
            raise LookupError(
                f"No '{moon}' measurements in the Showalter sheet. Available moons: {available}"
            )
        data = data.loc[mask]

    neptune_ra = pd.to_numeric(data[columns["neptune_ra"]], errors="coerce")
    neptune_dec = pd.to_numeric(data[columns["neptune_dec"]], errors="coerce")
    delta_ra = pd.to_numeric(data[columns["delta_ra"]], errors="coerce")
    delta_dec = pd.to_numeric(data[columns["delta_dec"]], errors="coerce")

    ra = neptune_ra + delta_ra / 3600.0 / np.cos(np.deg2rad(neptune_dec))
    dec = neptune_dec + delta_dec / 3600.0

    # The 'TDB' column is already seconds since J2000; fall back to converting
    # the UTC start time plus half the exposure.
    if "tdb" in columns:
        epoch = pd.to_numeric(data[columns["tdb"]], errors="coerce")
    elif "start_time" in columns:
        exposure = pd.to_numeric(
            _optional_column(data, columns.get("exposure")), errors="coerce"
        ).fillna(0.0)
        epoch = [
            _utc_to_tdb_seconds(start, duration)
            for start, duration in zip(data[columns["start_time"]], exposure, strict=False)
        ]
    else:
        raise ValueError(
            "Showalter astrometry sheet needs either a 'TDB' or a 'Start Time' column."
        )

    sigma = pd.to_numeric(_optional_column(data, columns.get("sigma_radec")), errors="coerce")
    weight = (
        pd.to_numeric(_optional_column(data, columns.get("weight")), errors="coerce")
        .fillna(1)
        .astype(int)
    )
    exposure_seconds = pd.to_numeric(
        _optional_column(data, columns.get("exposure")), errors="coerce"
    )

    frame = pd.DataFrame(
        {
            "source": "Springer",
            "moon": data[moon_column].astype(str).str.strip(),
            "epoch_TDB": epoch,
            "ra": ra,
            "dec": dec,
            "ra_error_arcsec": sigma,
            "dec_error_arcsec": sigma,
            "weight": weight,
            "instrument": _optional_column(data, columns.get("instrument")),
            "filter": _optional_column(data, columns.get("filter")),
            "exposure_seconds": exposure_seconds,
            "file_name": _optional_column(data, columns.get("file_name")),
        }
    )
    return _finalise(frame)


class MastQuery:
    """Thin wrapper around :class:`astroquery.mast.missions.MastMissions`."""

    def __init__(self, mission: str):
        from astroquery.mast.missions import MastMissions  # late import (slow)

        self.missions = MastMissions(mission=mission)

    def query_data(self, select_cols: list[str], **criteria) -> pd.DataFrame:
        """Run a MAST ``query_criteria`` search and return the results as a DataFrame."""
        results = self.missions.query_criteria(select_cols=select_cols, **criteria)
        return results.to_pandas()


class HSTQuery(MastQuery):
    """Retrieve and combine HST astrometry from the MAST archive and the Showalter sheet.

    The two sources are normalised to the same schema and combined into a single
    DataFrame with a ``source`` column (``"MAST"`` or ``"Springer"``).  Epochs are
    seconds since J2000 TDB and ``ra``/``dec`` are absolute coordinates of the
    observed moon in degrees.

    Attributes:
        _table (pd.DataFrame): The combined observation table after calling
            :meth:`retrieve_data`.
    """

    MAST_SOURCE = "MAST"
    SPRINGER_SOURCE = "Springer"

    def __init__(self, mission: str = "hst"):
        """Create an HST query object.

        Args:
            mission: MAST mission identifier (defaults to ``"hst"``).
        """
        super().__init__(mission)
        self._table = pd.DataFrame(columns=OUTPUT_COLUMNS)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def observation_table(self) -> pd.DataFrame:
        """Return a copy of the combined observation table."""
        return self._table.copy()

    @property
    def sources(self) -> list[str]:
        """List the distinct sources present in the current observation table."""
        if self._table.empty:
            return []
        return sorted(self._table["source"].unique().tolist())

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    @staticmethod
    def download_springer_sheet(destination: str | Path) -> Path:
        """Download the Showalter et al. (2019) astrometry supplement.

        Args:
            destination: Path to write the Excel workbook to.

        Returns:
            Path to the downloaded workbook.
        """
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(SPRINGER_XLSX_URL, timeout=30)
        response.raise_for_status()
        destination.write_bytes(response.content)
        logger.info("Downloaded Showalter HST astrometry sheet to %s.", destination)
        return destination

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_mast_data(self, target: str | None = "Triton", sci_aec: str = "S") -> pd.DataFrame:
        """Query the MAST archive and return the normalised observation table.

        Args:
            target: Target name (``sci_targname``) to query and keep.
            sci_aec: Science data quality flag passed to MAST (``"S"`` = science).

        Returns:
            Normalised DataFrame (``source == "MAST"``).
        """
        logger.info("Querying MAST for HST observations of %s.", target)
        raw = self.query_data(
            MAST_SELECT_COLUMNS, sci_aec=sci_aec, sci_targname=str(target).lower()
        )
        logger.info("MAST returned %d row(s) for %s.", len(raw), target)
        return normalise_mast_table(raw, target=target)

    def retrieve_springer_data(
        self, file_path: str | Path, moon: str | None = "Triton"
    ) -> pd.DataFrame:
        """Load the Showalter et al. (2019) supplement and normalise it.

        Args:
            file_path: Path to the Excel workbook (see
                :meth:`download_springer_sheet`).
            moon: Moon name to keep (case-insensitive).

        Returns:
            Normalised DataFrame (``source == "Springer"``).
        """
        book = pd.ExcelFile(file_path)
        sheet = SPRINGER_SHEET if SPRINGER_SHEET in book.sheet_names else book.sheet_names[0]
        # The second row of the sheet holds the unit labels, so skip it.
        raw = book.parse(sheet, header=0, skiprows=[1])
        logger.info("Loaded %d row(s) from Showalter sheet '%s'.", len(raw), sheet)
        return normalise_springer_table(raw, moon=moon)

    def retrieve_data(
        self,
        target: str | None = "Triton",
        springer_file: str | Path | None = None,
        use_mast: bool = True,
        use_springer: bool = True,
        sci_aec: str = "S",
    ) -> None:
        """Retrieve both HST sources and store the combined observation table.

        Args:
            target: Target/moon name to keep from both sources.
            springer_file: Path to the Showalter Excel workbook.  Required when
                ``use_springer`` is True.
            use_mast: Include the MAST archive observations.
            use_springer: Include the Showalter supplement observations.
            sci_aec: Science data quality flag passed to MAST.

        Returns:
            None.  The combined table is stored in :attr:`observation_table`.
        """
        frames: list[pd.DataFrame] = []
        if use_mast:
            frames.append(self.retrieve_mast_data(target=target, sci_aec=sci_aec))
        if use_springer:
            if springer_file is None:
                raise ValueError("springer_file must be provided when use_springer is True.")
            frames.append(self.retrieve_springer_data(springer_file, moon=target))
        if not frames:
            raise ValueError("At least one of use_mast / use_springer must be True.")

        table = pd.concat(frames, ignore_index=True)
        table = table.sort_values("epoch_TDB", ignore_index=True)
        self._table = table
        logger.info(
            "Combined HST observation table: %d row(s), sources=%s.",
            len(table),
            table["source"].value_counts().to_dict(),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the combined observation table."""
        return self.observation_table

# Source for 1991-1994 data STERN, CALDWELL, BOSH on MAST archive
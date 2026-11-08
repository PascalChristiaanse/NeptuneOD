"""Timescale conversion to TDB seconds since the J2000 epoch.

Observation sources record epochs in whatever timescale the observer used.
This module maps all of them onto **TDB seconds since the J2000 epoch**,
which is what TudatPy expects.

Two corrections matter for historical data, and neither is applied by a naive
timescale conversion:

Local mean time
    Washington mean time, Paris mean time, Pulkovo mean time and friends are
    *meridian* times, not zone times.  They differ from UT1 by a constant equal
    to the longitude of the reference meridian, so the conversion must shift the
    epoch by the meridian offset before treating it as a civil timescale.

The astronomical day
    Before 1925 January 1 the astronomical day ran from noon to noon, twelve
    hours behind the civil day that carried the same date.  An epoch recorded as
    ``1900-01-01 00:30`` therefore refers to the instant twelve hours *later*
    than the same digits do in the civil convention.  Every pre-1925 epoch needs
    a ``+12 h`` shift, regardless of which timescale was used.


References
----------
* Explanatory Supplement to the Astronomical Almanac, 3rd ed., ch. 2 (time
  scales) and ch. 6 (the astronomical day and its 1925 changeover).
* IAU 2006 resolutions on TCG/TCB, for the ``TCB_to_TDB`` relation.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from tudatpy.astro import time_representation as time_rep

logger = logging.getLogger(__name__)

__all__ = [
    "InputTimeScale",
    "MeridianTime",
    "ASTRONOMICAL_DAY_SWITCH_EPOCH",
    "ASTRONOMICAL_DAY_OFFSET",
    "TDB_TIME_SCALES",
    "LOCAL_MERIDIAN_TIMES",
    "resolve_timescale",
    "meridian_offset_for_station",
    "to_tdb_seconds_since_j2000",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECONDS_PER_DAY = 86400.0

#: Epoch at which the astronomical day was replaced by the civil day, as TDB
#: seconds since J2000 (1925-01-01T00:00:00).  Epochs strictly before this
#: instant are assumed to follow the noon-to-noon convention.
ASTRONOMICAL_DAY_SWITCH_EPOCH = -2366798400.0

#: Correction applied to epochs recorded before :data:`ASTRONOMICAL_DAY_SWITCH_EPOCH`.
#:
#: The astronomical day began at noon, so ``1900-01-01 00:30`` (astronomical)
#: is the civil instant ``1900-01-01 12:30`` - twelve hours *later* than the
#: literal reading of the digits.
ASTRONOMICAL_DAY_OFFSET = 12.0 * 3600.0


# ---------------------------------------------------------------------------
# Timescale vocabulary
# ---------------------------------------------------------------------------


class InputTimeScale(StrEnum):
    """Timescale of the epochs as recorded by the source.

    The members group into four families, each handled differently:

    ``UTC`` / ``TAI`` / ``TT`` / ``TDB`` / ``UT1``
        Native TudatPy timescales, converted directly.
    ``TCB`` / ``TCG``
        Barycentric and geocentric coordinate time, converted through the
        TudatPy IAU 2006 relations.
    ``LOCAL_MEAN_TIME`` / ``LOCAL_SIDEREAL_TIME``
        Meridian times such as Washington or Paris mean time.  These are UT1
        offset by a constant meridian longitude, so they require an extra
        longitude-dependent shift (see :func:`meridian_offset_for_station`).
        ``LOCAL_SIDEREAL_TIME`` is the same family but the recorded value is a
        sidereal rather than solar time; the observational difference at the
        precision of historical photographic astrometry is far below the
        measurement noise, so it is treated identically and flagged in the log.
    ``STANDARD_ZONE``
        Civil zone times such as US Pacific Standard Time, which are UT1 offset
        by a whole number of hours.
    ``UNKNOWN``
        The source carried no usable information.  Callers get UTC plus a
        warning; see :func:`resolve_timescale`.
    """

    UTC = "utc"
    TAI = "tai"
    TT = "tt"
    TDB = "tdb"
    UT1 = "ut1"
    TCB = "tcb"
    TCG = "tcg"
    LOCAL_MEAN_TIME = "local_mean_time"
    LOCAL_SIDEREAL_TIME = "local_sidereal_time"
    STANDARD_ZONE = "standard_zone"
    UNKNOWN = "unknown"


class MeridianTime(StrEnum):
    """Named local mean-time meridians, with their offset from Greenwich.

    The value is the longitude of the reference meridian expressed as a
    longitude *east* in degrees; local mean time is ahead of UT1 by
    ``longitude / 15`` hours.  Washington is 77.0667 W, hence ``-77.0667``.
    """

    WASHINGTON = "washington"
    PARIS = "paris"
    PULKOVO = "pulkovo"
    STRASBOURG = "strasbourg"
    NICE = "nice"
    MOUNT_HAMILTON = "mount_hamilton"
    LEANDER_MCCORMICK = "leander_mccormick"
    GREENWICH = "greenwich"


#: Longitude of each named meridian, in degrees east of Greenwich.
MERIDIAN_LONGITUDES: dict[MeridianTime, float] = {
    MeridianTime.WASHINGTON: -77.0667,  # U.S. Naval Observatory, 77d04m W
    MeridianTime.PARIS: 2.3372,  # Observatoire de Paris
    MeridianTime.PULKOVO: 30.3272,  # Pulkovo Observatory
    MeridianTime.STRASBOURG: 7.7522,  # Observatoire de Strasbourg
    MeridianTime.NICE: 7.2683,  # Observatoire de Nice
    MeridianTime.MOUNT_HAMILTON: -121.6425,  # Lick Observatory
    MeridianTime.LEANDER_MCCORMICK: -78.4894,  # McCormick Observatory
    MeridianTime.GREENWICH: 0.0,
}

#: Named meridians, keyed by the local mean time they define.
LOCAL_MERIDIAN_TIMES: dict[MeridianTime, InputTimeScale] = {
    meridian: InputTimeScale.LOCAL_MEAN_TIME for meridian in MeridianTime
}

#: Timescales that TudatPy can convert natively.
TDB_TIME_SCALES: dict[InputTimeScale, time_rep.TimeScales] = {
    InputTimeScale.UTC: time_rep.TimeScales.utc_scale,
    InputTimeScale.TAI: time_rep.TimeScales.tai_scale,
    InputTimeScale.TT: time_rep.TimeScales.tt_scale,
    InputTimeScale.TDB: time_rep.TimeScales.tdb_scale,
    InputTimeScale.UT1: time_rep.TimeScales.ut1_scale,
}


# ---------------------------------------------------------------------------
# Label resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimescaleResolution:
    """Result of interpreting a free-text timescale label.

    Attributes:
        scale: The resolved timescale family.
        meridian: Named meridian for local mean times, else ``None``.
        zone_offset_seconds: Fixed offset from UT1, for zone times.
        raw_label: The original label, preserved for logging and diagnostics.
        original_label_normalised: The label after cleaning, which is what was
            actually matched against the vocabulary.
    """

    scale: InputTimeScale
    meridian: MeridianTime | None = None
    zone_offset_seconds: float = 0.0
    raw_label: str = ""
    original_label_normalised: str = ""

    @property
    def needs_meridian(self) -> bool:
        """Whether this resolution requires a station meridian longitude."""
        return self.scale is InputTimeScale.LOCAL_MEAN_TIME and self.meridian is None


#: Direct label matches, checked before any pattern matching.
_SIMPLE_LABELS: dict[str, InputTimeScale] = {
    "utc": InputTimeScale.UTC,
    "ut": InputTimeScale.UT1,
    "ut1": InputTimeScale.UT1,
    "ut2": InputTimeScale.UT1,
    "tai": InputTimeScale.TAI,
    "iat": InputTimeScale.TAI,
    "tt": InputTimeScale.TT,
    "tdt": InputTimeScale.TT,
    "et": InputTimeScale.TT,
    "tdb": InputTimeScale.TDB,
    "tcb": InputTimeScale.TCB,
    "tcg": InputTimeScale.TCG,
    "gmt": InputTimeScale.UT1,
    "utg": InputTimeScale.UT1,
    "universal time": InputTimeScale.UT1,
    "universal time coordinated": InputTimeScale.UTC,
    "greenwich mean time": InputTimeScale.UT1,
    "greenwich mean astronomical time": InputTimeScale.UT1,
    "mean solar time": InputTimeScale.UT1,
}

#: Meridian abbreviations that appear without the meridian name spelled out.
_MERIDIAN_ABBREVIATIONS: dict[str, MeridianTime] = {
    "wmt": MeridianTime.WASHINGTON,
}

#: Named meridians matched as substrings of the label.
_MERIDIAN_LABELS: dict[str, MeridianTime] = {
    "washington": MeridianTime.WASHINGTON,
    "paris": MeridianTime.PARIS,
    "pulkovo": MeridianTime.PULKOVO,
    "strasbourg": MeridianTime.STRASBOURG,
    "strasb": MeridianTime.STRASBOURG,
    "nice": MeridianTime.NICE,
    "hamilton": MeridianTime.MOUNT_HAMILTON,
    "mccormick": MeridianTime.LEANDER_MCCORMICK,
    "greenwich": MeridianTime.GREENWICH,
}

#: Standard zone times: a fixed offset from UT1, in hours.
_ZONE_LABELS: dict[str, float] = {
    "eastern standard time": -5.0,
    "est": -5.0,
    "central standard time": -6.0,
    "central stand time": -6.0,
    "cst": -6.0,
    "mountain standard time": -7.0,
    "mst": -7.0,
    "pacific standard time": -8.0,
    "standard pacific time": -8.0,
    "pacific standard": -8.0,
    "pst": -8.0,
    "90 degree time": -6.0,  # 90th meridian west
    "90th meridian time": -6.0,
    "90 meridian time": -6.0,
}

#: Placement of US zone abbreviations, to disambiguate from other uses of "CST".
_ZONE_ABBREVIATIONS = {"est", "cst", "mst", "pst"}


def _normalise_label(label: str) -> str:
    """Lower-case, strip accents and collapse punctuation in a timescale label.

    Parenthesised asides are removed before matching.  Sources write labels such
    as ``"P.S.T. (Pacific S. T.)"`` or ``"90th Meridian Time = UTC - 6 h"``,
    where the parenthetical merely restates the abbreviation and would otherwise
    prevent an exact match.
    """
    text = unicodedata.normalize("NFKD", str(label))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.lower().replace(".", " ").replace(",", " ").replace("/", " ")
    text = text.replace("=", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def resolve_timescale(label: str) -> TimescaleResolution:
    """Interpret a free-text timescale label from an observation source.

    Handles the spellings and abbreviations that occur in real NSDB metadata,
    including dotted forms (``M. T.``), mixed case (``WMT``), redundant
    qualifiers (``GMT=UTC``) and free text (``90th Meridian Time = UTC - 6 h``).
    An unrecognised or empty label resolves to :attr:`InputTimeScale.UNKNOWN`
    with a warning, so ingestion can continue and the gap is visible in the log
    rather than raising in the middle of a 15 000-row ingest.

    Args:
        label: The raw timescale string from the source metadata.

    Returns:
        A :class:`TimescaleResolution` describing the scale and any offset the
        caller must apply.
    """
    normalised = _normalise_label(label or "")

    if not normalised or normalised.startswith("no information") or "no inf" in normalised:
        logger.warning("Timescale label %r carries no information; assuming UTC.", label)
        return TimescaleResolution(InputTimeScale.UNKNOWN, raw_label=str(label))

    # A label of the form "GMT UTC" or "GMT=UTC" resolves to UTC: the source
    # explicitly equates them, which is more information than "GMT" alone.
    tokens = set(normalised.split())
    if "utc" in tokens and ("gmt" in tokens or "greenwich" in normalised):
        return TimescaleResolution(
            InputTimeScale.UTC, raw_label=str(label), original_label_normalised=normalised
        )

    # Exact match first, so "ut" does not match inside another word.
    if normalised in _SIMPLE_LABELS:
        return TimescaleResolution(
            _SIMPLE_LABELS[normalised],
            raw_label=str(label),
            original_label_normalised=normalised,
        )

    # Aeroplane / geography abbreviations such as "P.S.T." arrive here as "p s t".
    squashed = normalised.replace(" ", "")
    if squashed in _SIMPLE_LABELS:
        return TimescaleResolution(
            _SIMPLE_LABELS[squashed],
            raw_label=str(label),
            original_label_normalised=normalised,
        )

    # Zone times before meridians: "pacific standard time" also contains no
    # meridian name, but "central standard time" must not be read as Greenwich.
    for pattern, hours in _ZONE_LABELS.items():
        if pattern in normalised:
            return TimescaleResolution(
                InputTimeScale.STANDARD_ZONE,
                zone_offset_seconds=hours * 3600.0,
                raw_label=str(label),
                original_label_normalised=normalised,
            )

    # Zone abbreviations written as separated initials, e.g. "P.S.T." arrives as
    # "p s t" once the dots are normalised to spaces.
    squashed = normalised.replace(" ", "")
    if squashed in _ZONE_ABBREVIATIONS:
        return TimescaleResolution(
            InputTimeScale.STANDARD_ZONE,
            zone_offset_seconds=_ZONE_LABELS[squashed] * 3600.0,
            raw_label=str(label),
            original_label_normalised=normalised,
        )

    # Zone abbreviations, but only as a standalone token so that e.g. "est"
    # does not match inside "manchester".
    for abbreviation in _ZONE_ABBREVIATIONS & tokens:
        return TimescaleResolution(
            InputTimeScale.STANDARD_ZONE,
            zone_offset_seconds=_ZONE_LABELS[abbreviation] * 3600.0,
            raw_label=str(label),
            original_label_normalised=normalised,
        )

    # Meridian abbreviations that stand alone, e.g. "WMT" for Washington.
    for abbreviation, meridian in _MERIDIAN_ABBREVIATIONS.items():
        if abbreviation in tokens:
            return TimescaleResolution(
                InputTimeScale.LOCAL_MEAN_TIME,
                meridian=meridian,
                raw_label=str(label),
                original_label_normalised=normalised,
            )

    # Named meridians, i.e. local mean time.
    for pattern, meridian in _MERIDIAN_LABELS.items():
        if pattern in normalised:
            return TimescaleResolution(
                InputTimeScale.LOCAL_MEAN_TIME,
                meridian=meridian,
                raw_label=str(label),
                original_label_normalised=normalised,
            )

    # A bare "mean time" with no named meridian: station-dependent.
    if "mean time" in normalised or "m t" in normalised or "mt" in squashed:
        return TimescaleResolution(
            InputTimeScale.LOCAL_MEAN_TIME,
            raw_label=str(label),
            original_label_normalised=normalised,
        )

    if "sidereal" in normalised:
        return TimescaleResolution(
            InputTimeScale.LOCAL_SIDEREAL_TIME,
            raw_label=str(label),
            original_label_normalised=normalised,
        )

    logger.warning(
        "Unrecognised timescale label %r (normalised to %r); assuming UTC. "
        "Add it to the vocabulary in orbitdet.preprocessing.time_scales if it "
        "denotes something other than UTC.",
        label,
        normalised,
    )
    return TimescaleResolution(
        InputTimeScale.UNKNOWN, raw_label=str(label), original_label_normalised=normalised
    )


# ---------------------------------------------------------------------------
# Meridian offsets
# ---------------------------------------------------------------------------


def meridian_offset_for_station(
    observatory_code: int | str,
    *,
    observatories_path: str | None = None,
) -> float:
    """Return the UT1 offset, in seconds, of a station's local mean time.

    A station's local mean time is ahead of UT1 by its longitude east of
    Greenwich, converted to time at 15 degrees per hour.

    Args:
        observatory_code: MPC observatory code.
        observatories_path: Path to ``observatories.txt``.  When ``None``, the
            file is looked up through
            :func:`orbitdet.preprocessing.observatories.observatories_file_path`,
            which requires prior configuration with
            :func:`orbitdet.preprocessing.observatories.set_observatories_path`.

    Returns:
        Offset in seconds: add this to a local mean time epoch to obtain the
        corresponding UT1 epoch.

    Raises:
        LookupError: If the observatory is not present in ``observatories.txt``
            and no explicit offset can be determined.
    """
    from orbitdet.preprocessing.observatories import get_observatory_longitude

    longitude = get_observatory_longitude(observatory_code, path=observatories_path)
    return longitude_to_time_offset(longitude)


def longitude_to_time_offset(longitude_radians: float) -> float:
    """Convert a longitude east of Greenwich to a UT1 offset in seconds.

    Args:
        longitude_radians: Longitude east of Greenwich, in radians.

    Returns:
        Offset in seconds.  A station 30 degrees east has local mean time
        2 hours ahead of UT1, so this returns ``+7200``.
    """
    return float(longitude_radians) * SECONDS_PER_DAY / (2.0 * np.pi)


def meridian_label_offset(meridian: MeridianTime) -> float:
    """Return the UT1 offset, in seconds, of a named meridian.

    Args:
        meridian: One of the named historical meridians.

    Returns:
        Offset in seconds, positive east.
    """
    longitude_degrees = MERIDIAN_LONGITUDES[meridian]
    return longitude_degrees * SECONDS_PER_DAY / 360.0


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def to_tdb_seconds_since_j2000(
    epochs: pd.Series | np.ndarray | list[float],
    scale: InputTimeScale | str,
    *,
    observatory_code: int | str | None = None,
    station_longitude_radians: float | None = None,
    station_earth_fixed_position: np.ndarray | None = None,
    allow_pre_1925_shift: bool = True,
) -> np.ndarray:
    """Convert epochs to TDB seconds since the J2000 epoch.

    The pipeline applied, in order:

    1. **Astronomical day correction** - ``+12 h`` for epochs before
       1925-01-01, if ``allow_pre_1925_shift``.
    2. **Meridian or zone correction** - a constant offset for local mean times
       and standard zone times, bringing the epochs to UT1.
    3. **Timescale conversion** - UT1/UTC/TAI/TT to TDB through TudatPy, or the
       IAU 2006 relation for TCB/TCG.

    Args:
        epochs: Epochs as seconds since J2000 in ``scale``.  The values are
            *raw calendar offsets* - :func:`orbitdet.preprocessing.time_formatting`
            produces them, and this function applies the scale.
        scale: The timescale of ``epochs``, as an :class:`InputTimeScale` or a
            free-text label accepted by :func:`resolve_timescale`.
        observatory_code: MPC code of the observing station.  Required when
            ``scale`` is a local mean time and neither ``station_longitude_radians``
            nor a named meridian is available.
        station_longitude_radians: Longitude of the station east of Greenwich.
            Takes precedence over ``observatory_code`` for local mean times.
        station_earth_fixed_position: Earth-fixed position of the station, as a
            ``(3, 1)`` array in metres.  Passed to TudatPy's
            ``TimeScaleConverter`` for the UT1 reductions.  Defaults to the
            geocentre, which is what the previous implementation used for every
            station.
        allow_pre_1925_shift: Whether to apply the ``+12 h`` astronomical-day
            correction.  Disable only for sources that are known to have already
            converted their epochs to the civil convention.

    Returns:
        Array of epochs as TDB seconds since J2000, same length as ``epochs``.
        Nulls in the input propagate as NaN.

    Raises:
        ValueError: If a local mean time is requested without any way to
            determine the meridian, or if a zone label is missing its offset.

    Examples:
        >>> import numpy as np
        >>> from orbitdet.preprocessing.time_scales import to_tdb_seconds_since_j2000
        >>> epochs = np.array([0.0])
        >>> out = to_tdb_seconds_since_j2000(epochs, "UTC")
        >>> bool(out[0] != 0.0)  # UTC is offset from TDB
        True
    """
    values = _as_float_array(epochs)
    resolution = scale if isinstance(scale, TimescaleResolution) else resolve_timescale(scale)

    offsets = np.zeros_like(values)

    # 1. Astronomical day: every pre-1925 epoch, whatever the timescale.
    if allow_pre_1925_shift:
        pre_1925 = np.isfinite(values) & (values < ASTRONOMICAL_DAY_SWITCH_EPOCH)
        if pre_1925.any():
            offsets = np.where(pre_1925, ASTRONOMICAL_DAY_OFFSET, offsets)
            logger.info(
                "Applied +12 h astronomical-day correction to %d epoch(s) before 1925-01-01.",
                int(pre_1925.sum()),
            )

    # 2. Meridian / zone offsets.
    if resolution.scale in (InputTimeScale.LOCAL_MEAN_TIME, InputTimeScale.LOCAL_SIDEREAL_TIME):
        offsets += _local_time_offset(
            resolution,
            observatory_code=observatory_code,
            station_longitude_radians=station_longitude_radians,
        )
        if resolution.scale is InputTimeScale.LOCAL_SIDEREAL_TIME:
            logger.warning(
                "Sidereal timescale %r treated as local mean solar time. The "
                "sidereal/solar difference is far below historical measurement "
                "noise for photographic astrometry.",
                resolution.raw_label,
            )
    elif resolution.scale is InputTimeScale.STANDARD_ZONE:
        offsets += resolution.zone_offset_seconds

    adjusted = values + offsets

    # 3. Timescale conversion to TDB.
    return _convert_to_tdb(adjusted, resolution, station_earth_fixed_position)


def _local_time_offset(
    resolution: TimescaleResolution,
    *,
    observatory_code: int | str | None,
    station_longitude_radians: float | None,
) -> float:
    """Resolve the constant UT1 offset for a local mean time.

    An explicit station longitude takes precedence over a named meridian, so a
    caller that knows the true station position can override the meridian the
    source metadata names.  Note that ``0.0`` is a valid longitude (Greenwich),
    so the check is against ``None`` rather than truthiness.
    """
    if station_longitude_radians is not None:
        return longitude_to_time_offset(station_longitude_radians)

    if resolution.meridian is not None:
        return meridian_label_offset(resolution.meridian)

    if observatory_code is not None:
        return meridian_offset_for_station(observatory_code)

    raise ValueError(
        f"Timescale {resolution.raw_label!r} is a local mean time, but no meridian "
        f"is named and neither 'station_longitude_radians' nor 'observatory_code' "
        f"was supplied, so the offset from UT1 cannot be determined."
    )


def _convert_to_tdb(
    adjusted: np.ndarray,
    resolution: TimescaleResolution,
    station_earth_fixed_position: np.ndarray | None,
) -> np.ndarray:
    """Convert UT1-based epochs to TDB, per timescale family."""
    out = np.full_like(adjusted, np.nan)
    finite = np.isfinite(adjusted)
    if not finite.any():
        return out

    values = adjusted[finite]

    if resolution.scale in (InputTimeScale.LOCAL_MEAN_TIME, InputTimeScale.LOCAL_SIDEREAL_TIME):
        # Meridian offset was applied above, so what remains is UT1.
        out[finite] = _convert_native(values, InputTimeScale.UT1, station_earth_fixed_position)
    elif resolution.scale is InputTimeScale.STANDARD_ZONE:
        out[finite] = _convert_native(values, InputTimeScale.UT1, station_earth_fixed_position)
    elif resolution.scale is InputTimeScale.TCB:
        # TCB -> TDB is a pure scale ratio; TCB epochs already share the TDB
        # origin, so no further shift is needed.
        out[finite] = np.array([time_rep.TCB_to_TDB(float(value)) for value in values])
    elif resolution.scale is InputTimeScale.TCG:
        # TCG -> TT (IAU 2006), then TT -> TDB.
        out[finite] = _convert_native(
            np.array([time_rep.TCG_to_TT(float(value)) for value in values]),
            InputTimeScale.TT,
            station_earth_fixed_position,
        )
    elif resolution.scale is InputTimeScale.UNKNOWN:
        # resolve_timescale already warned; fall back to UTC so ingestion
        # continues and the assumption is visible in the log.
        out[finite] = _convert_native(values, InputTimeScale.UTC, station_earth_fixed_position)
    else:
        out[finite] = _convert_native(values, resolution.scale, station_earth_fixed_position)

    return out


def _convert_native(
    values: np.ndarray,
    scale: InputTimeScale,
    station_earth_fixed_position: np.ndarray | None,
) -> np.ndarray:
    """Convert epochs in a natively supported timescale to TDB.

    TudatPy's ``convert_time`` accepts scalars only, so this loops.  The loop is
    ~1 us per epoch (~17 ms for 15 000 epochs), which is negligible next to the
    frame conversions.

    UT1 epochs before 1961 are handled by :func:`_convert_ut1_to_tdb`, because
    TudatPy's leap-second table does not reach that far back.
    """
    if scale is InputTimeScale.TDB:
        return values.copy()

    if scale is InputTimeScale.UT1:
        return _convert_ut1_to_tdb(values, station_earth_fixed_position)

    tudat_scale = TDB_TIME_SCALES[scale]
    converter = time_rep.default_time_scale_converter()

    if station_earth_fixed_position is None:
        position = np.zeros((3, 1))
    else:
        position = np.asarray(station_earth_fixed_position, dtype=float).reshape(3, 1)

    return np.array(
        [
            converter.convert_time(
                tudat_scale, time_rep.TimeScales.tdb_scale, float(value), position
            )
            for value in values
        ]
    )


#: First epoch at which TudatPy can convert UT1, as TDB seconds since J2000
#: (1961-01-01).  Before this the underlying leap-second table has no entry and
#: ``convert_time`` raises ``RuntimeError``.  Verified by bisection against the
#: installed build.
_TUDAT_UT1_SUPPORT_EPOCH = time_rep.iso_string_to_epoch("1961-01-01T00:00:00")


def _convert_ut1_to_tdb(
    values: np.ndarray,
    station_earth_fixed_position: np.ndarray | None,
) -> np.ndarray:
    """Convert UT1 epochs to TDB, including epochs before TudatPy supports.

    TudatPy converts UT1 through its leap-second table, which starts in 1961.
    For earlier epochs this falls back to the standard identity

    .. math::

        \\mathrm{TT} = \\mathrm{UT1} + \\Delta T

    with :math:`\\Delta T` from :func:`delta_t_seconds`, and then converts
    TT to TDB normally.  Away from the leap-second era ``UTC`` and ``UT1`` agree
    to well under a second, so this reproduces the intended result rather than
    approximating around it.

    The accuracy is limited by the :math:`\\Delta T` model - a few tenths of a
    second at worst for the years covered - which is far below the precision of
    the historical observations that need it.
    """
    out = np.full_like(values, np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        return out

    supported = finite & (values >= _TUDAT_UT1_SUPPORT_EPOCH)
    historical = finite & (values < _TUDAT_UT1_SUPPORT_EPOCH)

    if supported.any():
        out[supported] = _convert_native_supported(
            values[supported], InputTimeScale.UT1, station_earth_fixed_position
        )

    if historical.any():
        epochs = values[historical]
        logger.info(
            "Converting %d UT1 epoch(s) before 1961 through the Delta T model "
            "(TudatPy's leap-second table does not reach that far back).",
            int(historical.sum()),
        )
        tt_epochs = epochs + delta_t_seconds(epochs)
        out[historical] = _convert_native_supported(
            tt_epochs, InputTimeScale.TT, station_earth_fixed_position
        )

    return out


def _convert_native_supported(
    values: np.ndarray,
    scale: InputTimeScale,
    station_earth_fixed_position: np.ndarray | None,
) -> np.ndarray:
    """Convert epochs in a scale TudatPy can handle, without further fallbacks."""
    if scale is InputTimeScale.TDB:
        return values.copy()

    tudat_scale = TDB_TIME_SCALES[scale]
    converter = time_rep.default_time_scale_converter()

    if station_earth_fixed_position is None:
        position = np.zeros((3, 1))
    else:
        position = np.asarray(station_earth_fixed_position, dtype=float).reshape(3, 1)

    return np.array(
        [
            converter.convert_time(
                tudat_scale, time_rep.TimeScales.tdb_scale, float(value), position
            )
            for value in values
        ]
    )


# ---------------------------------------------------------------------------
# Delta T
# ---------------------------------------------------------------------------

#: Julian date of the J2000 epoch, for converting to decimal years.
_JD_J2000 = 2451545.0


def delta_t_seconds(epochs: np.ndarray) -> np.ndarray:
    """Return :math:`\\Delta T = \\mathrm{TT} - \\mathrm{UT1}` in seconds.

    Uses the Espenak & Meeus (2006) polynomial expressions for 1900-1975, which
    are the standard reference for historical epochs and agree with the
    published values to a few tenths of a second.  Outside that range the
    nearest available expression is used, and a warning is logged once.

    Neither TudatPy nor SPICE is usable here: TudatPy's leap-second table begins
    in 1961, and SPICE's ``DELTET`` clamps every epoch before 1972 to a constant
    ``+41.184 s``, which is wrong by over 40 seconds at 1900.

    TODO: This was fixed in a later version of TudatPy, but was released together with
    a few other new functionalities that would require porting significant parts of OrbitDet.
    For the thesis, we will keep this implementation, but it should be removed once the TudatPy
    version is updated.

    Args:
        epochs: Epochs as seconds since J2000 (nominally UT1).

    Returns:
        :math:`\\Delta T` in seconds, same shape as ``epochs``.

    References:
        Espenak, F. & Meeus, J. (2006), "Five Millennium Canon of Solar
        Eclipses", NASA/TP-2006-214141, polynomial expressions for
        :math:`\\Delta T`.
    """
    years = _decimal_years(epochs)
    out = np.empty_like(years)

    # 1900-1920
    mask = (years >= 1900) & (years < 1920)
    t = years[mask] - 1900
    out[mask] = -2.79 + 1.494119 * t - 0.0598939 * t**2 + 0.0061966 * t**3 - 0.000197 * t**4

    # 1920-1941
    mask = (years >= 1920) & (years < 1941)
    t = years[mask] - 1920
    out[mask] = 21.20 + 0.84493 * t - 0.076100 * t**2 + 0.0020936 * t**3

    # 1941-1961
    mask = (years >= 1941) & (years < 1961)
    t = years[mask] - 1950
    out[mask] = 29.07 + 0.407 * t - t**2 / 233 + t**3 / 2547

    # 1961-1986 (covers the tail of the range we need, and keeps the model
    # continuous where TudatPy takes over).
    mask = (years >= 1961) & (years < 1986)
    t = years[mask] - 1975
    out[mask] = 45.45 + 1.067 * t - t**2 / 260 - t**3 / 718

    # Outside the modelled range: clamp to the nearest endpoint expression and
    # warn, rather than silently returning a plausible-looking wrong number.
    outside = (years < 1900) | (years >= 1986)
    if outside.any():
        logger.warning(
            "Delta T requested for %d epoch(s) outside the modelled 1900-1986 "
            "range (year %.1f to %.1f); clamping to the nearest endpoint.",
            int(outside.sum()),
            float(years[outside].min()),
            float(years[outside].max()),
        )
        clamped = np.clip(years[outside], 1900.0, 1985.999)
        out[outside] = delta_t_seconds((clamped - 2000.0) * 365.25 * SECONDS_PER_DAY)

    return out


def _decimal_years(epochs: np.ndarray) -> np.ndarray:
    """Convert seconds since J2000 to decimal years."""
    jd = epochs / SECONDS_PER_DAY + _JD_J2000
    return 2000.0 + (jd - _JD_J2000) / 365.25


def _as_float_array(epochs: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    """Coerce an epoch container to a float64 array."""
    if isinstance(epochs, pd.Series):
        return epochs.to_numpy(dtype=float)
    return np.asarray(epochs, dtype=float)

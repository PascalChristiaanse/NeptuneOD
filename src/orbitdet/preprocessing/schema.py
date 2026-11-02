"""The standard observation dataframe contract.

Every observation source (NSDB, MAST, the Gaia archive, Voyager tables, ...) is
normalised by :mod:`orbitdet.preprocessing` into a single :class:`pandas.DataFrame`
that satisfies the schema defined here.  The :mod:`orbitdet.observations`
factories consume *only* this dataframe and never see a raw source file.

Units
-----
**All angular quantities are radians.**  Radians are the internal unit of
TudatPy and the pipeline does not deviate from it anywhere.  Source units
(degrees, hours, arcseconds, seconds of time) are converted on ingest, and the
original unit is preserved in the ``first_unit`` / ``second_unit`` provenance
columns so the conversion remains auditable.

Epochs
------
``epoch_tdb`` is **TDB seconds since the J2000 epoch**, matching
``tudatpy.astro.time_reference_epochs``.  Timescale conversion (Washington mean
time, GMT, UTC, UT1, TCB, ET, ...) happens in
:mod:`orbitdet.preprocessing.time_scales`; this schema only records the result.

Reference frames
----------------
Two independent pieces of information describe the frame of the incoming
coordinates, and they must not be conflated:

``original_equinox``
    The *orientation* of the coordinate axes - ``"J2000"``, ``"date"``,
    ``"B1950"``, ``"J2015.0"``, ``"1899.0"``, ...  Handled by
    :mod:`orbitdet.preprocessing.precession`.

``original_reference_frame``
    The *kind* of coordinate - ``"astrometric"`` (light-time + deflection
    corrected, no aberration), ``"apparent"`` (aberration included) or
    ``"bcrs"``.  Handled by :mod:`orbitdet.preprocessing.aberration`.

A source may be astrometric in the mean equinox of date, or apparent in J2000;
the two axes are orthogonal.

Reduction catalogues
--------------------
``original_reduction`` records which star catalogue the observation was reduced
against (``UCAC2``, ``Tycho-2``, ``Gaia DR2``, ...).  This is required to tie
the coordinates to Gaia-CRF3 / ICRF, and it is *not* optional: the tie is
coordinate-dependent, and for photographic data from the early 20th century the
stellar proper motion between the catalogue epoch and the observation epoch can
reach several arcseconds - orders of magnitude larger than the mas-level frame
bias.  ``catalogue_epoch``, ``pm_first`` and ``pm_second`` carry the data needed
to apply that correction.

Examples
--------
Build an empty, correctly typed frame and populate it::

    >>> from orbitdet.preprocessing import empty_frame, verify
    >>> frame = empty_frame()
    >>> frame.loc[0] = {
    ...     "observed_object": "Triton",
    ...     "observatory_code": "673",
    ...     "data_source": "nsdb",
    ...     "source_set_id": "nm0001",
    ...     "reference": "(2001) Communicated to NSDC by W. M. Owen Jr.",
    ...     "observation_type": "absolute",
    ...     "coordinate_kind": "ra_dec",
    ...     "frame_center": "topocentric",
    ...     "original_equinox": "J2000",
    ...     "original_reference_frame": "astrometric",
    ...     "original_reduction": "none",
    ...     "receptor_type": "ccd",
    ...     "epoch_tdb": 44492331.99169189,
    ...     "first_value": 2.0,
    ...     "second_value": -0.3,
    ... }
    >>> verify(frame)  # raises ObservationFrameError if the contract is broken

See Also
--------
orbitdet.preprocessing.time_formatting, orbitdet.preprocessing.time_scales,
orbitdet.preprocessing.precession, orbitdet.preprocessing.catalogues,
orbitdet.preprocessing.aberration, orbitdet.preprocessing.frames
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "ObservationFrameError",
    "StandardObservationFrame",
    "ObservationType",
    "CoordinateKind",
    "FrameCenter",
    "ReferenceFrame",
    "ReceptorType",
    "ReductionCatalogue",
    "DataSource",
    "ColumnSpec",
    "STANDARD_DATAFRAME_COLUMNS",
    "COLUMN_SPECS",
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "ANGLE_COLUMNS",
    "NOT_GIVEN",
    "NOT_GIVEN_COLUMNS",
    "ENUM_COLUMNS",
    "verify",
    "empty_frame",
    "build_frame",
    "describe",
]

TWO_PI = 2.0 * np.pi
HALF_PI = 0.5 * np.pi

#: Value marking a measurement that the source did not record.
#:
#: Some sources record a component that was not measured as a sentinel rather
#: than omitting it - NSDB writes ``(999 if not given)`` in the Format section
#: and then stores ``999``, ``999.9`` or ``999.99`` depending on the column's
#: declared precision.  A relative measurement can therefore be *one-dimensional*:
#: a position angle with no separation, or a separation with no position angle.
#:
#: The sentinel must be a value no physical measurement can take, because a
#: consumer cannot distinguish it from real data by looking at the number.  A
#: position angle in radians lies in ``[0, 2*pi)``, so ``9999.0`` is unambiguous
#: and comfortably outside every range in :data:`_RANGE_RULES`.  Keeping the
#: marker rather than replacing it with null is deliberate: the row still
#: describes a real observation of the *other* component, so dropping it or
#: nulling the column would discard a measurement that was actually made.
#:
#: See :func:`is_not_given` and :func:`apply_not_given`.
NOT_GIVEN: float = 9999.0

#: How close a value must be to a declared sentinel to count as that sentinel.
#:
#: A source declares a sentinel as an integer but may store it at the column's
#: own precision - NSDB declares ``999`` and writes ``999``, ``999.9`` or
#: ``999.99``.  One unit covers every such form while staying far below the
#: nearest legitimate value, because a sentinel always sits outside its column's
#: physical range: a position angle is at most 360 as against a declared 999, an
#: arcminute field at most 60 as against a declared 99.
_SENTINEL_TOLERANCE = 1.0

#: Slack applied to angular range checks, to absorb floating-point round-off in
#: values that sit exactly on a boundary (e.g. a declination of exactly +/-pi/2).
_RANGE_TOLERANCE = 1e-9


class ObservationFrameError(ValueError):
    """Raised when a dataframe does not satisfy the standard observation schema.

    The exception message lists *every* problem found, not just the first, so a
    single call is enough to fix a malformed frame.
    """


#: Type alias for a dataframe that satisfies the standard observation schema.
#:
#: This is a plain :class:`pandas.DataFrame` - the alias exists purely to make
#: function signatures self-documenting, and carries no runtime behaviour.  Use
#: :func:`verify` to check that a frame actually conforms.
StandardObservationFrame = pd.DataFrame


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------


class ObservationType(StrEnum):
    """Whether the measurement is absolute or relative to another body."""

    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class CoordinateKind(StrEnum):
    """How the two observation values encode a sky position.

    ``RA_DEC``
        ``first_value`` is right ascension, ``second_value`` is declination.
    ``POSITION_ANGLE_SEPARATION``
        ``first_value`` is the position angle, ``second_value`` the separation.
    ``CARTESIAN_XY``
        ``first_value`` / ``second_value`` are tangent-plane standard
        coordinates (gnomonic X/Y) about the reference body.
    """

    RA_DEC = "ra_dec"
    POSITION_ANGLE_SEPARATION = "position_angle_separation"
    CARTESIAN_XY = "cartesian_xy"


class FrameCenter(StrEnum):
    """Origin of the reference frame in which the observation was made."""

    TOPOCENTRIC = "topocentric"
    GEOCENTRIC = "geocentric"
    SPACECRAFT = "spacecraft"
    BARYCENTRIC = "barycentric"
    HELIOCENTRIC = "heliocentric"
    UNKNOWN = "unknown"


class ReferenceFrame(StrEnum):
    """Kind of coordinate, independent of the equinox orientation.

    ``ASTROMETRIC``
        Light-time and light-deflection corrected, but *no* stellar aberration.
    ``APPARENT``
        Stellar aberration included; must be removed before comparison with a
        geometric model.
    ``BCRS``
        Barycentric celestial reference system coordinates.
    """

    ASTROMETRIC = "astrometric"
    APPARENT = "apparent"
    BCRS = "bcrs"
    UNKNOWN = "unknown"


class ReceptorType(StrEnum):
    """Detector that produced the measurement."""

    CCD = "ccd"
    PHOTOGRAPHIC = "photographic"
    VISUAL = "visual"
    MICROMETRIC = "micrometric"
    MERIDIAN_TRANSIT_CIRCLE = "meridian_transit_circle"
    VIDICON = "vidicon"
    UNKNOWN = "unknown"


class ReductionCatalogue(StrEnum):
    """Star catalogue the observation was reduced against.

    ``NONE`` means no catalogue reduction was applied (or the source states "no
    information"), so no tie correction is possible or needed.  ``UNKNOWN``
    means the source named a catalogue that is not in this vocabulary; such rows
    are reported by :func:`verify` so the mapping can be extended rather than
    silently skipped.
    """

    NONE = "none"
    UCAC2 = "ucac2"
    UCAC4 = "ucac4"
    TYCHO2 = "tycho2"
    GAIA_DR1 = "gaia_dr1"
    GAIA_DR2 = "gaia_dr2"
    SAO = "sao"
    AST = "ast"
    GIVEN_IN_DATA = "given_in_data"
    UNKNOWN = "unknown"


class DataSource(StrEnum):
    """Provider the observation was acquired from."""

    NSDB = "nsdb"
    MAST = "mast"
    GAIA_ARCHIVE = "gaia_archive"
    VOYAGER = "voyager"
    MANUAL = "manual"
    SIMULATED = "simulated"


# ---------------------------------------------------------------------------
# Column specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnSpec:
    """Description of a single column in the standard observation dataframe.

    Attributes:
        name: Column name.
        dtype: Pandas dtype the column must have (``"float64"`` or ``"object"``).
        required: Whether the column must be present and non-null.
        description: Human-readable meaning of the column.
        unit: Physical unit, or ``None`` for dimensionless / textual columns.
    """

    name: str
    dtype: str
    required: bool
    description: str
    unit: str | None = None


#: The full standard observation dataframe schema, in canonical column order.
#:
#: This is the single source of truth for the dataframe contract: every column
#: name, dtype, unit and requirement is declared here, and :func:`verify`
#: enforces it.  Adding a column means adding a :class:`ColumnSpec` here and
#: nothing else.
STANDARD_DATAFRAME_COLUMNS: tuple[ColumnSpec, ...] = (
    # -- Provenance ---------------------------------------------------------
    ColumnSpec(
        "observed_object",
        "object",
        True,
        "Name of the observed body, matching the SystemOfBodies entry.",
    ),
    ColumnSpec(
        "observatory_code",
        "object",
        True,
        "MPC observatory code (zero-padded, e.g. '083'), or the body name when no "
        "code exists (e.g. 'Voyager 2', 'Gaia').",
    ),
    ColumnSpec(
        "data_source",
        "object",
        True,
        "Provider the observation was acquired from.",
    ),
    ColumnSpec(
        "source_set_id",
        "object",
        True,
        "Identifier of the set within its source, e.g. 'nm0001' for NSDB. Empty "
        "string when the source has no per-set identifier.",
    ),
    ColumnSpec(
        "reference",
        "object",
        True,
        "Academic reference documenting the observations.",
    ),
    # -- Measurement semantics ---------------------------------------------
    ColumnSpec(
        "observation_type",
        "object",
        True,
        "Whether the measurement is absolute or relative to another body.",
    ),
    ColumnSpec(
        "coordinate_kind",
        "object",
        True,
        "How first_value / second_value encode the sky position.",
    ),
    ColumnSpec(
        "frame_center",
        "object",
        True,
        "Origin of the reference frame (topocentric, geocentric, ...).",
    ),
    ColumnSpec(
        "original_equinox",
        "object",
        True,
        "Orientation of the coordinate axes as recorded by the source, e.g. "
        "'J2000', 'date', 'B1950', 'J2015.0', '1899.0'.",
    ),
    ColumnSpec(
        "original_reference_frame",
        "object",
        True,
        "Kind of coordinate as recorded by the source: astrometric, apparent or bcrs.",
    ),
    ColumnSpec(
        "original_reduction",
        "object",
        True,
        "Star catalogue the observation was reduced against.",
    ),
    ColumnSpec(
        "receptor_type",
        "object",
        True,
        "Detector that produced the measurement.",
    ),
    # -- The measurement ----------------------------------------------------
    ColumnSpec(
        "epoch_tdb",
        "float64",
        True,
        "Observation epoch as TDB seconds since the J2000 epoch.",
        "s",
    ),
    ColumnSpec(
        "first_value",
        "float64",
        True,
        "First observation value: right ascension, position angle or X.",
        "rad",
    ),
    ColumnSpec(
        "second_value",
        "float64",
        True,
        "Second observation value: declination, separation or Y.",
        "rad",
    ),
    # -- Optional / nullable ------------------------------------------------
    ColumnSpec(
        "relative_to_object",
        "object",
        False,
        "Body the measurement is relative to. Null for absolute observations.",
    ),
    ColumnSpec(
        "first_unit",
        "object",
        False,
        "Unit of first_value as it appeared in the source, for provenance "
        "(e.g. 'deg', 'hour', 'arcsec', 'sec_of_time').",
    ),
    ColumnSpec(
        "second_unit",
        "object",
        False,
        "Unit of second_value as it appeared in the source, for provenance.",
    ),
    ColumnSpec(
        "catalogue_epoch",
        "float64",
        False,
        "Epoch of the catalogue position used for the reduction, as TDB seconds "
        "since J2000. Required to apply proper motion.",
        "s",
    ),
    ColumnSpec(
        "pm_first",
        "float64",
        False,
        "Proper motion in the first coordinate, in radians per second.",
        "rad/s",
    ),
    ColumnSpec(
        "pm_second",
        "float64",
        False,
        "Proper motion in the second coordinate, in radians per second.",
        "rad/s",
    ),
    ColumnSpec(
        "first_uncertainty",
        "float64",
        False,
        "One-sigma uncertainty on first_value.",
        "rad",
    ),
    ColumnSpec(
        "second_uncertainty",
        "float64",
        False,
        "One-sigma uncertainty on second_value.",
        "rad",
    ),
    ColumnSpec(
        "scan_angle",
        "float64",
        False,
        "Position angle of scan, measured north through east: the direction in "
        "which the detector swept while the measurement was taken. Present only "
        "for scanning instruments (Gaia), where it defines the frame the "
        "uncertainties are diagonal in.",
        "rad",
    ),
)

#: Alias for :data:`STANDARD_DATAFRAME_COLUMNS`, for call sites that prefer a
#: shorter name.
COLUMN_SPECS: tuple[ColumnSpec, ...] = STANDARD_DATAFRAME_COLUMNS

#: Names of the columns that must be present and non-null.
REQUIRED_COLUMNS: tuple[str, ...] = tuple(
    spec.name for spec in STANDARD_DATAFRAME_COLUMNS if spec.required
)

#: Names of the columns that may be absent or null.
OPTIONAL_COLUMNS: tuple[str, ...] = tuple(
    spec.name for spec in STANDARD_DATAFRAME_COLUMNS if not spec.required
)

#: Columns holding angles in radians.
ANGLE_COLUMNS: tuple[str, ...] = (
    "first_value",
    "second_value",
    "pm_first",
    "pm_second",
    "first_uncertainty",
    "second_uncertainty",
    "scan_angle",
)

#: Columns where :data:`NOT_GIVEN` is a legal value.
#:
#: Only the measurement columns: a source that omitted a measurement can say so
#: there, but a missing observatory code or epoch is a genuine gap and must stay
#: null so that it remains visible.
NOT_GIVEN_COLUMNS: tuple[str, ...] = ("first_value", "second_value")

#: Columns constrained to a controlled vocabulary, mapped to that vocabulary.
ENUM_COLUMNS: dict[str, type[StrEnum]] = {
    "data_source": DataSource,
    "observation_type": ObservationType,
    "coordinate_kind": CoordinateKind,
    "frame_center": FrameCenter,
    "original_reference_frame": ReferenceFrame,
    "original_reduction": ReductionCatalogue,
    "receptor_type": ReceptorType,
}

#: Inclusive value bounds per coordinate kind, applied to the measurement columns.
#:
#: Right ascension is accepted in either wrap convention - ``[0, 2*pi)`` or
#: ``[-pi, pi]`` - because both appear in the pipeline today.  The bound is
#: deliberately loose (``+/-2*pi``) so that its real job is catching unit
#: mistakes: a value in degrees (up to 360) fails immediately.
_RANGE_RULES: dict[CoordinateKind, tuple[tuple[str, float, float], ...]] = {
    CoordinateKind.RA_DEC: (
        ("first_value", -TWO_PI, TWO_PI),
        ("second_value", -HALF_PI, HALF_PI),
    ),
    CoordinateKind.POSITION_ANGLE_SEPARATION: (
        ("first_value", -TWO_PI, TWO_PI),
        ("second_value", 0.0, np.inf),
    ),
    CoordinateKind.CARTESIAN_XY: (),
}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def empty_frame() -> pd.DataFrame:
    """Return an empty dataframe with every standard column and the right dtypes.

    Useful as a template for building a frame row by row, and as the return
    value for a source that yielded no observations.

    Returns:
        A zero-row :class:`pandas.DataFrame` with the columns of
        :data:`STANDARD_DATAFRAME_COLUMNS` in canonical order.
    """
    return pd.DataFrame(
        {spec.name: pd.Series(dtype=spec.dtype) for spec in STANDARD_DATAFRAME_COLUMNS}
    )


def build_frame(records: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Build a standard dataframe from an iterable of column->value mappings.

    This is the preferred constructor for preprocessing code.  It guarantees the
    canonical column order and dtypes, and fills any column a record omits with
    null, so callers never have to spell out the full schema.

    Prefer this over assigning rows into :func:`empty_frame` one at a time:
    pandas coerces an all-null object column to ``float64`` on row assignment,
    which loses the declared dtype.

    Args:
        records: One mapping per observation.  Keys must be standard column
            names; unknown keys raise :class:`KeyError` so typos are caught
            immediately rather than silently dropped.

    Returns:
        A dataframe with the columns of :data:`STANDARD_DATAFRAME_COLUMNS` in
        canonical order.  Zero rows if ``records`` is empty.

    Raises:
        KeyError: If a record contains a key that is not a standard column.

    Examples:
        >>> frame = build_frame([
        ...     {"observed_object": "Triton", "epoch_tdb": 0.0,
        ...      "first_value": 1.0, "second_value": 0.0},
        ... ])
        >>> list(frame.columns)[:3]
        ['observed_object', 'observatory_code', 'data_source']
    """
    known = {spec.name for spec in STANDARD_DATAFRAME_COLUMNS}
    rows = list(records)

    for index, record in enumerate(rows):
        unknown = set(record) - known
        if unknown:
            raise KeyError(
                f"Record {index} contains key(s) that are not standard observation "
                f"columns: {', '.join(sorted(unknown))}. "
                f"Valid columns: {', '.join(sorted(known))}"
            )

    if not rows:
        return empty_frame()

    frame = pd.DataFrame(rows, columns=[spec.name for spec in STANDARD_DATAFRAME_COLUMNS])

    # Restore the declared dtype for every column.  ``astype`` on an all-null
    # column is a no-op for object and yields NaN for float64, which is what we
    # want; for populated columns it performs the real conversion.
    for spec in STANDARD_DATAFRAME_COLUMNS:
        frame[spec.name] = frame[spec.name].astype(spec.dtype)

    return frame


def is_not_given(values: Any) -> np.ndarray:
    """Return a boolean mask of the entries marked :data:`NOT_GIVEN`.

    Args:
        values: A column, array or scalar holding measurement values.

    Returns:
        A boolean array, ``True`` where the value is the sentinel.  Nulls are
        ``False``: an absent value and an explicitly-not-given value are
        different things, and only the latter carries information.

    Examples:
        >>> is_not_given(np.array([1.0, NOT_GIVEN, np.nan])).tolist()
        [False, True, False]
    """
    return np.isclose(
        np.asarray(pd.to_numeric(values, errors="coerce"), dtype="float64"),
        NOT_GIVEN,
        rtol=0.0,
        atol=_SENTINEL_TOLERANCE,
    )


def apply_not_given(
    values: Any,
    sentinels: tuple[float, ...] | list[float],
    scale: float = 1.0,
) -> np.ndarray:
    """Scale ``values`` to radians, marking the entries that were not measured.

    Args:
        values: The raw measured values, in the source's own units.
        sentinels: Values the source uses to mean "not given".  These are taken
            from the source's own declaration rather than a fixed list, because a
            sentinel is an ordinary-looking number in the units of its own column
            - NSDB declares ``999`` degrees for a position angle, ``99`` for an
            arcminute field and ``-99.99`` arcseconds for a separation - so
            testing one column's sentinel against another column would silently
            corrupt real measurements.  A position angle of ``99.99`` degrees is
            a legitimate measurement and must survive.
        scale: Factor converting ``values`` to radians.

    Returns:
        A ``float64`` array of radians, holding :data:`NOT_GIVEN` where the
        source recorded nothing.  Nulls count as not given.

    Examples:
        >>> import numpy as np
        >>> apply_not_given(np.array([0.5, 999.0]), (999.0,)).tolist()
        [0.5, 9999.0]
    """
    numeric = np.asarray(pd.to_numeric(values, errors="coerce"), dtype="float64")
    missing = ~np.isfinite(numeric)
    for sentinel in sentinels:
        missing |= np.isclose(numeric, sentinel, rtol=0.0, atol=_SENTINEL_TOLERANCE)

    out = numeric.copy()
    out[~missing] *= scale
    out[missing] = NOT_GIVEN
    return out


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify(
    dataframe: pd.DataFrame,
    *,
    strict: bool = True,
    allow_empty: bool = True,
) -> list[str]:
    """Check a dataframe against the standard observation schema.

    Every problem found is collected before returning, so a single call reports
    all of them rather than stopping at the first.

    Args:
        dataframe: The dataframe to check.
        strict: If ``True`` (default), raise :class:`ObservationFrameError` when
            any problem is found.  If ``False``, return the problems instead.
        allow_empty: If ``True`` (default), a zero-row frame is valid as long as
            its columns and dtypes are correct.  If ``False``, an empty frame is
            reported as a problem.

    Returns:
        A list of human-readable problem descriptions. Empty when the frame is
        valid, and always empty when ``strict=True`` and no exception is raised.

    Raises:
        TypeError: If ``dataframe`` is not a :class:`pandas.DataFrame`.
        ObservationFrameError: If ``strict=True`` and the frame is invalid.

    Examples:
        >>> verify(empty_frame())
        []
        >>> verify(pd.DataFrame({"observed_object": ["Triton"]}), strict=False)
        ['missing required column(s): observatory_code, data_source, ...']
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(dataframe).__name__}")

    problems: list[str] = []

    # 1. Required columns must be present.
    missing = [name for name in REQUIRED_COLUMNS if name not in dataframe.columns]
    if missing:
        problems.append(f"missing required column(s): {', '.join(missing)}")

    # 2. Present columns must have the declared dtype.
    for spec in STANDARD_DATAFRAME_COLUMNS:
        if spec.name not in dataframe.columns:
            continue
        problem = _check_dtype(dataframe[spec.name], spec)
        if problem is not None:
            problems.append(problem)

    # 3. Emptiness.
    if dataframe.empty:
        if not allow_empty:
            problems.append("dataframe contains no observations")
        return _finish(problems, strict)

    # 4. Required columns must not contain nulls.
    for name in REQUIRED_COLUMNS:
        if name not in dataframe.columns:
            continue
        null_count = int(dataframe[name].isna().sum())
        if null_count:
            problems.append(f"column '{name}' is required but has {null_count} null value(s)")

    # 5. Controlled vocabularies.
    problems.extend(_check_enums(dataframe))

    # 6. The not-given marker may only appear where the schema allows it.
    problems.extend(_check_not_given(dataframe))

    # 7. Angular ranges, per coordinate kind.
    problems.extend(_check_ranges(dataframe))

    return _finish(problems, strict)


def _finish(problems: list[str], strict: bool) -> list[str]:
    """Raise or return the collected problems."""
    if not problems:
        return []
    if strict:
        bullet_list = "\n".join(f"  - {problem}" for problem in problems)
        raise ObservationFrameError(
            f"Dataframe does not satisfy the standard observation schema "
            f"({len(problems)} problem(s)):\n{bullet_list}"
        )
    return problems


def _check_not_given(dataframe: pd.DataFrame) -> list[str]:
    """Return problems for :data:`NOT_GIVEN` appearing in a column that forbids it."""
    problems: list[str] = []
    allowed = set(NOT_GIVEN_COLUMNS)
    for column in ANGLE_COLUMNS:
        if column in allowed or column not in dataframe.columns:
            continue
        count = int(is_not_given(dataframe[column]).sum())
        if count:
            problems.append(
                f"column '{column}' holds {count} not-given sentinel value(s) "
                f"({NOT_GIVEN:g}), which is only allowed in "
                f"{', '.join(NOT_GIVEN_COLUMNS)}"
            )
    return problems


def _check_dtype(series: pd.Series, spec: ColumnSpec) -> str | None:
    """Return a problem description if ``series`` has the wrong dtype.

    A column that is entirely null is accepted regardless of its dtype.  Pandas
    silently coerces an all-null object column to ``float64`` when rows are
    assigned one at a time, so rejecting that would make the schema impossible
    to satisfy for optional columns that a particular source does not populate.
    Required columns are still caught by the null check in :func:`verify`.
    """
    if series.isna().all():
        return None

    if spec.dtype == "float64":
        if not pd.api.types.is_numeric_dtype(series):
            return (
                f"column '{spec.name}' must be numeric ({spec.dtype}), got dtype '{series.dtype}'"
            )
        if pd.api.types.is_bool_dtype(series):
            return f"column '{spec.name}' must be numeric ({spec.dtype}), got boolean"
        return None

    # Textual column: accept object or pandas' dedicated string dtype.
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return None
    return f"column '{spec.name}' must be string-like ({spec.dtype}), got dtype '{series.dtype}'"


def _check_enums(dataframe: pd.DataFrame) -> list[str]:
    """Return problems for values outside a column's controlled vocabulary."""
    problems: list[str] = []
    for column, vocabulary in ENUM_COLUMNS.items():
        if column not in dataframe.columns:
            continue
        allowed = {member.value for member in vocabulary}
        present = set(dataframe[column].dropna().astype(str).unique())
        unexpected = sorted(present - allowed)
        if unexpected:
            problems.append(
                f"column '{column}' contains value(s) outside {vocabulary.__name__}: "
                f"{', '.join(repr(value) for value in unexpected)} "
                f"(allowed: {', '.join(sorted(allowed))})"
            )
    return problems


def _check_ranges(dataframe: pd.DataFrame) -> list[str]:
    """Return problems for measurement values outside their physical range."""
    if "coordinate_kind" not in dataframe.columns:
        return []

    problems: list[str] = []
    for kind, rules in _RANGE_RULES.items():
        if not rules:
            continue
        mask = dataframe["coordinate_kind"].astype(str) == kind.value
        if not mask.any():
            continue
        for column, lower, upper in rules:
            if column not in dataframe.columns:
                continue
            values = pd.to_numeric(dataframe.loc[mask, column], errors="coerce")
            # The sentinel is deliberately outside every range, so it must be
            # excluded before the bounds are applied - otherwise every source
            # that records a one-dimensional measurement would be reported as
            # holding a corrupt one.
            sentinel = is_not_given(values)
            finite = values[np.isfinite(values) & ~sentinel]
            if finite.empty:
                continue
            below = int((finite < lower - _RANGE_TOLERANCE).sum())
            above = int((finite > upper + _RANGE_TOLERANCE).sum())
            if below or above:
                problems.append(
                    f"column '{column}' is out of range for coordinate_kind "
                    f"'{kind.value}': expected [{lower:g}, {upper:g}] rad, "
                    f"found {below} value(s) below and {above} value(s) above "
                    f"(min={finite.min():.6g}, max={finite.max():.6g}) - "
                    f"check that the values are in radians, not degrees"
                )
    return problems


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def describe() -> pd.DataFrame:
    """Return the schema as a readable table.

    Returns:
        A dataframe with one row per column, listing its dtype, whether it is
        required, its unit and its description.
    """
    return pd.DataFrame(
        [
            {
                "column": spec.name,
                "dtype": spec.dtype,
                "required": spec.required,
                "unit": spec.unit or "",
                "description": spec.description,
            }
            for spec in STANDARD_DATAFRAME_COLUMNS
        ]
    )

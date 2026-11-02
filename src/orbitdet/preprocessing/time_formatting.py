"""Observation-epoch *formatting*: source columns to seconds since J2000.

This module provides methods for formatting common observation-epoch encodings into a single, 
uniform representation: seconds since the J2000.0 epoch, 2000-01-01T12:00:00, as TudatPy renders it.
Conversions are routed through :class:`tudatpy.astro.time_representation.DateTime` .

Design
------
Each source encoding is reduced to an ISO-8601 string and handed to 
``DateTime.from_iso_string(...).to_epoch()``, 
with the numeric forms going through ``DateTime.from_julian_day`` and
``DateTime.from_modified_julian_day``.  No epoch arithmetic is reimplemented
here.

What the module does own is the part ``DateTime`` cannot do - deciding **which
columns carry the epoch and in what form** (:func:`detect_epoch_layout`), and
reducing each source's quirks to something ``DateTime`` accepts.  There is no
``SystemOfBodies`` and no timescale here.

Epoch convention
----------------
One day is 86400 seconds, uniformly, so ``JD 2451545.0`` is epoch zero and
``2012-06-30T11:11:37`` is a fraction ``40297/86400`` through its day even though
that day carried a leap second.  This is TudatPy's convention, and it is the
convention here because the conversion *is* TudatPy's.

``astropy`` is not used for epochs anywhere in this module.  Its UTC Julian Date
is leap-second-aware and compresses the 86401-second day into a 1.0-day span, so
seconds-since-J2000 derived from it are displaced by up to a second on
leap-second days (~0.47 s on 2012-06-30) - comparable to the precision of the
historical data itself.

Supported input layouts
-----------------------
Julian Date
    A single column labelled ``JD``, ``Julian date``, ``Julian day``, etc.
Modified Julian Date
    A single column labelled ``MJD`` or ``Modified Julian date``.  Detected
    *before* the Julian-date search, because ``mjd`` would otherwise be claimed
    by it.
Calendar parts
    Separate ``year`` / ``month`` / ``day`` columns, with optional
    ``hour`` / ``minute`` / ``second``.  A fractional day carries into the time
    of day.  Right-ascension and declination components are excluded, since
    ``hour`` and ``second`` are angular units there.
ISO-8601 strings
    A column already holding ``YYYY-MM-DDThh:mm:ss`` text.  A bare date is
    padded with ``T00:00:00``, and a trailing ``Z`` or numeric UTC offset is
    resolved to the equivalent UTC instant.

Detection runs MJD, then JD, then a column whose label declares a Julian-Date
reference, then ISO, then calendar.

Source quirks
-------------
* A **fractional day** and the equivalent ``hour`` / ``minute`` / ``second``
  components describe the same instant, so a source may use either.
* An **absent time column** means midnight; an absent ``year`` / ``month`` /
  ``day`` means no epoch at all.
* A **declared reference** in a column label, as in ``with respect to JD
  2440000.0``, is carried on the layout as ``offset_days`` and added before
  conversion.  It is ignored when the values are already absolute dates.
* ``hour = 24`` advances the day rather than failing.

Failure behaviour
-----------------
A missing ``year`` / ``month`` / ``day``, an unparseable ISO string, and a
non-finite or unrepresentable Julian Date all yield NaN for that row, with a
warning where a value was present but unusable.  An impossible calendar date
raises ``RuntimeError``, and a negative month or day raises ``ValueError``: a
date that does not exist means the source is being misread, and a silently
wrong epoch is indistinguishable from a wrong orbit afterwards.

Public API
----------
:class:`EpochLayout`, :func:`detect_epoch_layout`,
:func:`to_seconds_since_j2000`, :func:`seconds_since_j2000_to_julian_date`,
:func:`seconds_since_j2000_to_datetime`, and the constants
:data:`SECONDS_PER_DAY`, :data:`MJD_OFFSET_DAYS`, :data:`JD_J2000`,
:data:`EPOCH_COLUMN`.

See Also
--------
orbitdet.preprocessing.time_scales : applies the clock to the count produced here.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from tudatpy.astro.time_representation import DateTime

from orbitdet.preprocessing.columns import find_column, numeric_series

logger = logging.getLogger(__name__)

__all__ = [
    "SECONDS_PER_DAY",
    "MJD_OFFSET_DAYS",
    "JD_J2000",
    "EPOCH_COLUMN",
    "EpochLayout",
    "detect_epoch_layout",
    "to_seconds_since_j2000",
    "seconds_since_j2000_to_julian_date",
    "seconds_since_j2000_to_datetime",
]

#: Seconds in a day.  Uniform: see the epoch-convention note above.
SECONDS_PER_DAY = 86400.0

#: MJD = JD - 2400000.5, a convenience offset with no astronomical meaning.
MJD_OFFSET_DAYS = 2400000.5

#: Julian Date of the J2000.0 epoch, 2000-01-01T12:00:00, as TudatPy renders
#: it.  Read from :class:`DateTime` rather than tabulated, so that the JD
#: boundary here is the same number ``DateTime.from_julian_day`` uses and the
#: two directions of the conversion invert one another exactly.
JD_J2000 = DateTime.from_epoch(0.0).to_julian_day()

#: Default name of the emitted seconds-since-J2000 column.
EPOCH_COLUMN = "epoch_j2000"

#: A Julian-Date reference declared *inside* an epoch column's label, as in
#: ``Measurement epoch with respect to JD 2440000.0`` or ``Gaia-centric epoch in
#: TCB relative to 2455197.5``.  Such a column stores days since that reference,
#: not an absolute Julian Date.
_EPOCH_REFERENCE_PATTERN = re.compile(
    r"(?:with\s+respect\s+to|relative\s+to|since|from|after)\s+(?:JD\s*)?"
    r"(?P<reference>\d{5,}(?:\.\d+)?)",
    re.IGNORECASE,
)

#: A value at or above this is already a Julian Date rather than an offset from
#: one.  Nothing astronomical is dated before JD 1,000,000 (year -4712 is JD 0),
#: whereas every offset form seen in practice is a count of days since a recent
#: reference and so is four digits or fewer.
_ABSOLUTE_JD_FLOOR = 1_000_000.0

#: Columns that name an angular component rather than a time component.
_ANGULAR_COMPONENT_EXCLUDES = [
    "right ascension",
    "alpha",
    "ra",
    "declination",
    "delta",
    "dec",
]


class EpochLayout:
    """Which columns carry the observation epoch, and in what form.

    Produced by :func:`detect_epoch_layout` so that a caller can log, validate or
    assert on how an epoch was interpreted rather than having to trust it.

    Attributes:
        kind: One of ``"jd"``, ``"mjd"``, ``"calendar"``, ``"iso"``.
        columns: The column labels involved, keyed by role.
        offset_days: Days to add to the stored value to reach an absolute Julian
            Date.  Zero for every layout except a numeric one whose label
            declares a reference, such as ``with respect to JD 2440000.0``.
    """

    __slots__ = ("kind", "columns", "offset_days")

    def __init__(
        self,
        kind: str,
        columns: dict[str, str | None],
        offset_days: float = 0.0,
    ) -> None:
        self.kind = kind
        self.columns = columns
        self.offset_days = offset_days

    def __repr__(self) -> str:
        roles = ", ".join(f"{k}={v!r}" for k, v in self.columns.items())
        if self.offset_days:
            return f"EpochLayout({self.kind!r}, {roles}, offset_days={self.offset_days!r})"
        return f"EpochLayout({self.kind!r}, {roles})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EpochLayout):
            return NotImplemented
        return (
            self.kind == other.kind
            and self.columns == other.columns
            and self.offset_days == other.offset_days
        )

    @property
    def is_calendar(self) -> bool:
        """Whether the epoch is stored as separate calendar components."""
        return self.kind == "calendar"


def _declared_epoch_column(dataframe: pd.DataFrame) -> str | None:
    """Find a numeric epoch column whose label declares a Julian-Date reference.

    Only such a column is returned, so that this cannot claim an unrelated
    column merely for containing the word ``epoch``: the reference number is
    what makes it a Julian-Date offset, and without one the column is left to
    the remaining detectors.

    Args:
        dataframe: The frame to inspect.

    Returns:
        The column label, or ``None``.
    """
    for name in dataframe.columns:
        if _EPOCH_REFERENCE_PATTERN.search(str(name)) is None:
            continue
        if not numeric_series(dataframe, name).notna().any():
            continue
        return str(name)
    return None


def _reference_offset_days(column_name: object, values: pd.Series) -> float:
    """Days to add to a numeric epoch column to reach an absolute Julian Date.

    A source may store an epoch as a *count* of days from a stated reference
    rather than a Julian Date, and says so inside the column label - for example
    ``Measurement epoch with respect to JD 2440000.0`` or ``Gaia-centric epoch
    in TCB relative to 2455197.5``.  The digits in the label are then the
    quantity the values are measured *from*, and the values are three- to
    four-digit numbers that look nothing like a Julian Date.

    This is detected rather than assumed because ignoring it is silently
    catastrophic: ``7967.96`` read as a Julian Date is July of 4713 BC, a
    six-and-a-half-thousand-year error that no downstream validation rejects.

    Args:
        column_name: The epoch column's label.
        values: The numeric epoch values, used to reject a label whose number is
            already an absolute date in its own right.

    Returns:
        The reference in days, or ``0.0`` when the label declares none.
    """
    match = _EPOCH_REFERENCE_PATTERN.search(str(column_name))
    if match is None:
        return 0.0

    reference = float(match.group("reference"))

    # A declared reference is only the base of an *offset* if the values are
    # well below it.  Some labels spell out a real, absolute Julian Date
    # (``JD 2451545.0``) and store absolute dates in the same breath; treating
    # those as offsets would subtract rather than add a base.  Compare on the
    # magnitude of the data, not just the label.
    finite = values.to_numpy(dtype="float64", na_value=np.nan)
    finite = finite[np.isfinite(finite)]
    if finite.size and float(np.nanmedian(finite)) >= _ABSOLUTE_JD_FLOOR:
        return 0.0

    return reference


def detect_epoch_layout(dataframe: pd.DataFrame) -> EpochLayout:
    """Determine how a frame encodes its observation epoch.

    Detection order matters and is deliberate:

    1. ``MJD`` before ``JD``, because the letters ``jd`` occur inside ``mjd``
       and a Julian-date search would otherwise claim a Modified Julian Date and
       misread it by 2400000.5 days.
    2. ``ISO`` before ``calendar``, because an ISO column is a calendar date in
       one column and reading it as ``year`` would find nothing.
    3. ``calendar`` last, as the fallback.

    A numeric epoch column may also declare a reference inside its own label,
    meaning it holds days *since* that reference rather than absolute dates.  Any
    such reference is carried on the returned layout as ``offset_days``, so the
    conversion can apply it and a caller can inspect it.

    Args:
        dataframe: The frame to inspect.

    Returns:
        The detected layout.

    Raises:
        ValueError: If no epoch encoding can be identified.  Raised rather than
            guessed, because a silently wrong epoch is indistinguishable from a
            wrong orbit afterwards.

    Examples:
        >>> import pandas as pd
        >>> detect_epoch_layout(pd.DataFrame({"MJD": [58000.0]})).kind
        'mjd'
        >>> detect_epoch_layout(pd.DataFrame({"JD": [2451545.0]})).kind
        'jd'
    """
    mjd_column = find_column(dataframe, "mjd")
    if mjd_column is None:
        mjd_column = find_column(dataframe, "modified", "julian")
    if mjd_column is not None:
        return EpochLayout(
            "mjd",
            {"mjd": mjd_column},
            offset_days=_reference_offset_days(mjd_column, numeric_series(dataframe, mjd_column)),
        )

    jd_column = find_column(dataframe, "jd", excludes=["mjd", "modified"])
    if jd_column is not None:
        return EpochLayout(
            "jd",
            {"jd": jd_column},
            offset_days=_reference_offset_days(jd_column, numeric_series(dataframe, jd_column)),
        )

    # A column can declare its Julian-Date reference without carrying the token
    # ``jd`` itself - ``Measurement epoch with respect to JD 2440000.0`` does,
    # but ``Gaia-centric epoch in TCB relative to 2455197.5`` does not, and the
    # finder above searches for the letters.  Catch those by looking for the
    # declaration itself, so an offset epoch is never mistaken for a calendar
    # or ISO one further down.
    declared = _declared_epoch_column(dataframe)
    if declared is not None:
        values = numeric_series(dataframe, declared)
        offset = _reference_offset_days(declared, values)
        if offset:
            finite = values.to_numpy(dtype="float64", na_value=np.nan)
            if np.isfinite(finite).any():
                return EpochLayout("jd", {"jd": declared}, offset_days=offset)

    iso_column = find_column(dataframe, "iso")
    if iso_column is not None:
        return EpochLayout("iso", {"iso": iso_column})

    year_column = find_column(dataframe, "year")
    month_column = find_column(dataframe, "month")
    day_column = find_column(dataframe, "day")
    if year_column is not None and month_column is not None and day_column is not None:
        return EpochLayout(
            "calendar",
            {
                "year": year_column,
                "month": month_column,
                "day": day_column,
                # Angular components must not be mistaken for time components:
                # "hour" and "second" name units of arc in an RA/Dec frame.
                "hour": find_column(dataframe, "hour", excludes=_ANGULAR_COMPONENT_EXCLUDES),
                "minute": find_column(dataframe, "minute", excludes=_ANGULAR_COMPONENT_EXCLUDES),
                "second": find_column(dataframe, "second", excludes=_ANGULAR_COMPONENT_EXCLUDES),
            },
        )

    raise ValueError(
        "Could not infer an observation epoch from the dataframe. Expected one of: "
        "a Julian Date column, a Modified Julian Date column, an ISO-8601 column, "
        "or year/month/day columns. "
        f"Available columns: {sorted(map(str, dataframe.columns))}"
    )


def _iso_string_from_calendar_components(
    year: float,
    month: float,
    day: float,
    seconds_of_day: float,
) -> str:
    """Render fractional calendar components as an ISO-8601 datetime string.

    ``DateTime`` cannot be handed a fractional day: its ``seconds`` field means
    seconds *within the minute* and must be below 60, and it has no field for a
    sub-day remainder at all.  A source that writes ``day = 9.520833`` therefore
    has to be reduced to the day *number* it falls inside plus the seconds that
    day fraction represents, and that running total has to be unpacked into
    hour, minute and second - which is what this does.  A dataset writing
    ``day = 9.5208`` and one writing ``day = 9`` with ``hour = 12`` produce the
    identical string, so the caller does not have to know which convention its
    source uses.

    Rendering to text rather than calling ``DateTime(...)`` keeps the conversion
    on one code path: the string is read back by ``DateTime.from_iso_string``
    and ``.to_epoch()``, the same route an ISO-8601 column takes.  The
    constructor and the parser were verified to agree to the last bit of the
    ``long double`` they carry, so this costs no precision.

    ``seconds_of_day`` is allowed to exceed a day, so a source that writes
    ``hour = 24`` (some do, meaning the following midnight) advances the day
    rather than failing.

    Args:
        year: Calendar year.
        month: Calendar month, 1-12.
        day: Day of month, possibly fractional.
        seconds_of_day: Seconds since the start of ``day``, which the day
            fraction of ``day`` has already been folded into.

    Returns:
        An ISO-8601 string such as ``"1989-09-19T12:00:00.000000000"``.

    Raises:
        ValueError: If ``month`` is below 1 or ``day`` is below 1.  A negative
            field would otherwise render as ``2000-01--1``, which the parser's
            lexical cast accepts as a *plausible* instant rather than rejecting,
            so the check has to happen here.

    Examples:
        >>> _iso_string_from_calendar_components(2000, 1, 1, 43200.0)
        '2000-01-01T12:00:00.000000000'
        >>> _iso_string_from_calendar_components(1989, 9, 19, 12 * 3600.0)
        '1989-09-19T12:00:00.000000000'
    """
    if month < 1.0 or day < 1.0:
        raise ValueError(
            f"Calendar date {year}-{month}-{day} is not a real date: month and day "
            "must both be at least 1."
        )

    # Nine fractional digits is the precision ``DateTime`` carries, and the
    # width the seconds field is rendered at below, so rounding here is where
    # the instant is actually decided.  Rounding *before* the split is what makes
    # it safe: a value a hair under a minute boundary rounds up to the next
    # minute on its own, whereas splitting first and then subtracting 60 from the
    # remainder leaves a negative zero that would render as ``-0.000000000``.
    total_seconds = round(seconds_of_day, 9)
    carried_days, seconds_within_day = divmod(total_seconds, SECONDS_PER_DAY)
    hours, remainder = divmod(seconds_within_day, 3600.0)
    minutes, seconds = divmod(remainder, 60.0)

    day_number = int(np.floor(day)) + int(carried_days)
    return (
        f"{int(year):04d}-{int(month):02d}-{day_number:02d}"
        f"T{int(hours):02d}:{int(minutes):02d}:{seconds:012.9f}"
    )


def _calendar_seconds_via_tudat(
    year: pd.Series,
    month: pd.Series,
    day: pd.Series,
    hour: pd.Series,
    minute: pd.Series,
    second: pd.Series,
) -> np.ndarray:
    """Convert calendar component columns to seconds since J2000 through TudatPy.

    Each record is rendered as an ISO-8601 string and read back by
    ``DateTime.from_iso_string(...).to_epoch()`` - the same path an ISO-8601
    column takes, so there is a single conversion route for every layout.
    ``DateTime`` is the reference for this conversion, the implementation the
    rest of the pipeline feeds, so agreeing with it is guaranteed rather than
    established by a comparison test.  Replacing the previous hand-rolled
    integer calendar-to-JD algorithm also removes its float noise: measured
    against ``DateTime`` it was displaced by up to 1.5e-5 s.

    A record whose year, month or day is missing has no epoch and yields NaN; a
    missing time component is zero.  Both are decided on the *raw* values before
    any defaulting, so an absent column and a blank cell are treated alike.

    Args:
        year: Calendar year column.
        month: Calendar month column.
        day: Day-of-month column, possibly fractional.
        hour: Hour column, or zeros when the source has none.
        minute: Minute column, or zeros when the source has none.
        second: Second column, or zeros when the source has none.

    Returns:
        Seconds since J2000, aligned with the input index.

    Raises:
        RuntimeError: If a record names a calendar date that does not exist.
            Propagated deliberately: an impossible date means the source is
            being misread, and a silently wrong epoch is indistinguishable from
            a wrong orbit afterwards.
    """
    index = year.index
    year_values = year.to_numpy(dtype="float64", na_value=np.nan)
    month_values = month.to_numpy(dtype="float64", na_value=np.nan)
    day_values = day.to_numpy(dtype="float64", na_value=np.nan)

    hour_values = hour.to_numpy(dtype="float64", na_value=0.0)
    minute_values = minute.to_numpy(dtype="float64", na_value=0.0)
    second_values = second.to_numpy(dtype="float64", na_value=0.0)

    # A blank time component means midnight, the start of the day, so it is
    # filled with zero above; an infinite one is not a time at all.
    seconds_of_day = (hour_values * 3600.0 + minute_values * 60.0 + second_values) + (
        day_values - np.floor(day_values)
    ) * SECONDS_PER_DAY

    # A record missing any of year/month/day has no epoch, and must not be
    # rendered as a plausible-looking date.  Checked on the raw values, before
    # defaulting, so an absent column and a blank cell behave alike.
    missing = ~(
        np.isfinite(year_values)
        & np.isfinite(month_values)
        & np.isfinite(day_values)
        & np.isfinite(seconds_of_day)
    )

    seconds = np.full(year_values.shape, np.nan, dtype="float64")
    for position in np.flatnonzero(~missing):
        text = _iso_string_from_calendar_components(
            year_values[position],
            month_values[position],
            day_values[position],
            seconds_of_day[position],
        )
        seconds[position] = DateTime.from_iso_string(text).to_epoch()

    return pd.Series(seconds, index=index)


def _iso_string_seconds_via_tudat(value: str) -> float:
    """Parse one ISO-8601 string to seconds since J2000.

    The string is handed to ``DateTime`` unchanged where it is already in the
    form that constructor accepts.  Two shapes are normalised first, because a
    source may legitimately write either and ``DateTime`` accepts neither:

    * a **date with no time**, padded with ``T00:00:00`` - NSDB writes such dates,
      and they carry time-of-day as a separate fraction of the day column;
    * a **trailing ``Z`` or numeric UTC offset**, converted to the equivalent UTC
      calendar instant.  The offset is an explicit statement about the clock, so
      applying it here loses nothing; the timescale layer still owns the scale.

    A calendar-components layout is normalised into this same form by
    :func:`_iso_string_from_calendar_components`, so every non-numeric layout
    reaches ``DateTime`` by one route.

    Args:
        value: An ISO-8601 date or date-time string.

    Returns:
        Seconds since the J2000 epoch, or ``nan`` if unparseable.  An
        unparseable value is logged and recorded as NaN rather than raising:
        the file is machine-written, so a bad value means a corrupt row, and one
        corrupt row should not abort an entire dataset.

    Examples:
        >>> _iso_string_seconds_via_tudat("2000-01-01T12:00:00")
        0.0
        >>> _iso_string_seconds_via_tudat("2000-01-01")
        -43200.0
    """
    text = str(value).strip()
    if not text or text in {"nan", "NaT", "None"}:
        return float("nan")

    if len(text) == 10:
        text = f"{text}T00:00:00"
    elif text.endswith("Z") or text.endswith("z"):
        text = f"{text[:-1]}+00:00"

    if "+" in text[10:]:
        try:
            # The offset is normalised to UTC and then dropped: the result is a
            # bare calendar instant, and the timescale layer applies the scale.
            moment = datetime.fromisoformat(text).astimezone(tz=UTC)
        except ValueError:
            logger.warning("Could not parse ISO-8601 epoch %r; recording NaN.", value)
            return float("nan")
        text = moment.replace(tzinfo=None, microsecond=0).isoformat()

    try:
        return DateTime.from_iso_string(text).to_epoch()
    except RuntimeError:
        logger.warning("Could not parse ISO-8601 epoch %r; recording NaN.", value)
        return float("nan")


def _numeric_epochs_via_tudat(values: np.ndarray, kind: str) -> np.ndarray:
    """Convert a column of Julian or Modified Julian Dates to seconds since J2000.

    ``DateTime.from_julian_day`` and ``DateTime.from_modified_julian_day`` are
    the reference conversions, and match the arithmetic they replace exactly -
    subtraction from JD 2451545.0 and scaling by 86400 - so this is a swap of
    implementation, not of convention.

    Each value is converted on its own so that one malformed row cannot abort a
    dataset: a non-finite value, or one outside the range ``DateTime`` can
    represent, is recorded as NaN with a warning.  The old arithmetic had the
    same effective behaviour, producing NaN from non-finite input rather than
    raising.

    Args:
        values: The numeric epoch column, as float64.
        kind: ``"jd"`` or ``"mjd"``, selecting the constructor.

    Returns:
        Seconds since J2000, same shape as ``values``.

    Raises:
        ValueError: If ``kind`` is neither ``"jd"`` nor ``"mjd"``.
    """
    if kind == "jd":
        constructor = DateTime.from_julian_day
    elif kind == "mjd":
        constructor = DateTime.from_modified_julian_day
    else:  # pragma: no cover - the caller only passes the two numeric kinds
        raise ValueError(f"Unsupported numeric epoch kind {kind!r}")

    seconds = np.full(values.shape, np.nan, dtype="float64")
    for position, value in enumerate(values):
        if not np.isfinite(value):
            continue
        try:
            seconds[position] = constructor(float(value)).to_epoch()
        except RuntimeError as error:
            logger.warning(
                "Could not interpret %s %s as a %s; recording NaN. (%s)",
                kind.upper(),
                value,
                "Julian Date",
                error,
            )
    return seconds


def to_seconds_since_j2000(
    dataframe: pd.DataFrame,
    layout: EpochLayout | None = None,
    *,
    in_place: bool = False,
    column: str = EPOCH_COLUMN,
) -> pd.DataFrame:
    """Add a seconds-since-J2000 column for a frame's observation epoch.

    The result is on **the timescale the source recorded**, which this function
    neither knows nor changes; hand it to
    :func:`orbitdet.preprocessing.time_scales.to_tdb_seconds_since_j2000` to
    reach TDB.  Keeping the scale out of this step is what lets a ``JD UT1``
    epoch and an ``ISO UTC`` epoch remain distinguishable.

    Every kind is converted by TudatPy - ``DateTime.from_iso_string``,
    ``DateTime.from_julian_day`` and ``DateTime.from_modified_julian_day`` - so
    the pipeline's notion of an instant *is* Tudat's, rather than a
    reimplementation that has to be kept in agreement with it.

    Args:
        dataframe: The frame holding the epoch.
        layout: How the epoch is encoded.  Detected automatically when omitted.
        in_place: If ``True``, add the column to ``dataframe`` and return it.
            If ``False`` (default), return a copy.
        column: Name of the column to write.

    Returns:
        The frame with the seconds column added.

    Raises:
        ValueError: If the layout cannot be determined.

    Examples:
        >>> import pandas as pd
        >>> frame = pd.DataFrame({"JD": [2451545.0]})
        >>> float(to_seconds_since_j2000(frame)[EPOCH_COLUMN].iloc[0])
        0.0
        >>> frame = pd.DataFrame({"MJD of observation": [58000.0]})
        >>> float(to_seconds_since_j2000(frame)[EPOCH_COLUMN].iloc[0])
        557755200.0
        >>> frame = pd.DataFrame({"iso_time": ["2000-01-01"]})
        >>> float(to_seconds_since_j2000(frame)[EPOCH_COLUMN].iloc[0])
        -43200.0
    """
    if layout is None:
        layout = detect_epoch_layout(dataframe)

    result = dataframe if in_place else dataframe.copy()

    if layout.kind == "jd":
        result[column] = _numeric_epochs_via_tudat(
            numeric_series(result, layout.columns["jd"]).to_numpy("float64") + layout.offset_days,
            "jd",
        )

    elif layout.kind == "mjd":
        result[column] = _numeric_epochs_via_tudat(
            numeric_series(result, layout.columns["mjd"]).to_numpy("float64") + layout.offset_days,
            "mjd",
        )

    elif layout.kind == "iso":
        values = result[layout.columns["iso"]].astype("string")
        present = values.notna().to_numpy()
        seconds = np.full(len(values), np.nan, dtype="float64")
        if present.any():
            seconds[present] = [
                _iso_string_seconds_via_tudat(value) for value in values[present].to_numpy()
            ]
        result[column] = seconds

    elif layout.kind == "calendar":
        # numeric_series supplies zeros for the hour/minute/second columns a
        # source does not have, and NaN for the ones it must have.
        result[column] = _calendar_seconds_via_tudat(
            numeric_series(result, layout.columns["year"]),
            numeric_series(result, layout.columns["month"]),
            numeric_series(result, layout.columns["day"]),
            numeric_series(result, layout.columns["hour"]),
            numeric_series(result, layout.columns["minute"]),
            numeric_series(result, layout.columns["second"]),
        )

    else:  # pragma: no cover - detect_epoch_layout only returns the four kinds
        raise ValueError(f"Unsupported epoch layout kind {layout.kind!r}")

    if layout.offset_days:
        logger.debug(
            "Epoch column declared a reference of JD %s; added %s days",
            layout.offset_days,
            layout.offset_days,
        )
    logger.debug("Resolved epoch layout %r from columns %r", layout.kind, layout.columns)
    return result


def seconds_since_j2000_to_julian_date(seconds: np.ndarray | pd.Series) -> np.ndarray:
    """Convert seconds since J2000 to a Julian Date.

    The inverse of the conversions above, for callers that need to report or
    compare against tabulated Julian Dates.  Derived by inverting Tudat's own
    Julian-Date conversion, not by a tabulated epoch constant, so that
    ``DateTime.from_julian_day`` after this is the identity.

    Args:
        seconds: Seconds since the J2000 epoch.

    Returns:
        The corresponding Julian Date.

    Examples:
        >>> import numpy as np
        >>> float(seconds_since_j2000_to_julian_date(np.array([0.0]))[0])
        2451545.0
    """
    values = np.asarray(seconds, dtype="float64")
    return values / SECONDS_PER_DAY + DateTime.from_epoch(0.0).to_julian_day()


def seconds_since_j2000_to_datetime(seconds: float) -> datetime:
    """Render seconds since J2000 as a naive UTC :class:`datetime`.

    Convenience for logging and diagnostics.  It converts *the count* back to
    calendar form; it does not apply or undo a timescale.  The calendar form
    comes from ``DateTime``, so it agrees with how the pipeline reads epochs,
    which ``datetime``'s own fixed-day arithmetic (correct for every date since
    1582) would only happen to match.

    Args:
        seconds: Seconds since the J2000 epoch.

    Returns:
        The corresponding calendar instant.

    Examples:
        >>> seconds_since_j2000_to_datetime(0.0).isoformat()
        '2000-01-01T12:00:00'
    """
    return DateTime.from_epoch(float(seconds)).to_python_datetime()

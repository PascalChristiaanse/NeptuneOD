"""Data preprocessing: standardise heterogeneous observation sources into one dataframe.

This package is responsible for turning raw observation data (NSDB text files,
Voyager tables, Gaia archive pulls, ...) into a single, well-defined
:class:`pandas.DataFrame` that the :mod:`orbitdet.observations` factories can
consume without knowing anything about the original source format.

Design
------
The package is split into two layers:

**Layer 1 - pure functions.**  No Tudat objects, no ``SystemOfBodies``, no I/O.
These are trivially unit-testable and cover the bulk of the work:

* :mod:`~orbitdet.preprocessing.columns` - fuzzy column discovery and the
  NSDB ``format_columns`` overlay.
* :mod:`~orbitdet.preprocessing.time_formatting` - multicolumn / ISO / JD / MJD
  to a common epoch representation, read by TudatPy's ``DateTime``.
  Does **not** apply a timescale.
* :mod:`~orbitdet.preprocessing.time_scales` - Washington mean time, GMT, UTC,
  UT1, TCB, ET, ... to TDB seconds since J2000.
* :mod:`~orbitdet.preprocessing.precession` - mean-of-date, true-of-date and
  Besselian/Julian equinoxes to J2000.
* :mod:`~orbitdet.preprocessing.catalogues` - reduction-catalogue tie to
  Gaia-CRF3 / ICRF, including stellar proper motion.
* :mod:`~orbitdet.preprocessing.aberration` - apparent to astrometric
  (stellar aberration).
* :mod:`~orbitdet.preprocessing.frames` - frame-name normalisation and the
  dispatch between the above.

**Layer 2 - context-aware orchestration.**  A thin layer that accepts a
``SystemOfBodies`` (or the pieces of it that are needed) and produces the
standard dataframe.  This exists because two steps genuinely require
environment context:

* timescale conversion needs the observer's Earth-fixed position, and
* the relative X/Y gnomonic inversion needs the target's topocentric direction.

Units
-----
**All angular quantities in the standard dataframe are radians.**  This is the
internal unit of TudatPy and the pipeline does not deviate from it.  Source
units (degrees, hours, arcseconds, seconds of time) are converted on ingest and
the original unit is recorded in the ``*_unit`` provenance columns.

See Also
--------
orbitdet.preprocessing.schema : the dataframe contract itself.
"""

from .catalogues import (
    CATALOGUE_TIES,
    CatalogueTie,
    TieReport,
    apply_proper_motion,
    describe_tie,
    normalize_catalogue,
    normalize_catalogue_series,
    tie_to_icrf,
)
from .columns import (
    ARCSECOND_TO_RADIANS,
    SECONDS_OF_TIME_TO_RADIANS,
    apply_format_columns,
    find_column,
    find_column_substring,
    normalize_column_name,
    numeric_series,
    relative_position_scale,
)
from .observatories import (
    ObservatoryInfo,
    configure_from_config,
    get_observatory_info,
    get_observatory_location,
    normalize_observatory_code,
    set_observatories_path,
)
from .schema import (
    ANGLE_COLUMNS,
    COLUMN_SPECS,
    NOT_GIVEN,
    NOT_GIVEN_COLUMNS,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    STANDARD_DATAFRAME_COLUMNS,
    ObservationFrameError,
    ReductionCatalogue,
    StandardObservationFrame,
    apply_not_given,
    build_frame,
    empty_frame,
    is_not_given,
    verify,
)
from .time_formatting import (
    EPOCH_COLUMN,
    EpochLayout,
    detect_epoch_layout,
    seconds_since_j2000_to_datetime,
    seconds_since_j2000_to_julian_date,
    to_seconds_since_j2000,
)
from .time_scales import (
    ASTRONOMICAL_DAY_OFFSET,
    ASTRONOMICAL_DAY_SWITCH_EPOCH,
    InputTimeScale,
    MeridianTime,
    TimescaleResolution,
    delta_t_seconds,
    resolve_timescale,
    to_tdb_seconds_since_j2000,
)

__all__ = [
    # Columns
    "normalize_column_name",
    "find_column",
    "find_column_substring",
    "numeric_series",
    "apply_format_columns",
    "relative_position_scale",
    "ARCSECOND_TO_RADIANS",
    "SECONDS_OF_TIME_TO_RADIANS",
    # Epoch formatting
    "EPOCH_COLUMN",
    "EpochLayout",
    "detect_epoch_layout",
    "to_seconds_since_j2000",
    "seconds_since_j2000_to_julian_date",
    "seconds_since_j2000_to_datetime",
    # Reduction catalogues
    "ReductionCatalogue",
    "CatalogueTie",
    "TieReport",
    "CATALOGUE_TIES",
    "normalize_catalogue",
    "normalize_catalogue_series",
    "catalogue_epoch_seconds_since_j2000",
    "apply_proper_motion",
    "tie_to_icrf",
    "describe_tie",
    # Schema
    "StandardObservationFrame",
    "ObservationFrameError",
    "STANDARD_DATAFRAME_COLUMNS",
    "COLUMN_SPECS",
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "ANGLE_COLUMNS",
    "NOT_GIVEN",
    "NOT_GIVEN_COLUMNS",
    "is_not_given",
    "apply_not_given",
    "verify",
    "empty_frame",
    "build_frame",
    # Observatory reference data
    "ObservatoryInfo",
    "configure_from_config",
    "get_observatory_info",
    "get_observatory_location",
    "normalize_observatory_code",
    "set_observatories_path",
    # Timescales
    "InputTimeScale",
    "MeridianTime",
    "TimescaleResolution",
    "ASTRONOMICAL_DAY_OFFSET",
    "ASTRONOMICAL_DAY_SWITCH_EPOCH",
    "resolve_timescale",
    "to_tdb_seconds_since_j2000",
    "delta_t_seconds",
]

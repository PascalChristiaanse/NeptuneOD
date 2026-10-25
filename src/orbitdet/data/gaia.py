"""
Adapter around :class:`tudatpy.data.gaia.gaia.GaiaAstrometry` for the NeptuneOD pipeline.

Historically this module contained a home-grown ``GaiaQuery`` class that retrieved
Gaia FPR astrometry from the archive and turned it into a Tudat
``ObservationCollection``.  That functionality now lives in the official tudatpy
``GaiaAstrometry`` interface (which supports the full transit covariance weight
matrix, asteroid *and* natural-satellite astrometry, and local archive loading).

The :class:`GaiaQuery` class below is a thin adapter that keeps the NeptuneOD
call-sites stable while delegating all retrieval, filtering and weighting to
``GaiaAstrometry``.  Archive (and local-archive) reads are cached on disk for
one week so that repeated queries with identical parameters skip the network.

Triton is a natural planetary satellite, and Gaia assigns MPC number 0 to every
natural satellite.  Such objects are therefore keyed and queried by their Gaia
``source_id`` (Triton: ``-4293959295``) instead of an MPC number.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Memory
from joblib.memory import expires_after
from platformdirs import user_cache_dir
from tudatpy.astro.time_representation import DateTime
from tudatpy.data.gaia import GaiaAstrometry
from tudatpy.data.gaia.gaia import _ASTROMETRY_CATALOG
from tudatpy.dynamics.environment import SystemOfBodies
from tudatpy.estimation import observations

from orbitdet.data.light_deflection import relativistic_light_deflection
from orbitdet.data.photocenter import photocenter_offset_spherical
from orbitdet.transformations import convert_radec_frame

logger = logging.getLogger(__name__)

# On-disk cache for Gaia archive reads.  The Gaia FPR release changes on a roughly
# yearly cadence and archive outages/retries are common, so caching for a week
# strikes a balance between avoiding repeated round-trips within a campaign and
# not serving stale data.  Stored under the user cache directory (e.g.
# ``~/.cache/orbitdet`` on Linux, ``~/Library/Caches/orbitdet`` on macOS).
#
# Entries are keyed on the full set of query arguments, so distinct queries
# coexist rather than evicting one another, and the number of entries is
# unlimited (joblib only enforces a cap via ``Memory.reduce_size``, which is not
# used here).  joblib also hashes the function source, so editing the fetch logic
# invalidates all cached entries automatically.
_GAIA_QUERY_CACHE = Memory(
    location=user_cache_dir("orbitdet") + "/gaia_query_cache",
    verbose=0,
)
_GAIA_QUERY_CACHE_TTL = expires_after(weeks=1)

# Registry of position angle of scan (radians) keyed by the observation epoch
# (seconds since J2000 TDB).  The Tudat ``ObservationCollection`` /
# ``SingleObservationSet`` pybind classes do not allow attaching arbitrary
# attributes, so we carry the scan angles alongside via this module-level map.
# Epochs are unique per observation, so they form a stable key that survives
# merging of observation collections.
_SCAN_ANGLE_REGISTRY: dict[tuple[float, ...], np.ndarray] = {}


def _register_scan_angles(epochs: np.ndarray | list[float], scan_angles: np.ndarray) -> None:
    """Store the per-observation scan angles keyed by their epochs.

    Args:
        epochs: Observation epochs in seconds since J2000 TDB.
        scan_angles: Position angle of scan (radians) for each observation.
    """
    _SCAN_ANGLE_REGISTRY[tuple(float(e) for e in epochs)] = np.asarray(scan_angles, dtype=float)


def get_scan_angles_for_epochs(epochs: np.ndarray | list[float]) -> np.ndarray | None:
    """Return the scan angles (radians) matching the given epochs, if registered.

    Performs an exact match first (fast path for a collection built from a
    single :class:`GaiaQuery`).  If no exact match is found, falls back to a
    per-epoch lookup across all registered epoch sets, which is robust to
    observation collections formed by merging several Gaia queries.

    Args:
        epochs: Observation epochs in seconds since J2000 TDB.

    Returns:
        Array of scan angles matching ``epochs`` (same length), or ``None`` if
        no matching scan angles could be found.
    """
    key = tuple(float(e) for e in epochs)
    direct = _SCAN_ANGLE_REGISTRY.get(key)
    if direct is not None:
        return direct

    # Fallback: build an epoch->angle map across all registered sets and look up
    # each requested epoch individually.
    epoch_map: dict[float, float] = {}
    for registered_epochs, angles in _SCAN_ANGLE_REGISTRY.items():
        epoch_map.update(dict(zip(registered_epochs, angles.tolist())))

    try:
        return np.asarray([epoch_map[float(e)] for e in epochs], dtype=float)
    except KeyError:
        return None


# Registry of archived Gaia random/systematic errors keyed by the observation
# epoch (seconds since J2000 TDB).  Mirrors ``_SCAN_ANGLE_REGISTRY``: the Tudat
# ``SingleObservationSet`` pybind classes cannot carry arbitrary attributes, so
# the archived random/systematic errors are carried alongside via this map.
# Each value is an ``(n, 6)`` array with columns
# ``[ra_err_random, dec_err_random, corr_random,
#    ra_err_systematic, dec_err_systematic, corr_systematic]``
# in radians (RA errors already de-projected by ``cos(dec)``).
_GAIA_ERROR_REGISTRY: dict[tuple[float, ...], np.ndarray] = {}

#: Column order of the arrays stored in :data:`_GAIA_ERROR_REGISTRY`.
GAIA_ERROR_COLUMNS = (
    "ra_error_random",
    "dec_error_random",
    "ra_dec_correlation_random",
    "ra_error_systematic",
    "dec_error_systematic",
    "ra_dec_correlation_systematic",
)


def _register_gaia_errors(epochs: np.ndarray | list[float], errors: np.ndarray) -> None:
    """Store the per-observation Gaia errors keyed by their epochs.

    Args:
        epochs: Observation epochs in seconds since J2000 TDB.
        errors: ``(n, 6)`` array in :data:`GAIA_ERROR_COLUMNS` order.
    """
    _GAIA_ERROR_REGISTRY[tuple(float(e) for e in epochs)] = np.asarray(errors, dtype=float)


def get_gaia_errors_for_epochs(epochs: np.ndarray | list[float]) -> np.ndarray | None:
    """Return the Gaia errors matching the given epochs, if registered.

    Mirrors :func:`get_scan_angles_for_epochs`: exact match first, then a
    per-epoch fallback across all registered sets.

    Args:
        epochs: Observation epochs in seconds since J2000 TDB.

    Returns:
        ``(n, 6)`` array in :data:`GAIA_ERROR_COLUMNS` order, or ``None`` if no
        matching errors could be found.
    """
    key = tuple(float(e) for e in epochs)
    direct = _GAIA_ERROR_REGISTRY.get(key)
    if direct is not None:
        return direct

    # Fallback: build an epoch->row map across all registered sets.
    epoch_map: dict[float, np.ndarray] = {}
    for registered_epochs, errors in _GAIA_ERROR_REGISTRY.items():
        epoch_map.update(dict(zip(registered_epochs, errors)))

    try:
        return np.asarray([epoch_map[float(e)] for e in epochs], dtype=float)
    except KeyError:
        return None


def get_gaia_ephemeris(
    source_ids: list[int],
    geocentric: bool = True,
    filter_outcomes: bool = True,
):
    """Build a Gaia tabulated ephemeris state history from the archive.

    Retrieves the Gaia state vectors archived alongside the observations and
    packs them into a Tudat-compatible state history (keys = seconds since
    J2000 TDB, values = 6x1 position/velocity in SI units).

    Args:
        source_ids: Gaia ``source_id`` values used to select the observations.
        geocentric: If True, use the geocentric Gaia states (frame origin
            ``Earth``).  If False, use the barycentric states (frame origin
            ``SSB``).
        filter_outcomes: If True, keep only rows with
            ``astrometric_outcome_ccd == 1`` and ``astrometric_outcome_transit
            == 1``.  If False, use all rows returned by the archive (no
            filtering).

    Returns:
        EphemerisSettings: Tabulated ephemeris settings.
    """
    query = GaiaQuery()
    query.retrieve_data(
        source_ids=source_ids,
        filter_outcomes=filter_outcomes,
    )
    return query.get_gaia_ephemeris(geocentric=geocentric)


class GaiaQuery:
    """Adapter around :class:`tudatpy.data.gaia.gaia.GaiaAstrometry`.

    Supports both asteroids (queried by MPC number) and natural planetary
    satellites such as Triton (queried by Gaia ``source_id``).  All retrieval,
    unit conversion, filtering and weighting is delegated to
    ``GaiaAstrometry``; this class only adds the NeptuneOD-specific behaviour:

    * frame conversion of the archived RA/DEC into the experiment frame;
    * per-object astrometric corrections (photocenter and light deflection);
    * registration of the archived scan angles for AL/AC residual plots.

    Attributes:
        _astrometry (GaiaAstrometry | None): Wrapped astrometry after calling
            :meth:`retrieve_data` or :meth:`retrieve_data_locally`.
    """

    def __init__(self):
        """Create a Gaia query object."""
        self._astrometry: GaiaAstrometry | None = None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def astrometry(self) -> GaiaAstrometry:
        """The wrapped :class:`GaiaAstrometry` object (raises if not loaded)."""
        if self._astrometry is None:
            raise RuntimeError("No observations loaded")
        return self._astrometry

    @property
    def observation_table(self) -> pd.DataFrame:
        """Return a copy of the observation table."""
        if self._astrometry is None:
            return pd.DataFrame()
        return self._astrometry.table

    @property
    def epoch_start(self) -> float:
        """Return first epoch in the observation table (any object)."""
        return self.observation_table["epoch"].iloc[0]  # Table is ordered by epoch

    @property
    def epoch_end(self) -> float:
        """Return last epoch in the observation table (any object)."""
        return self.observation_table["epoch"].iloc[-1]  # Table is ordered by epoch

    @property
    def mpc_numbers(self) -> np.ndarray:
        """Which asteroid MPC numbers appear in the observations table."""
        return self.astrometry.mpc_numbers_in_table

    def copy(self) -> GaiaQuery:
        """Return a deep copy of the query object."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @staticmethod
    @_GAIA_QUERY_CACHE.cache(cache_validation_callback=_GAIA_QUERY_CACHE_TTL)
    def _cached_archive_fetch(
        mpc_numbers: tuple[int, ...] | None,
        source_ids: tuple[int, ...] | None,
    ) -> pd.DataFrame:
        """Fetch Gaia observations from the archive, with a one-week disk cache.

        Results are cached on disk (via ``joblib``) for one week, keyed on the
        query keys, so repeated runs within a campaign skip the archive.
        Distinct queries are cached independently and do not evict one another;
        the cache has no entry limit.  Entries older than one week are
        transparently cleared and recomputed on next access.

        Args:
            mpc_numbers: MPC numbers as a hashable tuple, or None.
            source_ids: Gaia source IDs as a hashable tuple, or None.

        Returns:
            Unit-converted observation table (identical to what
            ``GaiaAstrometry.load_from_astroquery`` / ``..._for_source_ids``
            would produce).
        """
        from astroquery.gaia import Gaia  # late import because it tends to be slow

        if mpc_numbers is not None:
            where_clause = f"number_mp IN ({', '.join(str(m) for m in mpc_numbers)})"
        else:
            where_clause = f"source_id IN ({', '.join(str(s) for s in source_ids)})"

        query = f"""
        SELECT *
        FROM {_ASTROMETRY_CATALOG}
        WHERE {where_clause}
        """
        logger.info("Fetching Gaia observations from archive: %s", " ".join(query.split()))

        try:
            job = Gaia.launch_job_async(query)
            table = job.get_results()
        except Exception as err:
            logger.error("Gaia archive query failed: %s", err)
            raise RuntimeError(f"Error while retrieving astrometric observations: \n{err}") from err

        logger.info("Gaia archive query succeeded: retrieved %d row(s).", len(table))
        table = table.to_pandas()

        if table.empty:
            raise LookupError(f"No observations found for query {where_clause}")

        # Convert units and sort into the tudat-compatible format.
        return GaiaAstrometry._prepare_table(table)

    def _load_astrometry(
        self,
        mpc_numbers: tuple[int, ...] | None,
        source_ids: tuple[int, ...] | None,
        cache: bool,
    ) -> GaiaAstrometry:
        """Load a :class:`GaiaAstrometry` object, optionally via the disk cache."""
        if cache:
            table = self._cached_archive_fetch(mpc_numbers, source_ids)
        else:
            from astroquery.gaia import Gaia  # late import because it tends to be slow

            mpc_str = ", ".join(str(m) for m in mpc_numbers) if mpc_numbers is not None else None
            src_str = ", ".join(str(s) for s in source_ids) if source_ids is not None else None
            where_clause = (
                f"number_mp IN ({mpc_str})"
                if mpc_numbers is not None
                else f"source_id IN ({src_str})"
            )
            query = f"SELECT * FROM {_ASTROMETRY_CATALOG} WHERE {where_clause}"
            try:
                table = Gaia.launch_job_async(query).get_results().to_pandas()
            except Exception as err:
                raise RuntimeError(
                    f"Error while retrieving astrometric observations: \n{err}"
                ) from err
            if table.empty:
                raise LookupError(f"No observations found for query {where_clause}")
            table = GaiaAstrometry._prepare_table(table)

        return GaiaAstrometry(table)

    def retrieve_data(
        self,
        mpc_numbers: tuple[int] | list[int] | None = None,
        source_ids: list[int] | None = None,
        catalog: str = "FPR",
        filter_outcomes: bool = True,
        cache: bool = True,
    ) -> None:
        """Retrieve astrometric observations through astroquery.

        Observations are stored in the observation table attribute.  The
        archive fetch is cached on disk for one week so repeated queries with
        identical parameters skip the archive round-trip.

        Args:
            mpc_numbers: List of asteroid MPC numbers to retrieve.
            source_ids: List of Gaia ``source_id`` values to retrieve (e.g. for
                Triton).  Mutually exclusive with ``mpc_numbers``.
            catalog: Which catalog to use.  Only the Gaia FPR catalog is
                supported through the ``GaiaAstrometry`` interface.
            filter_outcomes: If True, keep only rows with
                ``astrometric_outcome_ccd == 1`` and
                ``astrometric_outcome_transit == 1``.  If False, use all rows
                returned by the archive (no filtering).
            cache: If True (default), use the one-week on-disk cache for the
                archive fetch.  If False, always query the archive.
        """
        if (mpc_numbers is None) == (source_ids is None):
            raise ValueError("Provide exactly one of mpc_numbers or source_ids")

        if catalog != "FPR":
            raise ValueError(
                "Only the Gaia FPR catalog is supported by the GaiaAstrometry interface "
                f"(got '{catalog}')."
            )

        self._astrometry = self._load_astrometry(
            mpc_numbers=tuple(mpc_numbers) if mpc_numbers is not None else None,
            source_ids=tuple(source_ids) if source_ids is not None else None,
            cache=cache,
        )

        if filter_outcomes:
            self._astrometry.apply_filters(exclude_poor_observations=True)

    def retrieve_data_locally(
        self,
        mpc_numbers: list[int] | None = None,
        source_ids: list[int] | None = None,
        file_path: str = "",
        filter_outcomes: bool = True,
        cache: bool = True,
    ) -> None:
        """Retrieve astrometric observations from a local archive.

        Args:
            mpc_numbers: List of asteroid MPC numbers to retrieve.
            source_ids: List of Gaia ``source_id`` values to retrieve.
            file_path: Path to a .parquet archive file (generated via
                :func:`tudatpy.data.gaia.generate_astrometry_parquet`) or to a
                directory containing the ``SsoObservation_*.csv`` chunks.
            filter_outcomes: If True, keep only rows with
                ``astrometric_outcome_ccd = 1`` and
                ``astrometric_outcome_transit = 1``.  If False, use all rows
                (no filtering).
            cache: If True (default), cache the concatenated CSV archive read on
                disk for one week.  Applies to the CSV-directory path only.
        """
        if (mpc_numbers is None) == (source_ids is None):
            raise ValueError("Provide exactly one of mpc_numbers or source_ids")

        path = Path(file_path)

        if path.suffix == ".parquet":
            if source_ids is not None:
                self._astrometry = GaiaAstrometry.load_from_local_archive_for_source_ids(
                    path, source_ids
                )
            else:
                self._astrometry = GaiaAstrometry.load_from_local_archive(path, mpc_numbers)
        else:
            table = self._read_csv_archive_directory(path, cache=cache)
            if source_ids is not None:
                table = table[table["source_id"].isin(source_ids)]
            else:
                table = table[table["number_mp"].isin(mpc_numbers)]
            if table.empty:
                raise LookupError("No observations found for query")
            self._astrometry = GaiaAstrometry(GaiaAstrometry._prepare_table(table))

        if filter_outcomes:
            self._astrometry.apply_filters(exclude_poor_observations=True)

    @staticmethod
    def _read_csv_archive_directory(directory: Path, cache: bool = True) -> pd.DataFrame:
        """Concatenate the 20 Gaia ``SsoObservation`` CSV chunks in a directory.

        The concatenated table is cached on disk for one week (when ``cache`` is
        True) so repeated runs skip the (slow) CSV parsing of the full archive.
        """
        if cache:
            return GaiaQuery._cached_read_csv_archive(str(Path(directory).resolve()))
        return GaiaQuery._read_csv_archive_directory_uncached(str(Path(directory).resolve()))

    @staticmethod
    @_GAIA_QUERY_CACHE.cache(cache_validation_callback=_GAIA_QUERY_CACHE_TTL)
    def _cached_read_csv_archive(directory: str) -> pd.DataFrame:
        """One-week-cached read of the CSV archive directory."""
        return GaiaQuery._read_csv_archive_directory_uncached(directory)

    @staticmethod
    def _read_csv_archive_directory_uncached(directory: str) -> pd.DataFrame:
        """Concatenate the 20 Gaia ``SsoObservation`` CSV chunks in a directory."""
        directory = Path(directory)
        chunks = []
        for i in range(20):
            file_number = f"{i:02d}"
            compressed = directory / f"SsoObservation_{file_number}.csv.gz"
            plain = directory / f"SsoObservation_{file_number}.csv"
            csv_file = compressed if compressed.exists() else plain
            logger.info("Loading file %s...", csv_file.name)
            chunks.append(pd.read_csv(csv_file, comment="#"))
        return pd.concat(chunks, ignore_index=True)

    def filter(
        self,
        epoch_start: float | datetime | DateTime,
        epoch_end: float | datetime | DateTime,
    ) -> None:
        """Filter the observations after they have been loaded (in-place)."""
        if self._astrometry is None:
            raise Exception("No observations loaded")

        if isinstance(epoch_start, datetime) and isinstance(epoch_end, datetime):
            epoch_start = DateTime.from_python_datetime(epoch_start)
            epoch_end = DateTime.from_python_datetime(epoch_end)

        if isinstance(epoch_start, DateTime) and isinstance(epoch_end, DateTime):
            epoch_start = epoch_start.to_epoch()
            epoch_end = epoch_end.to_epoch()

        self._astrometry.apply_filters(
            exclude_poor_observations=False,
            epoch_start=epoch_start,
            epoch_end=epoch_end,
        )

    def summary(self) -> None:
        """Log a convenient summary of the astrometric observations."""
        if self._astrometry is None:
            logger.info("Observations not loaded")
            return
        self._astrometry.print_summary()

    # ------------------------------------------------------------------
    # Astrometric corrections
    # ------------------------------------------------------------------

    def correct_observations(
        self,
        target_name: str,
        bodies: SystemOfBodies,
        light_deflection: tuple | list = ("Sun",),
        correct_photocenter: bool = True,
        diameter: float = 0.0,
    ) -> None:
        """Apply astrometric corrections to the observations (in-place).

        The observed RA/DEC stored in the table are shifted by relativistic
        light bending (Sun and optionally other bodies) and/or the photocenter
        offset, so that they refer to the centre of mass of the target.  These
        corrections should be ADDED to the archived astrometric coordinates
        before fitting an orbit.

        Args:
            target_name: Name of the observed body (e.g. ``"Triton"``).
            bodies: SystemOfBodies object which must have appropriate
                ephemerides loaded (target, Sun, and any light-deflecting body).
            light_deflection: Body objects which exert relativistic light
                bending.
            correct_photocenter: Apply a photocenter correction.
            diameter: Body diameter in meters, used for the photocenter offset.
                For asteroids this may be omitted (queried from SBDB), but for
                other bodies it should be given explicitly (e.g. Triton ~
                2.7068e6 m).

        Returns:
            None
        """
        if self._astrometry is None:
            raise Exception("No observations loaded")

        if correct_photocenter and diameter <= 0.0:
            raise ValueError(
                f"A positive 'diameter' (m) is required for the photocenter "
                f"correction of {target_name}."
            )

        if light_deflection:
            light_deflection = list(light_deflection)
            if not all(bodies.does_body_exist(body) for body in light_deflection):
                raise ValueError("Light deflection bodies missing from bodies object")

        local_table = self._astrometry._table
        target_names = self._target_names_for_bodies(target_name)
        for object_key in self._astrometry.object_keys_in_table:
            mask = local_table["_object_key"] == object_key
            object_table = local_table.loc[mask]
            body_name = self._astrometry._body_name_for_object(object_key, target_names)

            if not bodies.does_body_exist(body_name) or bodies.get(body_name).ephemeris is None:
                raise ValueError(
                    f"Correction not possible for {body_name}. Body needs to be in "
                    "SystemOfBodies with loaded ephemeris"
                )

            corrections = np.zeros((len(object_table), 2))

            if correct_photocenter:
                logger.info("Applying photocenter correction to %s observations.", body_name)
                corrections += photocenter_offset_spherical(
                    body_name, object_table, bodies, diameter
                )

            if light_deflection:
                logger.info(
                    "Applying relativistic light deflection to %s observations (bodies: %s).",
                    body_name,
                    light_deflection,
                )
                corrections += relativistic_light_deflection(
                    body_name, object_table, bodies, light_deflection
                )

            local_table.loc[mask, ["ra", "dec"]] += corrections

        # Wrap RA back into the principal interval after applying corrections.
        local_table["ra"] = (local_table["ra"] + np.pi) % (2 * np.pi) - np.pi

    # ------------------------------------------------------------------
    # Conversion to Tudat
    # ------------------------------------------------------------------

    def _target_names_for_bodies(self, target_name: str) -> dict[int, str]:
        """Map satellite object keys (MPC number 0) to the requested body name."""
        target_names: dict[int, str] = {}
        for object_key in self._astrometry.object_keys_in_table:
            rows = self._astrometry._table_for_single_object(object_key)
            if int(rows["number_mp"].iloc[0]) == 0:
                target_names[int(object_key)] = target_name
        return target_names

    def _cache_archive_weights(
        self, collection: observations.ObservationCollection, set_id: str = "Gaia"
    ) -> None:
        """Cache the diagonal of the full Gaia weight matrix for the ``gaia_archive`` strategy.

        The full transit covariance weight matrix is carried by the observation
        sets themselves; the diagonal is cached so the ``gaia_archive`` weighting
        strategy (which historically consumed a per-observation weight vector)
        can still retrieve it.  Note that the strategy no longer *applies* this
        diagonal over the full matrix — it leaves the full matrix untouched.
        """
        from orbitdet.observations.weighting.gaia_cache import store_gaia_weights

        diagonals = [
            np.asarray(observation_set.weights_diagonal_vector, dtype=float)
            for observation_set in collection.get_single_observation_sets()
        ]
        if diagonals:
            store_gaia_weights(set_id, np.concatenate(diagonals))

    def to_tudat(
        self,
        bodies: SystemOfBodies,
        target_name: str = "Triton",
        input_frame: str = "ICRS",
        output_frame: str | None = None,
    ) -> observations.ObservationCollection:
        """Convert observations into a Tudat ObservationCollection.

        Args:
            bodies: SystemOfBodies object.  Must contain the target body (with
                an ephemeris) and the Gaia body (with an ephemeris).
            target_name: Name of the observed body in ``bodies``.  Defaults to
                ``"Triton"``.
            input_frame: Frame of the RA/DEC columns as retrieved from the
                archive (default "ICRS").
            output_frame: Frame in which the observation angles should be
                expressed.  If None, a rotation from *input_frame* to
                *input_frame* is a no-op (defaults to *input_frame*).

        Returns:
            ObservationCollection: Tudat ObservationCollection containing the
            observations of the target body.
        """
        if self._astrometry is None:
            raise RuntimeError("No observations loaded")

        if output_frame is None or output_frame == input_frame:
            collection = self._astrometry.to_observation_collection(
                bodies, target_names=self._target_names_for_bodies(target_name)
            )
            table = self._astrometry.table
        else:
            # Rotate the archived RA/DEC into the experiment frame before
            # building the collection.  The Gaia table stores angles in radians
            # and epochs in seconds since J2000 TDB, but the ICRS->J2000
            # frame-bias and any subsequent SPICE conversion need time as
            # seconds since J2000 (Astropy FrameBias uses TDB epoch).
            converted = convert_radec_frame(
                self._astrometry.table,
                "ra",
                "dec",
                input_frame=input_frame,
                output_frame=output_frame,
                time_column="epoch",
                ra_wrap=False,
                angle_unit="rad",
            )
            rotated = GaiaAstrometry(converted)
            collection = rotated.to_observation_collection(
                bodies, target_names=self._target_names_for_bodies(target_name)
            )
            table = converted

        # Register the per-observation scan angles (radians) keyed by epoch so the
        # residual visualisations can rotate RA/Dec residuals into the Gaia
        # along-scan / across-scan frame without re-querying the archive.
        set_epochs = [
            np.array([epoch.to_float() for epoch in observation_set.observation_times])
            for observation_set in collection.get_single_observation_sets()
        ]
        epochs = np.concatenate(set_epochs) if set_epochs else np.array([])
        if len(epochs) > 0 and "position_angle_scan" in table.columns:
            epoch_to_angle = dict(zip(table["epoch"], table["position_angle_scan"]))
            scan_angles = np.array([epoch_to_angle[float(epoch)] for epoch in epochs], dtype=float)
            _register_scan_angles(epochs, scan_angles)

        # Register the per-observation random/systematic errors (radians) keyed by
        # epoch so the scan-residual visualisations can draw archive error bars.
        if len(epochs) > 0 and all(col in table.columns for col in GAIA_ERROR_COLUMNS):
            epoch_to_errors = {
                float(epoch): row
                for epoch, row in zip(table["epoch"], table[list(GAIA_ERROR_COLUMNS)].to_numpy())
            }
            try:
                errors = np.array([epoch_to_errors[float(epoch)] for epoch in epochs], dtype=float)
            except KeyError:
                errors = None
            if errors is not None:
                _register_gaia_errors(epochs, errors)

        # Cache the archive-weight diagonal so the gaia_archive strategy can
        # retrieve it (the full covariance matrix itself lives on the sets).
        self._cache_archive_weights(collection)

        return collection

    def get_gaia_ephemeris(self, geocentric: bool = True):
        """Get tabulated ephemeris settings generated from the archived Gaia state vectors.

        Args:
            geocentric: If true, use geocentric Gaia states (recommended). If
                false, use barycentric states.

        Returns:
            EphemerisSettings: Tabulated ephemeris settings.
        """
        return self._astrometry.get_gaia_ephemeris_settings(geocentric=geocentric)

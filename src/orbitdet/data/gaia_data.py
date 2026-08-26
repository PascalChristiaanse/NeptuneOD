"""
Retrieve Gaia FPR astrometry from the archives.

This module provides the :class:`GaiaQuery` class, which retrieves astrometric
observations of solar-system objects (asteroids, and now Triton) from the Gaia
archive.  The class caches its pulls from the Gaia archive to reduce internet
traffic, and converts the raw archive columns into the units and conventions
expected by Tudat.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from astroquery.gaia import Gaia
from scipy.constants import arcsec
from scipy.linalg import block_diag
from tudatpy.astro.time_representation import (
    DateTime,
    TCB_to_TDB,
    julian_day_to_seconds_since_epoch,
)
from tudatpy.constants import ASTRONOMICAL_UNIT
from tudatpy.dynamics.environment import SystemOfBodies
from tudatpy.dynamics.environment_setup import ephemeris
from tudatpy.estimation import observations
from tudatpy.estimation.observable_models_setup import links, model_settings

from orbitdet.data.light_deflection import relativistic_light_deflection
from orbitdet.data.photocenter import photocenter_offset_spherical
from orbitdet.transformations import convert_radec_frame

logger = logging.getLogger(__name__)

# Reference time J2010.0 in Julian days
J2010 = 2455197.5
# See e.g. Klioner (2003)
TIME_SCALE_CORRECTION = 1 - 1.550519768e-8
DAY_IN_S = 86400.0

TABLE_COLUMNS = [
    "epoch",
    "ra",
    "dec",
    "number_mp",
    "transit_id",
    "position_angle_scan",
    "ra_error_systematic",
    "dec_error_systematic",
    "ra_dec_correlation_systematic",
    "ra_error_random",
    "dec_error_random",
    "ra_dec_correlation_random",
    "x_gaia",
    "y_gaia",
    "z_gaia",
    "vx_gaia",
    "vy_gaia",
    "vz_gaia",
    "x_gaia_geocentric",
    "y_gaia_geocentric",
    "z_gaia_geocentric",
    "vx_gaia_geocentric",
    "vy_gaia_geocentric",
    "vz_gaia_geocentric",
]

CATALOG_NAMES = {
    "DR2": "gaiadr2.sso_observation",
    "DR3": "gaiadr3.sso_observation",
    "FPR": "gaiafpr.sso_observation",
}

# Registry of position angle of scan (radians) keyed by the observation epoch
# (seconds since J2000 TDB).  The Tudat ``ObservationCollection`` / 
# ``SingleObservationSet`` pybind classes do not allow attaching arbitrary
# attributes, so we carry the scan angles alongside via this module-level map.
# Epochs are unique per observation, so they form a stable key that survives
# merging of observation collections.
_SCAN_ANGLE_REGISTRY: dict[tuple[float, ...], np.ndarray] = {}


def register_scan_angles(epochs: np.ndarray | list[float], scan_angles: np.ndarray) -> None:
    """Store the per-observation scan angles keyed by their epochs.

    Args:
        epochs: Observation epochs in seconds since J2000 TDB.
        scan_angles: Position angle of scan (radians) for each observation.
    """
    _SCAN_ANGLE_REGISTRY[tuple(float(e) for e in epochs)] = np.asarray(scan_angles, dtype=float)


def get_scan_angles_for_epochs(epochs: np.ndarray | list[float]) -> np.ndarray | None:
    """Return the scan angles (radians) matching the given epochs, if registered.

    Performs an exact match first (fast path for a collection built from a
    single ``GaiaQuery``).  If no exact match is found, falls back to a
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


def build_gaia_tabulated_state_history(
    source_ids: list[int],
    cache_file: str | Path | None = None,
    geocentric: bool = False,
    filter_outcomes: bool = True,
) -> dict[float, np.ndarray]:
    """Build a Gaia tabulated ephemeris state history from the archive.

    Retrieves the Gaia state vectors archived alongside the observations and
    packs them into a Tudat-compatible state history (keys = seconds since
    J2000 TDB, values = 6x1 position/velocity in SI units).

    Args:
        source_ids: Gaia ``source_id`` values used to select the observations.
        cache_file: Optional path to a pickle cache to avoid re-querying the
            archive.
        geocentric: If True, use the geocentric Gaia states (frame origin
            ``Earth``).  If False (default), use the barycentric states (frame
            origin ``SSB``).
        filter_outcomes: If True, keep only rows with
            ``astrometric_outcome_ccd == 1`` and ``astrometric_outcome_transit
            == 1``.  If False, use all rows returned by the archive (no
            filtering).

    Returns:
        dict[float, np.ndarray]: State history with epochs (seconds since J2000
        TDB) as keys and 6x1 [position (m), velocity (m/s)] states as values.
    """
    query = GaiaQuery()
    query.retrieve_data(
        source_ids=source_ids, cache_file=cache_file, filter_outcomes=filter_outcomes
    )

    state_vector_labels = ["x_gaia", "y_gaia", "z_gaia", "vx_gaia", "vy_gaia", "vz_gaia"]
    if geocentric:
        state_vector_labels = [label + "_geocentric" for label in state_vector_labels]

    table = query.observation_table
    epochs = table["epoch"].to_numpy(dtype=float)
    states = table[state_vector_labels].to_numpy(dtype=float)

    # _convert_units already scaled the states to SI (m, m/s)
    state_history: dict[float, np.ndarray] = {}
    for index, time_value in enumerate(epochs):
        state = states[index].reshape(6, 1)
        state_history[float(time_value)] = state

    logger.info(
        "Built Gaia tabulated ephemeris state history with %d samples.", len(state_history)
    )
    return state_history


class GaiaQuery:
    """Retrieve and hold Gaia astrometric observations of solar-system objects.

    The class supports both asteroids (queried by MPC number) and Triton
    (queried by Gaia ``source_id``).  Raw archive pulls are cached to disk as a
    pickle to reduce internet traffic on subsequent calls.

    Attributes:
        _table (pd.DataFrame): Holds astrometry and metadata after calling
            :meth:`retrieve_data` or :meth:`retrieve_data_locally`.
    """

    def __init__(self):
        """Create a Gaia query object."""
        self._table = pd.DataFrame()

    @property
    def observation_table(self) -> pd.DataFrame:
        """Return a copy of the observation table."""
        return self._table.copy()

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
        return pd.unique(self.observation_table["number_mp"])

    def copy(self) -> "GaiaQuery":
        """Return a deep copy of the query object."""
        return copy.deepcopy(self)

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
        if self._table.empty:
            raise RuntimeError("No observations loaded")

        # Check if Gaia is in bodies
        if not bodies.does_body_exist("Gaia") or bodies.get("Gaia").ephemeris is None:
            raise ValueError(
                "Gaia satellite and associated ephemeris must be loaded in SystemOfBodies"
            )

        if not bodies.does_body_exist(target_name) or bodies.get(target_name).ephemeris is None:
            raise ValueError(
                f"Target body '{target_name}' and associated ephemeris must be loaded in SystemOfBodies"
            )

        output_frame = output_frame or input_frame

        # Convert RA/DEC columns to the requested output frame. The Gaia table
        # stores angles in radians and epochs in seconds since J2000 TDB, but
        # the ICRS->J2000 frame-bias and any subsequent SPICE conversion need
        # time as seconds since J2000 (Astropy FrameBias uses TDB epoch).
        table = convert_radec_frame(
            self.observation_table,
            "ra",
            "dec",
            input_frame=input_frame,
            output_frame=output_frame,
            time_column="epoch",
            ra_wrap=False,
            angle_unit="rad",
        )

        link_ends = {}
        link_ends[links.transmitter] = links.body_origin_link_end_id(target_name)
        link_ends[links.receiver] = links.body_origin_link_end_id("Gaia")

        observation_angles = table.loc[:, ["ra", "dec"]].to_numpy()
        observation_times = table["epoch"].to_numpy()

        observation_set = observations.create_single_observation_set(
            model_settings.angular_position_type,
            link_ends,
            observation_angles,
            observation_times,
            links.receiver,
        )

        # Per-observation weights from the Gaia covariance (random + systematic)
        weights = self._per_observation_weights(table)
        observation_set.set_tabulated_weights(weights.reshape(-1, 1))

        observation_collection = observations.ObservationCollection([observation_set])

        # Register the per-observation scan angles (radians) keyed by epoch so the
        # residual visualisations can rotate RA/Dec residuals into the Gaia
        # along-scan / across-scan frame without re-querying the archive.
        register_scan_angles(
            observation_times,
            table["position_angle_scan"].to_numpy(dtype=float),
        )

        return observation_collection

    def _per_observation_weights(self, table: pd.DataFrame) -> np.ndarray:
        """Compute the per-observation weight vector from the Gaia covariance.

        The Gaia covariance within a transit has a random (independent) and a
        systematic (fully correlated) component.  For each transit we build the
        full covariance, invert it, and take the diagonal of the resulting
        weight matrix as the per-observation weight vector (this approximately
        accounts for the intra-transit systematic correlation).

        Args:
            table: Observation table with the error columns converted to
                radians (as produced by :meth:`_convert_units`).

        Returns:
            np.ndarray: (2 * n_observations,) weight vector.
        """
        transit_ids_unique = pd.unique(table["transit_id"])
        weight_blocks = []

        for transit_id in transit_ids_unique:
            transit = table[table["transit_id"] == transit_id]

            # Error components
            uncertainty_random = transit[
                ["ra_error_random", "dec_error_random", "ra_dec_correlation_random"]
            ]
            sigma_ra_r, sigma_dec_r, corr_r = uncertainty_random.to_numpy().T

            sigma_ra_s = np.mean(transit["ra_error_systematic"])
            sigma_dec_s = np.mean(transit["dec_error_systematic"])
            corr_s = np.mean(transit["ra_dec_correlation_systematic"])

            # Random component for observation AF1-9
            covariance_random = [
                np.array(
                    [
                        [sigma_ra_r[ii] ** 2, corr_r[ii] * sigma_ra_r[ii] * sigma_dec_r[ii]],
                        [corr_r[ii] * sigma_ra_r[ii] * sigma_dec_r[ii], sigma_dec_r[ii] ** 2],
                    ]
                )
                for ii in range(len(transit))
            ]
            covariance_random = block_diag(*covariance_random)

            # Systematic component over transit
            covariance_systematic_sub = np.array(
                [
                    [sigma_ra_s**2, corr_s * sigma_ra_s * sigma_dec_s],
                    [corr_s * sigma_ra_s * sigma_dec_s, sigma_dec_s**2],
                ]
            )
            covariance_systematic = np.tile(
                covariance_systematic_sub, (len(transit), len(transit))
            )

            weight_block = np.linalg.inv(covariance_random + covariance_systematic)
            weight_blocks.append(np.diag(weight_block))

        return np.concatenate(weight_blocks)

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
        print("hi")
        if not bodies.does_body_exist(target_name) or bodies.get(target_name).ephemeris is None:
            raise ValueError(
                f"Correction not possible for {target_name}. Body needs to be in SystemOfBodies "
                "with loaded ephemeris"
            )

        if correct_photocenter:
            if diameter <= 0.0:
                raise ValueError(
                    f"A positive 'diameter' (m) is required for the photocenter "
                    f"correction of {target_name}."
                )
            logger.info("Applying photocenter correction to %s observations.", target_name)
            corrections = photocenter_offset_spherical(
                target_name, self.observation_table, bodies, diameter
            )
            self._table.loc[:, ["ra", "dec"]] += corrections

        if light_deflection:
            if not all(bodies.does_body_exist(body) for body in light_deflection):
                raise ValueError("Light deflection bodies missing from bodies object")
            logger.info(
                "Applying relativistic light deflection to %s observations "
                "(bodies: %s).",
                target_name,
                list(light_deflection),
            )
            corrections = relativistic_light_deflection(
                target_name, self.observation_table, bodies, list(light_deflection)
            )
            self._table.loc[:, ["ra", "dec"]] += corrections

    def retrieve_data(
        self,
        mpc_numbers: tuple[int] | list[int] | None = None,
        source_ids: list[int] | None = None,
        catalog: str = "FPR",
        username: str | None = None,
        password: str | None = None,
        cache_file: str | Path | None = None,
        filter_outcomes: bool = True,
    ) -> None:
        """Retrieve astrometric observations through astroquery.

        Observations are stored in the observation table attribute.  If
        ``cache_file`` is provided and exists, the cached table is loaded
        instead of querying the archive.

        Args:
            mpc_numbers: List of asteroid MPC numbers to retrieve.
            source_ids: List of Gaia ``source_id`` values to retrieve (e.g. for
                Triton).  Mutually exclusive with ``mpc_numbers``.
            catalog: Which catalog to use. Options: DR2, DR3, FPR.
            username: Username for the Gaia archives (optional).
            password: Password for the Gaia archives (optional).
            cache_file: Path to a pickle file used to cache the raw archive pull.
            filter_outcomes: If True, keep only rows with
                ``astrometric_outcome_ccd == 1`` and ``astrometric_outcome_transit
                == 1``.  If False, use all rows returned by the archive (no
                filtering).
        """
        if (mpc_numbers is None) == (source_ids is None):
            raise ValueError("Provide exactly one of mpc_numbers or source_ids")

        if catalog not in CATALOG_NAMES:
            raise ValueError(
                f"Catalog not available. Catalog options are: {', '.join(CATALOG_NAMES.keys())}"
            )

        # Try to load from cache first
        if cache_file is not None:
            cache_path = Path(cache_file)
            if cache_path.exists():
                logger.info("Loading Gaia observations from cache %s", cache_path)
                table = pd.read_pickle(cache_path)
                self._table = table
                return

        # Define query to database
        query_catalog = CATALOG_NAMES[catalog]

        if mpc_numbers is not None:
            query_mpc_numbers = ", ".join(str(mpc_number) for mpc_number in mpc_numbers)
            where_clause = f"number_mp IN ({query_mpc_numbers})"
        else:
            query_source_ids = ", ".join(str(source_id) for source_id in source_ids)
            where_clause = f"source_id IN ({query_source_ids})"

        # Apply the astrometric-outcome filter only when requested.  The FPR
        # ``astrometric_outcome_ccd`` / ``astrometric_outcome_transit`` = 1 rows
        # are the ones used in the FPR astrometric solution; other values (e.g. 2
        # for multiple/non-unique solutions) are dropped by default.
        outcome_filter = ""
        if filter_outcomes:
            outcome_filter = (
                "\n            AND astrometric_outcome_ccd = 1\n"
                "            AND astrometric_outcome_transit = 1"
            )

        login_provided = username is not None and password is not None
        if login_provided:
            Gaia.login(user=username, password=password)

        try:
            query = f"""
            SELECT *
            FROM {query_catalog}
            WHERE {where_clause}{outcome_filter}
            ORDER BY epoch ASC
            """
            job = Gaia.launch_job_async(query)
            table = job.get_results()
        except Exception as err:
            raise RuntimeError(f"Error while retrieving astrometric observations: \n{err}") from err

        table = table.to_pandas()  # Convert astropy table to dataframe
        if table.empty:
            raise LookupError(f"No observations found for query {where_clause}")

        # Pre-process data
        table = self._convert_units(table)
        table = table.reset_index(drop=True)
        assert table["epoch"].is_monotonic_increasing  # Sanity check for ordering by epoch

        # Store
        self._table = table

        # Cache the table
        if cache_file is not None:
            cache_path = Path(cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            table.to_pickle(cache_path)
            logger.info("Cached Gaia observations to %s", cache_path)

    def retrieve_data_locally(
        self,
        mpc_numbers: list[int] | None = None,
        source_ids: list[int] | None = None,
        file_path: str = "",
        filter_outcomes: bool = True,
    ) -> None:
        """Retrieve astrometric observations from local .csv files.

        Files 00 through 19 must be saved in the directory.  The archive is
        saved as a .pkl in the corresponding directory to make subsequent calls
        faster.

        Args:
            mpc_numbers: List of asteroid MPC numbers to retrieve.
            source_ids: List of Gaia ``source_id`` values to retrieve.
            file_path: Path to the .csv files.
            filter_outcomes: If True, keep only rows with
                ``astrometric_outcome_ccd == 1`` and ``astrometric_outcome_transit
                == 1``.  If False, use all rows (no filtering).
        """
        if (mpc_numbers is None) == (source_ids is None):
            raise ValueError("Provide exactly one of mpc_numbers or source_ids")

        # Load from cached pkl file
        try:
            table = pd.read_pickle(file_path + "gaia_archive_dataframe.pkl")
        except FileNotFoundError:
            table = pd.DataFrame()
            for i in range(20):
                print(f"Loading file number {i}/19...")
                file_number = str(i) if i >= 10 else "0" + str(i)
                file_name = "SsoObservation_" + file_number + ".csv"
                table_chunk = pd.read_csv(file_path + file_name, comment="#")
                table = pd.concat([table, table_chunk], ignore_index=True)

            # Cache for future calls
            table.to_pickle(file_path + "gaia_archive_dataframe.pkl")

        # Filter and sort
        if mpc_numbers is not None:
            table = table[table["number_mp"].isin(mpc_numbers)]
        else:
            table = table[table["source_id"].isin(source_ids)]
        if filter_outcomes:
            table = table[table["astrometric_outcome_ccd"] == 1]
            table = table[table["astrometric_outcome_transit"] == 1]
        if table.empty:
            raise LookupError(f"No observations found for query")
        table = table[TABLE_COLUMNS]
        table = table.sort_values(by="epoch")

        # Convert units and save
        table = self._convert_units(table)
        table = table.reset_index(drop=True)

        self._table = table

    def _convert_units(self, table: pd.DataFrame) -> pd.DataFrame:
        """Convert the table columns into the correct format for Tudat."""
        # Convert epoch to seconds since J2000
        func = lambda jd: julian_day_to_seconds_since_epoch(jd + J2010)
        table["epoch"] = table["epoch"].apply(func)

        # Convert TCB to TDB epoch
        table["epoch"] = table["epoch"].apply(TCB_to_TDB)

        # Convert angles to rad
        table["ra"] = np.deg2rad(table["ra"]) % (2 * np.pi)
        table["dec"] = np.deg2rad(table["dec"])
        table["position_angle_scan"] = np.deg2rad(table["position_angle_scan"])

        # Convert mas to radians
        table[
            [
                "ra_error_random",
                "dec_error_random",
                "ra_error_systematic",
                "dec_error_systematic",
            ]
        ] *= arcsec / 1e3

        # Remove the cos delta factor from the right ascension uncertainty values
        table["ra_error_random"] /= np.cos(table["dec"])
        table["ra_error_systematic"] /= np.cos(table["dec"])

        # Convert Gaia state vectors to SI, apply correction to position vectors
        # due to time scale change
        pos_names = [
            "x_gaia",
            "y_gaia",
            "z_gaia",
            "x_gaia_geocentric",
            "y_gaia_geocentric",
            "z_gaia_geocentric",
        ]
        table.loc[:, pos_names] *= ASTRONOMICAL_UNIT * TIME_SCALE_CORRECTION
        vel_names = [
            "vx_gaia",
            "vy_gaia",
            "vz_gaia",
            "vx_gaia_geocentric",
            "vy_gaia_geocentric",
            "vz_gaia_geocentric",
        ]
        table.loc[:, vel_names] *= ASTRONOMICAL_UNIT / DAY_IN_S

        return table

    def filter(
        self,
        epoch_start: float | datetime | DateTime,
        epoch_end: float | datetime | DateTime,
    ) -> None:
        """Filter the observations after they have been loaded (in-place)."""
        if len(self._table) == 0:
            raise Exception("No observations loaded")

        # Convert parameters to seconds since J2000
        if isinstance(epoch_start, datetime) and isinstance(epoch_end, datetime):
            epoch_start = DateTime.from_python_datetime(epoch_start)
            epoch_end = DateTime.from_python_datetime(epoch_end)

        if isinstance(epoch_start, DateTime) and isinstance(epoch_end, DateTime):
            epoch_start = epoch_start.to_epoch()
            epoch_end = epoch_end.to_epoch()

        # Find observations in time span
        obs_in_timespan = self._table.query("@epoch_start <= epoch <= @epoch_end")

        if not obs_in_timespan.empty:
            self._table = obs_in_timespan
        else:
            raise Exception("No observations left after filtering")

    def summary(self) -> None:
        """Print a convenient summary of the astrometric observations."""
        if len(self._table) == 0:
            print("Observations not loaded")
            return

        print("Summary:")
        print(f"Observations for {len(self.mpc_numbers)} objects:")

        first_epoch = DateTime.from_epoch(self.epoch_start)
        final_epoch = DateTime.from_epoch(self.epoch_end)
        print(
            f"First observation yy/mm/dd: {first_epoch.year}, {first_epoch.month}, {first_epoch.day}"
        )
        print(
            f"Final observation yy/mm/dd: {final_epoch.year}, {final_epoch.month}, {final_epoch.day}"
        )

        for mpc_number in self.mpc_numbers:
            print(f"\nMinor planet {mpc_number}:")
            table_single_obj = self.observation_table.query("number_mp == @mpc_number")

            nr_of_observations = len(table_single_obj)
            print(f"Number of observations: {nr_of_observations}")

            epochs_as_list = table_single_obj["epoch"].to_list()
            first_epoch = DateTime.from_epoch(epochs_as_list[0])
            final_epoch = DateTime.from_epoch(epochs_as_list[-1])

            print(
                f"First observation yy/mm/dd: {first_epoch.year}, {first_epoch.month}, {first_epoch.day}"
            )
            print(
                f"Final observation yy/mm/dd: {final_epoch.year}, {final_epoch.month}, {final_epoch.day}"
            )

    def get_gaia_ephemeris(self, geocentric: bool = True):
        """Get tabulated ephemeris settings generated from the archived Gaia state vectors.

        Args:
            geocentric: If true, use geocentric Gaia states (recommended). If
                false, use barycentric states.

        Returns:
            EphemerisSettings: Tabulated ephemeris settings.
        """
        state_vector_labels = ["x_gaia", "y_gaia", "z_gaia", "vx_gaia", "vy_gaia", "vz_gaia"]
        if geocentric:
            state_vector_labels = [label + "_geocentric" for label in state_vector_labels]

        # Create dict of state vectors
        table = self.observation_table
        epochs = table["epoch"].to_numpy()
        states = table[state_vector_labels].to_numpy()
        gaia_state_history = dict(zip(epochs, states))

        settings = ephemeris.tabulated(
            gaia_state_history,
            frame_origin="Earth" if geocentric else "SSB",
            frame_orientation="J2000",
        )
        return settings
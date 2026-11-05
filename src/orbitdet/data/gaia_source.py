"""Produce a standard observation frame from Gaia FPR astrometry.

Gaia's astrometry reaches the pipeline through the official tudatpy
``GaiaAstrometry`` interface, which owns retrieval from the archive, the unit
conversions and - importantly - the *full transit covariance*.  This module adds
only the step that turns its observation table into the standard frame, so Gaia
joins NSDB and Voyager as a source that feeds the same factories.

What the frame can and cannot carry
-----------------------------------
The measurements, epochs, uncertainties and scan angle are all per-observation
quantities, so they map onto the frame directly and the scan angle finally has a
home: it used to sit in a module-level registry keyed by the epoch tuple, because
"the Tudat ``SingleObservationSet`` pybind classes do not allow attaching
arbitrary attributes".  A dataframe column is the natural place for it and
removes that registry.

The **transit covariance cannot** be represented in a wide dataframe, and this is
structural rather than an omission.  Gaia reports the covariance of a whole
transit, which is an ``(n, n)`` block across the observations of that transit,
not a per-row uncertainty.  A dataframe column holds one scalar per row.  So the
reader records the diagonal (``ra_error_random`` and friends are per-row) and the
factory still asks ``GaiaAstrometry`` for the matrix, which is what
``set_full_weight_matrix`` consumes.  Trying to force the blocks through a frame
would lose exactly the information that makes Gaia weighting worth doing.

Pipeline it completes:

.. code-block:: text

    archive / parquet / CSV chunks
        -> GaiaAstrometry     (retrieve, filter, units)
        -> this module        (table to standard frame)
        -> observation factory
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from orbitdet.preprocessing.catalogues import normalize_catalogue
from orbitdet.preprocessing.schema import DataSource, ObservationType, build_frame

logger = logging.getLogger(__name__)

__all__ = ["GaiaSource"]

#: Column order of the archived per-observation errors, in ``GaiaAstrometry``'s
#: own naming.  Kept here rather than imported from the archive column list so
#: this module states the mapping it relies on.
_RANDOM_ERROR_COLUMNS = ("ra_error_random", "dec_error_random", "ra_dec_correlation_random")
_SYSTEMATIC_ERROR_COLUMNS = (
    "ra_error_systematic",
    "dec_error_systematic",
    "ra_dec_correlation_systematic",
)


class GaiaSource:
    """Read Gaia FPR astrometry as a standard observation frame.

    Attributes:
        cfg: Experiment configuration.
        query: An existing :class:`~orbitdet.data.gaia.GaiaQuery`, reused when the
            caller has already retrieved the data.  Retrieval is expensive and
            cached, but building the query twice still means rebuilding the
            table twice.

    Examples:
        >>> source = GaiaSource(cfg)                          # doctest: +SKIP
        >>> frame = source.frame_from_config(dataset_cfg)     # doctest: +SKIP
    """

    def __init__(self, cfg: DictConfig | None = None, *, query=None, system_of_bodies=None) -> None:
        self.cfg = cfg
        self.query = query
        self.system_of_bodies = system_of_bodies
        #: Covariance blocks per object key, captured when the table is
        #: retrieved.  Exposed so the factory can attach the full transit matrix,
        #: which has no representation in a wide dataframe.
        self.covariance_blocks: dict[int, list] = {}
        #: Random and systematic covariance blocks per object key, kept apart so
        #: a strategy can re-weight the two components independently.  The blocks
        #: in :attr:`covariance_blocks` are their sum.
        self.covariance_components: dict[int, tuple[list, list]] = {}

    def observation_table(self, dataset_cfg) -> pd.DataFrame:
        """Retrieve (or reuse) Gaia's observation table for a dataset.

        Args:
            dataset_cfg: A Gaia dataset config, declaring ``source_ids`` or
                ``mpc_numbers``.

        Returns:
            The observation table, per-observation, in radians and TDB seconds.

        Raises:
            ValueError: If the config names neither selection key.
        """
        if self.query is None:
            from .gaia import GaiaQuery

            self.query = GaiaQuery()

            source_ids = dataset_cfg.get("source_ids")
            mpc_numbers = dataset_cfg.get("mpc_numbers")
            filter_outcomes = bool(dataset_cfg.get("filter_outcomes", True))
            file_path = dataset_cfg.get("local_archive")

            if file_path:
                self.query.retrieve_data_locally(
                    mpc_numbers=list(mpc_numbers) if mpc_numbers else None,
                    source_ids=list(source_ids) if source_ids else None,
                    file_path=str(file_path),
                    filter_outcomes=filter_outcomes,
                )
            elif source_ids is not None:
                self.query.retrieve_data(
                    source_ids=list(source_ids), filter_outcomes=filter_outcomes
                )
            elif mpc_numbers is not None:
                self.query.retrieve_data(
                    mpc_numbers=list(mpc_numbers), filter_outcomes=filter_outcomes
                )
            else:
                raise ValueError(
                    f"Gaia dataset {dataset_cfg.get('identifier', '?')} must specify one of "
                    f"'source_ids', 'mpc_numbers' or 'local_archive'."
                )

        table = self.query.observation_table
        if table.empty:
            raise LookupError(
                f"Gaia dataset {dataset_cfg.get('identifier', '?')}: the retrieved table is "
                f"empty, so no frame can be built."
            )

        # Capture the per-transit covariance blocks now, because they belong to
        # the table this query holds and have no dataframe representation.  Doing
        # it here keeps the astrometry object and the numbers derived from it in
        # step; recomputing them later from a stale query is how the matrix and
        # the observations drift apart.
        for object_key in self.query.astrometry.object_keys_in_table:
            self.covariance_blocks[int(object_key)] = (
                self.query.astrometry._get_observation_covariance(object_key)
            )
            self.covariance_components[int(object_key)] = (
                self.query.astrometry.get_observation_covariance_components(object_key)
            )

        self._apply_corrections(dataset_cfg)
        return self.query.observation_table

    def _apply_corrections(self, dataset_cfg) -> None:
        """Apply the astrometric corrections the config asks for, in place.

        Light deflection and the photocenter offset shift the archived RA/Dec
        onto the target's centre of mass.  They need ephemerides, so the config's
        request is honoured only when a system of bodies was supplied; the
        corrections are skipped rather than silently wrong when it was not.

        Args:
            dataset_cfg: A Gaia dataset config.
        """
        target = str(dataset_cfg.get("target") or "Triton")
        correct_photocenter = bool(dataset_cfg.get("correct_photocenter", False))
        light_deflection = dataset_cfg.get("light_deflection", ("Sun",))
        if isinstance(light_deflection, str):
            light_deflection = [light_deflection] if light_deflection.strip() else []
        else:
            light_deflection = list(light_deflection or [])
        diameter = float(dataset_cfg.get("diameter", 0.0) or 0.0)

        if not (correct_photocenter or light_deflection):
            return

        if self.system_of_bodies is None:
            raise ValueError(
                f"Gaia dataset {dataset_cfg.get('identifier', '?')} requests astrometric "
                f"corrections, which need a system of bodies to compute from ephemerides. "
                f"Pass one to GaiaSource, or disable 'correct_photocenter' and "
                f"'light_deflection'."
            )

        if correct_photocenter and diameter <= 0.0:
            raise ValueError(
                f"Gaia dataset {dataset_cfg.get('identifier', '?')}: a positive 'diameter' "
                f"(m) is required for the photocenter correction of {target}."
            )

        self.query.correct_observations(
            target,
            self.system_of_bodies,
            light_deflection=light_deflection,
            correct_photocenter=correct_photocenter,
            diameter=diameter,
        )

    def frame_from_config(self, dataset_cfg) -> pd.DataFrame:
        """Build the standard frame described by a Gaia dataset config.

        The returned frame is *not* orientation-converted here.  Its
        ``first_value`` / ``second_value`` hold the archived RA/Dec in the
        ``input_frame`` the config names (ICRS by default), and ``original_equinox``
        records it, so the shared
        :func:`orbitdet.preprocessing.frames.convert_standard_frame` performs the
        rotation exactly as it does for NSDB.

        Args:
            dataset_cfg: A Gaia dataset config.

        Returns:
            A frame satisfying :mod:`orbitdet.preprocessing.schema`.
        """
        table = self.observation_table(dataset_cfg)
        return self.frame_from_table(table, dataset_cfg)

    def frame_from_table(self, table: pd.DataFrame, dataset_cfg) -> pd.DataFrame:
        """Build the standard frame from an already-retrieved Gaia table.

        Split out from :meth:`frame_from_config` because
        :class:`~orbitdet.data.gaia.GaiaQuery` must apply its astrometric
        corrections to the table *before* the frame is built, and that happens at
        the call site which has a system of bodies.

        Args:
            table: Gaia's per-observation table, in radians and TDB seconds.
            dataset_cfg: A Gaia dataset config.

        Returns:
            A frame satisfying :mod:`orbitdet.preprocessing.schema`.
        """
        target = str(dataset_cfg.get("target") or "Triton")
        identifier = str(dataset_cfg.get("identifier") or "")

        epochs = pd.to_numeric(table["epoch"], errors="coerce")
        ra = pd.to_numeric(table["ra"], errors="coerce")
        dec = pd.to_numeric(table["dec"], errors="coerce")

        # A row with no epoch or no coordinate is not an observation.  Gaia's own
        # filtering (astrometric_outcome) usually removes these, but a local
        # archive read with filtering disabled can still hand us some.
        usable = np.isfinite(epochs) & np.isfinite(ra) & np.isfinite(dec)
        dropped = int((~usable).sum())
        if dropped:
            logger.info(
                "Gaia dataset %s: dropped %d row(s) with no epoch or no RA/Dec.",
                identifier,
                dropped,
            )

        scan_angle = self._optional_column(table, "position_angle_scan", usable)
        first_uncertainty = self._combined_uncertainty(table, "ra", usable)
        second_uncertainty = self._combined_uncertainty(table, "dec", usable)

        equinox = str(dataset_cfg.get("epoch_of_equinox") or "ICRS")
        records = [
            {
                "observed_object": target,
                # Gaia has no MPC code; the schema's convention for that is the
                # observer's body name, and it is what the factory's link end and
                # the weights router key on.
                "observatory_code": "Gaia",
                "data_source": DataSource.GAIA_ARCHIVE.value,
                "source_set_id": identifier,
                "reference": str(dataset_cfg.get("reference") or "Gaia Collaboration, FPR"),
                "observation_type": ObservationType.ABSOLUTE.value,
                "coordinate_kind": "ra_dec",
                # Gaia observes from a spacecraft, not from the ground.
                "frame_center": "spacecraft",
                "original_equinox": equinox,
                # Gaia's astrometry is reduced against the Gaia catalogue itself,
                # which is the ICRS realisation, so the reduction *is* the tie.
                "original_reference_frame": "astrometric",
                "original_reduction": normalize_catalogue("Gaia DR2").value,
                "receptor_type": "ccd",
                "epoch_tdb": float(epoch_value),
                "first_value": float(ra_value),
                "second_value": float(dec_value),
                "first_unit": "rad",
                "second_unit": "rad",
                "first_uncertainty": first_uncertainty[index],
                "second_uncertainty": second_uncertainty[index],
                "scan_angle": scan_angle[index],
            }
            for index, (epoch_value, ra_value, dec_value) in enumerate(
                zip(epochs[usable].tolist(), ra[usable].tolist(), dec[usable].tolist())
            )
        ]

        # ``build_frame`` rejects unknown keys, and ``None`` is a legal value for
        # an optional column, so the scan angle may be recorded as None above when
        # the archive did not supply it.
        return build_frame(records)

    # ------------------------------------------------------------------
    # Column helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _optional_column(
        table: pd.DataFrame, column: str, usable: np.ndarray
    ) -> list[float | None]:
        """Return a column's usable values as a list, or ``None`` per row if absent."""
        if column not in table.columns:
            return [None] * int(usable.sum())
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype="float64")[usable]
        return [None if not np.isfinite(value) else float(value) for value in values.tolist()]

    @staticmethod
    def _combined_uncertainty(
        table: pd.DataFrame, coordinate: str, usable: np.ndarray
    ) -> list[float | None]:
        """Combine the random and systematic errors of one coordinate.

        Gaia reports the two components separately and they add in quadrature.
        The correlation between coordinates is deliberately not folded in here:
        it cannot be represented per-row, and the factory gets the full transit
        matrix from ``GaiaAstrometry`` for the weight it actually applies.  This
        value is the marginal one-sigma on the coordinate, which is what a
        diagnostic plot wants.

        Args:
            table: Gaia's observation table.
            coordinate: ``"ra"`` or ``"dec"``.
            usable: Mask of the rows that became observations.

        Returns:
            One combined sigma per usable row, or ``None`` where neither
            component is present.
        """
        random_column = f"{coordinate}_error_random"
        systematic_column = f"{coordinate}_error_systematic"
        have = [c for c in (random_column, systematic_column) if c in table.columns]
        if not have:
            return [None] * int(usable.sum())

        total = np.zeros(int(usable.sum()), dtype="float64")
        seen = np.zeros(int(usable.sum()), dtype=bool)
        for column in have:
            values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype="float64")[usable]
            finite = np.isfinite(values)
            total[finite] += values[finite] ** 2
            seen |= finite

        return [float(np.sqrt(value)) if ok else None for value, ok in zip(total.tolist(), seen)]

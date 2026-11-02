"""Produce a standard observation frame from the Voyager 2 manual tables.

Voyager 2 observed Triton across the 1989 Neptune encounter, and those
measurements survive as two hand-transcribed tables (Jacobson 1991) rather than
as a machine-readable archive: absolute right ascension and declination, already
reduced, with the spacecraft's own trajectory tabulated alongside them.

The same two tables also back the tabulated Voyager ephemeris, so this reader
deliberately reuses :func:`orbitdet.data.voyager_data.load_and_merge_voyager_tables`
rather than re-reading the CSV.  One parse, one merge, two consumers - and the
merge is what attaches the epochs, so duplicating it would risk the observations
and the ephemeris drifting onto different time grids.

The pipeline it completes:

.. code-block:: text

    data table + ancillary table
        -> voyager_data       (merge, epochs)
        -> this module        (config to standard frame)
        -> observation factory

What is deliberately *not* mapped
---------------------------------
The tables carry per-observation ``sigma_alpha cos(delta)`` and ``sigma_delta``
columns, and they are left out of the frame.  Two reasons: the columns are read
as text and four of the 413 entries are corrupt (``'0.0005 051'``,
``'40 0.00042'``), and their unit is ambiguous - the header says arcseconds, but
the values are ~3.5e-4, which is 0.35 mas as arcseconds (implausibly precise for
a 1979 vidicon) against 1.26 arcsec as degrees (which matches both the documented
Voyager accuracy and the 1 arcsec hardcoded in ``conf/weighting``).  Weighting is
owned by :mod:`orbitdet.observations.weighting` and currently applies fixed
Voyager sigmas from config, so nothing consumes these columns today; recording
them under an unverified unit would be worse than omitting them.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from orbitdet.preprocessing.catalogues import normalize_catalogue
from orbitdet.preprocessing.observatories import normalize_observatory_code
from orbitdet.preprocessing.schema import DataSource, ObservationType, build_frame

from .nsdb_formatter import (
    normalize_frame_center,
    normalize_receptor,
    normalize_reference_frame,
    set_ra_dec_columns,
)

logger = logging.getLogger(__name__)

__all__ = ["VoyagerSource"]


class VoyagerSource:
    """Read the Voyager 2 manual tables as a standard observation frame.

    Attributes:
        cfg: Experiment configuration.  Needed because locating the tables runs
            through the shared merge in :mod:`orbitdet.data.voyager_data`, which
            inspects ``bodies_to_create`` to decide whether an unmatched merge
            key is worth warning about.

    Examples:
        >>> source = VoyagerSource(cfg)                       # doctest: +SKIP
        >>> frame = source.frame_from_config(dataset_cfg)     # doctest: +SKIP
    """

    def __init__(self, cfg: DictConfig | None = None) -> None:
        self.cfg = cfg

    def frame_from_config(self, dataset_cfg) -> pd.DataFrame:
        """Build the standard frame described by a Voyager dataset config.

        Args:
            dataset_cfg: A Voyager dataset config, i.e. one declaring
                ``type: voyager`` with ``data.file`` and ``ancillary_files.file``.

        Returns:
            A frame satisfying :mod:`orbitdet.preprocessing.schema`.

        Raises:
            ValueError: If the config names no observable satellite, or the
                descriptor's coordinates are not RA/Dec.
        """
        from .voyager_data import load_and_merge_voyager_tables

        merged = load_and_merge_voyager_tables(self.cfg, dataset_cfg)

        # Locate the RA/Dec pair with the same prose-heading search NSDB uses,
        # rather than hardcoding "alpha (Deg)"/"delta (Deg)".  The reader is
        # reused because the tables spell the two quantities in degrees and the
        # search also knows how to assemble the sexagesimal component layout
        # should a future transcription use it.
        ra_column, dec_column = set_ra_dec_columns(merged)

        epoch = pd.to_numeric(merged["epoch_TDB"], errors="coerce")
        ra = pd.to_numeric(merged[ra_column], errors="coerce")
        dec = pd.to_numeric(merged[dec_column], errors="coerce")

        # A row without an epoch or without both coordinates is not an
        # observation.  This mirrors the filtering the old factory applied, so
        # the frame holds exactly the rows that can become observations.
        usable = np.isfinite(epoch) & np.isfinite(ra) & np.isfinite(dec)
        dropped = int((~usable).sum())
        if dropped:
            logger.info(
                "Voyager dataset %s: dropped %d row(s) with no epoch or no RA/Dec.",
                dataset_cfg.get("identifier", "?"),
                dropped,
            )

        target = self._target_name(dataset_cfg)
        receiver = self._receiver_name(dataset_cfg)
        parameters = self._parameters(dataset_cfg)
        identifier = str(dataset_cfg.get("identifier") or "")

        records = [
            {
                "observed_object": target,
                "observatory_code": receiver,
                "data_source": DataSource.VOYAGER.value,
                "source_set_id": identifier,
                "reference": dataset_cfg.get("reference") or identifier,
                "observation_type": ObservationType.ABSOLUTE.value,
                "coordinate_kind": "ra_dec",
                "frame_center": parameters["frame_center"].value,
                "original_equinox": parameters["equinox"],
                "original_reference_frame": parameters["reference_frame"].value,
                "original_reduction": parameters["reduction"].value,
                "receptor_type": parameters["receptor"].value,
                # The tabulated Julian date is already TDB (the config's
                # ``time_scale``), so the epoch passes through unmodified: there
                # is no clock to correct.
                "epoch_tdb": float(epoch_jd),
                "first_value": np.deg2rad(float(ra_value)),
                "second_value": np.deg2rad(float(dec_value)),
                "first_unit": "deg",
                "second_unit": "deg",
            }
            for epoch_jd, ra_value, dec_value in zip(
                epoch[usable].tolist(), ra[usable].tolist(), dec[usable].tolist()
            )
        ]

        return build_frame(records)

    @staticmethod
    def _target_name(dataset_cfg) -> str:
        """Return the observed body, which is the dataset's one satellite.

        Voyager observed Triton; the planet is the origin the spacecraft state is
        expressed relative to, not the object measured.  A file holding several
        satellites is rejected rather than silently reduced to the first, which
        is what the old factory did.
        """
        satellites = dataset_cfg.get("satellites") or {}
        if not satellites:
            raise ValueError(
                f"Voyager dataset {dataset_cfg.get('identifier', '?')} names no satellite, "
                f"so there is no observed body."
            )
        if len(satellites) > 1:
            raise NotImplementedError(
                f"Voyager dataset {dataset_cfg.get('identifier', '?')}: multiple satellites "
                f"in one file are not supported yet."
            )
        return str(next(iter(satellites)))

    @staticmethod
    def _receiver_name(dataset_cfg) -> str:
        """Return the receiver's name, which for Voyager is the spacecraft.

        The frame's ``observatory_code`` carries a *body* name here rather than an
        MPC code, which is the convention the schema documents for sources with no
        observatory code.  It also has to be the name rather than the numeric code
        (``-31``): the link end and the weights router both key on
        ``"Voyager 2"``, and a numeric code would send the receiver down the
        ground-station path, where there is no station to look up.
        """
        observatory = dataset_cfg.get("observatory")
        name = observatory.get("name") if observatory else None
        if name:
            return normalize_observatory_code(name)
        if observatory is not None and observatory.get("code") is not None:
            return normalize_observatory_code(observatory.get("code"))
        raise ValueError(
            f"Voyager dataset {dataset_cfg.get('identifier', '?')} names no observatory, "
            f"so the receiver cannot be identified."
        )

    @staticmethod
    def _parameters(dataset_cfg) -> dict:
        """Resolve the descriptor fields the schema requires.

        Voyager's config carries the same descriptive fields as an NSDB
        descriptor, so they are normalised through the same vocabulary mappers
        rather than cast directly - the labels are prose (``"no information"``,
        ``"Vidicon 800x800 CCD"``) and need the same interpretation.
        """
        return {
            "frame_center": normalize_frame_center(
                dataset_cfg.get("centre_of_frame") or dataset_cfg.get("center_of_frame")
            ),
            "reference_frame": normalize_reference_frame(dataset_cfg.get("reference_frame")),
            "reduction": normalize_catalogue(dataset_cfg.get("reduction")),
            "receptor": normalize_receptor(dataset_cfg.get("receptor")),
            "equinox": str(dataset_cfg.get("epoch_of_equinox") or "J2000"),
        }

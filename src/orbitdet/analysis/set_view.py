"""A per-set window onto the access layer, for weighting and outlier strategies.

Strategies need three things about one observation set: its residuals, its
epochs, and the per-observation metadata that lives in the standard frame
(scan angle, observatory code, data source, ...).  Historically they got the
first two from the Tudat ``SingleObservationSet`` and the third from module-level
registries keyed by epoch, because the pybind set cannot carry arbitrary
attributes.

:class:`SetView` replaces that with one object that reads everything from the
access layer's derived frame.  The frame already holds the residuals, the
weights and the standard columns, joined to the set through
``observation_set_index``; the only thing missing was the *ordering*, since the
frame is in file order while a set's observations are in the order the set was
built.  :func:`orbitdet.analysis.access._match_rows_by_epoch` resolves that, and
:class:`SetView` caches the result as :attr:`SetView.rows`.

With a ``SetView`` a strategy can read ``view.scan_angles`` instead of calling
``get_scan_angles_for_epochs``, which is what retires the epoch-keyed registry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tudatpy.estimation.observations as obs

from orbitdet.analysis.access import (
    OBSERVATION_SET_INDEX_COLUMN,
    ObservationAccess,
    _match_rows_by_epoch,
)

logger = logging.getLogger(__name__)

__all__ = ["SetView"]


@dataclass
class SetView:
    """One observation set, read through the access layer's derived frame.

    Parameters
    ----------
    access : ObservationAccess
        The access layer the set belongs to.
    set_index : int
        Position of the set in ``access.collection.get_single_observation_sets()``.
    rows : np.ndarray
        Frame row positions belonging to this set, **in the set's observation
        order** (not file order).  Every array this view returns is indexed by
        the same order, so ``view.residuals[i]`` and ``view.scan_angles[i]``
        describe the same observation.
    observation_set : SingleObservationSet
        The underlying Tudat set, for the few operations that genuinely need it
        (applying weights, reading the link definition).
    set_id : str
        Human-readable identifier (observatory code or body name).
    """

    access: ObservationAccess
    set_index: int
    rows: np.ndarray
    observation_set: obs.SingleObservationSet
    set_id: str

    # -- Construction ---------------------------------------------------

    @classmethod
    def build(cls, access: ObservationAccess, set_index: int) -> SetView:
        """Build the view for one set, resolving its frame rows by epoch.

        Parameters
        ----------
        access : ObservationAccess
            The access layer.
        set_index : int
            Position of the set in the collection.

        Returns
        -------
        SetView
            The view.  :attr:`rows` is empty when the set cannot be joined to
            the frame (a row that fed two sets names only the first).

        Raises
        ------
        IndexError
            If ``set_index`` is out of range.
        """
        sets = access.collection.get_single_observation_sets()
        if not 0 <= set_index < len(sets):
            raise IndexError(
                f"set_index {set_index} is out of range for a collection with {len(sets)} set(s)."
            )
        observation_set = sets[set_index]

        frame = access.frame
        set_epochs = [float(t) for t in observation_set.observation_times]

        rows = np.zeros(0, dtype=int)
        if OBSERVATION_SET_INDEX_COLUMN in frame.columns and set_epochs:
            frame_epochs = pd.to_numeric(frame["epoch_tdb"], errors="coerce").to_numpy("float64")
            set_index_values = pd.to_numeric(
                frame[OBSERVATION_SET_INDEX_COLUMN], errors="coerce"
            ).to_numpy("float64")
            positions = np.flatnonzero(set_index_values == float(set_index))
            if positions.size:
                matched = _match_rows_by_epoch(frame_epochs, positions, set_epochs)
                if matched is not None:
                    rows = matched
                else:
                    logger.warning(
                        "SetView [%d]: could not join every set observation to a frame row; "
                        "the view will report no rows.",
                        set_index,
                    )

        return cls(
            access=access,
            set_index=set_index,
            rows=rows,
            observation_set=observation_set,
            set_id=_set_identifier(observation_set),
        )

    # -- Frame access ---------------------------------------------------

    @property
    def frame(self) -> pd.DataFrame:
        """The standard-frame rows of this set, in the set's observation order."""
        return self.access.frame.iloc[self.rows]

    @property
    def n_obs(self) -> int:
        """Number of observations in the set."""
        return len(self.observation_set.observation_times)

    @property
    def is_joined(self) -> bool:
        """Whether every set observation resolved to a frame row."""
        return self.rows.size == self.n_obs

    # -- Per-observation arrays -----------------------------------------

    @property
    def epochs(self) -> np.ndarray:
        """Observation epochs (seconds since J2000 TDB), in set order."""
        return np.asarray(
            [float(t) for t in self.observation_set.observation_times], dtype="float64"
        )

    @property
    def residuals(self) -> np.ndarray:
        """``(n_obs, 2)`` residuals in radians, in set order.

        Read from the frame when the set is joined (so it matches the residuals
        the plots show); falls back to the set itself otherwise.
        """
        if self.is_joined:
            return np.column_stack(
                [
                    self.frame["residual_first_rad"].to_numpy("float64"),
                    self.frame["residual_second_rad"].to_numpy("float64"),
                ]
            )
        return _residuals_from_set(self.observation_set)

    @property
    def observations(self) -> np.ndarray:
        """``(n_obs, 2)`` observed values, in set order."""
        return np.asarray(self.observation_set.concatenated_observations, dtype="float64").reshape(
            -1, 2
        )

    @property
    def scan_angles(self) -> np.ndarray | None:
        """Per-observation position angle of scan (radians), or ``None``.

        Read from the standard frame's ``scan_angle`` column, which is populated
        on ingest from the Gaia archive.  ``None`` means the set is not a
        scanning instrument (or the frame could not be joined), so a strategy
        should fall back to a non-scan weighting.
        """
        if not self.is_joined or "scan_angle" not in self.frame.columns:
            return None
        angles = pd.to_numeric(self.frame["scan_angle"], errors="coerce").to_numpy("float64")
        if angles.size != self.n_obs or not np.all(np.isfinite(angles)):
            return None
        return angles

    @property
    def weights(self) -> np.ndarray:
        """``(n_obs, 2)`` applied diagonal weights, in set order."""
        if self.is_joined:
            return np.column_stack(
                [
                    self.frame["weight_first"].to_numpy("float64"),
                    self.frame["weight_second"].to_numpy("float64"),
                ]
            )
        return np.asarray(self.observation_set.weights_vector, dtype="float64").reshape(-1, 2)

    @property
    def observable_type(self) -> str:
        """Human-readable observable type (``"absolute"`` / ``"relative"``)."""
        from orbitdet.observations.weighting.dataframe import observable_type_str

        return observable_type_str(self.observation_set)

    # -- Column access --------------------------------------------------

    def column(self, name: str) -> np.ndarray:
        """Return one standard-frame column for this set, in set order.

        Parameters
        ----------
        name : str
            A standard-frame column name (e.g. ``"observatory_code"``).

        Returns
        -------
        np.ndarray
            The column's values for this set's rows.

        Raises
        ------
        KeyError
            If the column is not in the frame.
        """
        if name not in self.access.frame.columns:
            raise KeyError(
                f"'{name}' is not a standard-frame column. "
                f"Available: {sorted(self.access.frame.columns)}."
            )
        return self.frame[name].to_numpy()

    def __len__(self) -> int:
        return self.n_obs

    def __repr__(self) -> str:
        return (
            f"SetView(set_id={self.set_id!r}, set_index={self.set_index}, "
            f"n_obs={self.n_obs}, joined={self.is_joined})"
        )


def _set_identifier(observation_set: obs.SingleObservationSet) -> str:
    """Return the set's human-readable identifier (observatory code or body name)."""
    from orbitdet.observations.utils import get_set_identifier

    return get_set_identifier(observation_set)


def _residuals_from_set(observation_set: obs.SingleObservationSet) -> np.ndarray:
    """Read ``(n_obs, 2)`` residuals straight from the set (fallback path)."""
    residuals = np.asarray(observation_set.residuals, dtype="float64")
    if residuals.ndim == 1:
        if residuals.size == 0:
            return np.empty((0, 2))
        return residuals.reshape(1, 2)
    return residuals

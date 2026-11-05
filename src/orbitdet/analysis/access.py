"""Access layer joining the standard frame to the live observation sets.

The standard observation frame (:mod:`orbitdet.preprocessing.schema`) is the
canonical *ingest* contract: provenance, measurement semantics and the measured
values, one row per observation.  It deliberately carries no residuals and no
weights, because those are produced later, by the outlier-rejection and
weighting stages, and change on every estimation.  This module produces a
**derived** frame that adds exactly those columns, by joining the standard frame
to the :class:`~tudatpy.estimation.observations.SingleObservationSet` objects it
was built into.

The join key already exists
---------------------------
:data:`orbitdet.preprocessing.schema.OBSERVATION_SET_INDEX_COLUMN`
(``observation_set_index``) is filled in by
:func:`orbitdet.observations.collection.create_observation_collection`: it is
the position of the set in ``collection.get_single_observation_sets()`` that each
row was used to build.  It is a foreign key, not a copy of the data - the set
stays the authority on which observations it holds.  The derived frame reads
through it, so a row can be traced to its set and back.

Canonical columns, views on the side
------------------------------------
The derived frame keeps the **canonical** representation: radians, with the two
scalar components kept as ``first`` / ``second`` exactly as the standard frame
does.  Conversion to arcseconds, to named (RA, Dec), or into the scan frame is a
*view*: a pure function in :mod:`orbitdet.analysis.views` that returns a copy.
Nothing is written back to the frame, so there is one representation of the data
and the conversions are cheap, explicit and impossible to get out of sync.

Refresh, do not recompute
-------------------------
:meth:`ObservationAccess.refresh` re-reads residuals, weights and the sigmas
implied by the applied weights from the (mutated) collection.  It never
recomputes weights from residuals - that is the weighting subsystem's job - and
it never touches the provenance and measurement columns.  This mirrors the
post-estimation refresh the pipeline already performs on the per-observation
weights frame: residuals change, weights do not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tudatpy.estimation.observations as obs

from orbitdet.observations.utils import get_set_identifier
from orbitdet.observations.weighting.dataframe import observable_type_str
from orbitdet.preprocessing.schema import OBSERVATION_SET_INDEX_COLUMN

from .basis import BasisTransform

logger = logging.getLogger(__name__)

#: How a set may be addressed: its integer ``observation_set_index``, its
#: identifier string (e.g. ``"Gaia"``, ``"673"``), or the set object itself.
SetKey = int | str | obs.SingleObservationSet

__all__ = [
    "ObservationAccess",
    "JOINED_COLUMNS",
    "SET_INDEX_COLUMN",
    "SET_ID_COLUMN",
    "OBSERVABLE_TYPE_COLUMN",
    "REJECTED_COLUMN",
    "SetKey",
]

#: Number of scalar components a two-component observation carries.
_TWO = 2

#: Column holding the (float) set index the join was made on.
SET_INDEX_COLUMN = "set_index"

#: Human-readable identifier of the set a row was joined to (observatory code,
#: body name, ...), from :func:`orbitdet.observations.utils.get_set_identifier`.
SET_ID_COLUMN = "set_id"

#: Human-readable observable type (``"absolute"``, ``"relative"``, ...).
OBSERVABLE_TYPE_COLUMN = "observable_type"

#: Whether the row's observation was removed by outlier rejection.
#:
#: A row belongs to a set (it carries an ``observation_set_index``) but its epoch
#: is absent from that set's surviving observations; a set with no surviving
#: observations marks all of its rows.  This is derived from the accepted sets on
#: every :meth:`ObservationAccess.refresh`, so it can never disagree with the
#: residuals in :attr:`ObservationAccess.frame`.
REJECTED_COLUMN = "rejected"

#: Columns :meth:`ObservationAccess.build` adds to the standard frame.  They are
#: derived (residuals/weights/rejection) or denormalised (identifiers); the
#: standard frame is the authority for everything else.  Kept in one place so a
#: consumer can tell the derived columns from the ingested ones.
JOINED_COLUMNS: tuple[str, ...] = (
    SET_INDEX_COLUMN,
    SET_ID_COLUMN,
    OBSERVABLE_TYPE_COLUMN,
    "residual_first_rad",
    "residual_second_rad",
    "weight_first",
    "weight_second",
    "sigma_first_rad",
    "sigma_second_rad",
    "has_full_weight_matrix",
    REJECTED_COLUMN,
)


# ---------------------------------------------------------------------------
# Small, tolerant accessors for the Tudat set (property vs method drift)
# ---------------------------------------------------------------------------


def _as_bool(value) -> bool:
    """Coerce a Tudat flag that may be a property or a zero-arg method to bool."""
    if callable(value):
        value = value()
    return bool(value)


def _components(flat: np.ndarray | list, n_obs: int, single_size: int) -> np.ndarray:
    """Reshape a concatenated per-observation vector to an ``(n_obs, 2)`` array.

    A two-component observable (``single_size == 2``) is one row per observation
    with the two scalar components in columns.  A one-dimensional observable
    (``single_size == 1``, e.g. a separation with no position angle) has no
    second component: its column is filled with ``NaN`` rather than dropped, so
    the derived frame keeps a fixed shape.

    Parameters
    ----------
    flat : np.ndarray | list
        Concatenated vector as returned by Tudat (``residuals``,
        ``weights``, ...).
    n_obs : int
        Number of observations in the set.
    single_size : int
        Number of scalar components in one observation
        (``SingleObservationSet.single_observable_size``).

    Returns
    -------
    np.ndarray
        ``(n_obs, 2)`` array of floats.
    """
    values = np.asarray(flat, dtype="float64").ravel()
    if single_size == _TWO:
        return values.reshape(n_obs, _TWO)
    # One-dimensional (or unexpected) observable: put what exists in the first
    # column and leave the second undefined.
    out = np.full((n_obs, _TWO), np.nan)
    if single_size == 1 and values.size == n_obs:
        out[:, 0] = values
    return out


def _marginal_sigmas(
    observation_set: obs.SingleObservationSet,
    weights: np.ndarray,
) -> np.ndarray:
    """Per-component 1σ uncertainties implied by the *applied* weight matrix.

    With diagonal weights ``sigma_i = 1 / sqrt(w_i)``.  With a full (or
    block-diagonal) weight matrix the marginal sigma is
    ``sigma_i = sqrt(diag(W^-1))`` - **not** ``1 / sqrt(diag(W))``, which is
    wrong by orders of magnitude for a near-rank-deficient block such as a Gaia
    transit covariance.  A component with zero (or non-finite) weight has no
    finite sigma and is reported as ``NaN``.

    Parameters
    ----------
    observation_set : SingleObservationSet
        The set whose applied weights are read.
    weights : np.ndarray
        ``(n_obs, 2)`` array of the diagonal weights, as filled by
        :func:`_components`; used directly for the diagonal case.

    Returns
    -------
    np.ndarray
        ``(n_obs, 2)`` array of marginal sigmas in radians (``NaN`` where
        undefined).
    """
    n_obs = weights.shape[0]
    has_off_diagonal = _as_bool(getattr(observation_set, "has_off_diagonal_weights", False))
    has_full = _as_bool(getattr(observation_set, "has_full_weight_matrix_contribution", False))

    if not (has_off_diagonal or has_full):
        # Purely diagonal: sigma = 1/sqrt(w), with zero weights left undefined.
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = 1.0 / np.sqrt(weights)
        sigma[~np.isfinite(sigma) | (weights <= 0.0)] = np.nan
        return sigma

    # Full / block-diagonal: the marginal sigma needs diag(W^-1).  Invert the
    # whole applied matrix (it is at most a few hundred rows) and read the
    # diagonal, falling back to the pseudo-inverse for a rank-deficient block.
    weight_matrix = np.asarray(observation_set.weight_matrix, dtype="float64")
    expected = _TWO * n_obs
    if weight_matrix.shape != (expected, expected):
        logger.warning(
            "Applied weight matrix has shape %s, expected (%d, %d); marginal "
            "sigmas are unavailable for this set.",
            weight_matrix.shape,
            expected,
            expected,
        )
        return np.full((n_obs, _TWO), np.nan)

    try:
        covariance = np.linalg.inv(weight_matrix)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(weight_matrix)

    marginal = np.clip(np.diag(covariance), 0.0, None)
    return np.sqrt(marginal).reshape(n_obs, _TWO)


def _match_rows_by_epoch(
    frame_epochs: np.ndarray,
    positions: np.ndarray,
    set_epochs: list[float],
) -> np.ndarray | None:
    """Map each set observation to its row in the frame, by epoch.

    ``observation_set_index`` already says *which* rows built a set; this
    resolves *which is which* within that group.  Rows are consumed in file
    order and each set epoch takes the first unused row that matches it, so a
    frame holding the same epoch twice maps its two rows to the set's two
    observations in the order both were built in - the same rule the collection
    builder uses in ``_rows_of_set``.

    Parameters
    ----------
    frame_epochs : np.ndarray
        ``epoch_tdb`` column of the standard frame, one entry per row.
    positions : np.ndarray
        Row positions in the frame that carry this set's index.
    set_epochs : list[float]
        The set's ``observation_times`` as seconds since J2000.

    Returns
    -------
    np.ndarray | None
        Frame row positions in the set's observation order, or ``None`` when a
        set epoch has no matching row (the set cannot be joined fully).
    """
    used = np.zeros(frame_epochs.shape, dtype=bool)
    resolved: list[int] = []
    for epoch in set_epochs:
        close = np.isclose(frame_epochs, float(epoch), rtol=1e-12, atol=1e-6)
        matches = positions[close[positions] & ~used[positions]]
        if matches.size == 0:
            return None
        position = int(matches[0])
        used[position] = True
        resolved.append(position)
    return np.asarray(resolved, dtype=int)


@dataclass
class ObservationAccess:
    """Read-only access to observations through a joined, derived frame.

    The instance holds the standard frame and the collection it belongs to, plus
    a derived frame that adds the residual, weight and sigma columns described
    in :data:`JOINED_COLUMNS`.  Build it once and call :meth:`refresh` after the
    residuals change (e.g. after an estimation); the provenance columns are never
    touched.

    Parameters
    ----------
    collection : ObservationCollection
        The collection the frame was built into.
    frame : pd.DataFrame
        The derived frame: standard columns plus :data:`JOINED_COLUMNS`.
    """

    collection: obs.ObservationCollection
    frame: pd.DataFrame

    # -- Construction ---------------------------------------------------

    @classmethod
    def build(
        cls,
        collection: obs.ObservationCollection,
        standard_frame: pd.DataFrame,
    ) -> ObservationAccess:
        """Join a standard frame to the collection it was built into.

        Parameters
        ----------
        collection : ObservationCollection
            The collection returned alongside the standard frame by
            :func:`orbitdet.observations.collection.create_observation_collection`.
        standard_frame : pd.DataFrame
            The standard frame carrying
            :data:`orbitdet.preprocessing.schema.OBSERVATION_SET_INDEX_COLUMN`.

        Returns
        -------
        ObservationAccess
            An instance whose :attr:`frame` is the derived frame.
        """
        frame = standard_frame.reset_index(drop=True).copy()
        if OBSERVATION_SET_INDEX_COLUMN not in frame.columns:
            logger.warning(
                "Standard frame has no '%s' column; it was never assembled into a "
                "collection, so residuals and weights cannot be joined.",
                OBSERVATION_SET_INDEX_COLUMN,
            )
            return cls(collection, _with_empty_join_columns(frame))

        _ensure_join_columns(frame)
        frame[SET_INDEX_COLUMN] = frame[OBSERVATION_SET_INDEX_COLUMN]

        access = cls(collection, frame)
        return access.refresh()

    # -- Refresh --------------------------------------------------------

    def refresh(self) -> ObservationAccess:
        """Re-read residuals, weights and sigmas from the collection.

        The derived columns are rebuilt from the current state of every set; the
        standard columns are left untouched.  Weights are read, never
        recomputed.  Returns ``self`` for chaining.

        Returns
        -------
        ObservationAccess
            This instance, with :attr:`frame` refreshed in place.
        """
        frame = self.frame
        if OBSERVATION_SET_INDEX_COLUMN not in frame.columns:
            return self

        # Reset only the derived columns; the standard columns stay as ingested.
        for column in (
            "residual_first_rad",
            "residual_second_rad",
            "weight_first",
            "weight_second",
            "sigma_first_rad",
            "sigma_second_rad",
        ):
            frame[column] = np.nan
        frame[SET_ID_COLUMN] = ""
        frame[OBSERVABLE_TYPE_COLUMN] = ""
        frame["has_full_weight_matrix"] = False
        frame[REJECTED_COLUMN] = False

        frame_epochs = pd.to_numeric(frame["epoch_tdb"], errors="coerce").to_numpy("float64")
        set_index_values = pd.to_numeric(
            frame[OBSERVATION_SET_INDEX_COLUMN], errors="coerce"
        ).to_numpy("float64")

        for set_index, observation_set in enumerate(self.collection.get_single_observation_sets()):
            positions = np.flatnonzero(set_index_values == float(set_index))
            if positions.size == 0:
                continue

            n_obs = observation_set.number_of_observables
            set_positions = frame.index[positions]
            if n_obs == 0:
                # A set with no surviving observations means every row it was
                # built from was rejected.
                frame.loc[set_positions, REJECTED_COLUMN] = True
                continue

            set_epochs = [float(t) for t in observation_set.observation_times]
            rows = _match_rows_by_epoch(frame_epochs, positions, set_epochs)
            if rows is None or rows.size != n_obs:
                logger.warning(
                    "Observation set %d ('%s') could not be joined to the frame: "
                    "%d observation(s) have no matching row (a row that fed two "
                    "sets names only the first). Its residuals and weights are "
                    "left null.",
                    set_index,
                    get_set_identifier(observation_set),
                    n_obs,
                )
                continue

            # Rows of this set that the set no longer holds were rejected.
            accepted_labels = frame.index[rows]
            rejected_labels = set_positions.difference(accepted_labels)
            if rejected_labels.size:
                frame.loc[rejected_labels, REJECTED_COLUMN] = True

            single_size = int(observation_set.single_observable_size)
            residuals = _components(observation_set.concatenated_residuals, n_obs, single_size)
            weights = _components(observation_set.weights_vector, n_obs, single_size)
            sigmas = _marginal_sigmas(observation_set, weights)

            frame.loc[frame.index[rows], SET_ID_COLUMN] = get_set_identifier(observation_set)
            frame.loc[frame.index[rows], OBSERVABLE_TYPE_COLUMN] = observable_type_str(
                observation_set
            )
            frame.loc[frame.index[rows], "residual_first_rad"] = residuals[:, 0]
            frame.loc[frame.index[rows], "residual_second_rad"] = residuals[:, 1]
            frame.loc[frame.index[rows], "weight_first"] = weights[:, 0]
            frame.loc[frame.index[rows], "weight_second"] = weights[:, 1]
            frame.loc[frame.index[rows], "sigma_first_rad"] = sigmas[:, 0]
            frame.loc[frame.index[rows], "sigma_second_rad"] = sigmas[:, 1]
            frame.loc[frame.index[rows], "has_full_weight_matrix"] = _as_bool(
                getattr(observation_set, "has_full_weight_matrix_contribution", False)
            ) or _as_bool(getattr(observation_set, "has_off_diagonal_weights", False))

        return self

    # -- Ergonomic getters ---------------------------------------------

    @property
    def set_ids(self) -> list[str]:
        """The set identifiers present in the collection, in collection order."""
        return [get_set_identifier(s) for s in self.collection.get_single_observation_sets()]

    def __len__(self) -> int:
        return len(self.frame)

    def record(self, row: int) -> pd.Series:
        """Return one observation as a single row, by frame position.

        Parameters
        ----------
        row : int
            Positional index into :attr:`frame` (as accepted by ``frame.iloc``).

        Returns
        -------
        pd.Series
            The full derived row: standard columns plus the joined columns.
        """
        return self.frame.iloc[row]

    def residual(self, row: int) -> tuple[float, float]:
        """Return one observation's residual as ``(first, second)`` radians.

        Parameters
        ----------
        row : int
            Positional index into :attr:`frame`.

        Returns
        -------
        tuple[float, float]
            ``(first, second)`` residual in radians (``NaN`` where undefined).
        """
        record = self.frame.iloc[row]
        return (float(record["residual_first_rad"]), float(record["residual_second_rad"]))

    def set_frame(self, set_index: int) -> pd.DataFrame:
        """Return the derived rows that belong to one set.

        Parameters
        ----------
        set_index : int
            Position of the set in ``collection.get_single_observation_sets()``.

        Returns
        -------
        pd.DataFrame
            The rows of :attr:`frame` joined to that set (possibly empty).
        """
        mask = self.frame.get(OBSERVATION_SET_INDEX_COLUMN)
        if mask is None:
            return self.frame.iloc[0:0]
        return self.frame.loc[pd.to_numeric(mask, errors="coerce") == float(set_index)]

    def by_set_id(self, set_id: str) -> pd.DataFrame:
        """Return the derived rows joined to the sets with a given identifier.

        Parameters
        ----------
        set_id : str
            A value of :data:`SET_ID_COLUMN` (e.g. an observatory code).

        Returns
        -------
        pd.DataFrame
            The matching rows (possibly empty).
        """
        return self.frame.loc[self.frame[SET_ID_COLUMN] == set_id]

    # -- Set-level accessors -------------------------------------------
    #
    # A weight matrix is a property of a whole observation set (it can be
    # ``2n x 2n`` with off-diagonal cross-observation terms), so it does not fit
    # the per-row derived frame.  These accessors read it from the set directly.
    #
    # Every set-level accessor accepts a ``SetKey``: the integer
    # ``observation_set_index``, the set identifier string (e.g. ``"Gaia"``,
    # ``"673"``), or the :class:`SingleObservationSet` itself.  A set identifier
    # is not guaranteed unique across sets, so a string key that is ambiguous
    # raises rather than guessing.

    def set_indices(self, set_id: str) -> list[int]:
        """Return every set index whose set identifier equals ``set_id``.

        Parameters
        ----------
        set_id : str
            A value of :data:`SET_ID_COLUMN` (e.g. ``"Gaia"``, ``"673"``).

        Returns
        -------
        list[int]
            The matching indices in ``collection.get_single_observation_sets()``
            order (possibly empty).
        """
        return [
            index
            for index, candidate in enumerate(self.collection.get_single_observation_sets())
            if get_set_identifier(candidate) == set_id
        ]

    def set_index_for(self, set_id: str) -> int:
        """Return the single set index for a set identifier.

        Parameters
        ----------
        set_id : str
            A value of :data:`SET_ID_COLUMN`.

        Returns
        -------
        int
            The unique index of the set with that identifier.

        Raises
        ------
        KeyError
            If no set has that identifier.
        ValueError
            If more than one set shares it (use :meth:`set_indices`).
        """
        indices = self.set_indices(set_id)
        if not indices:
            raise KeyError(f"No observation set with identifier '{set_id}'.")
        if len(indices) > 1:
            raise ValueError(
                f"Set identifier '{set_id}' is ambiguous ({len(indices)} sets: {indices}). "
                "Use set_indices() or address the set by index."
            )
        return indices[0]

    def _resolve_set_index(self, key: SetKey) -> int:
        """Resolve a :data:`SetKey` (index, identifier or set) to a set index."""
        if isinstance(key, (int, np.integer)):
            return int(key)
        if isinstance(key, str):
            return self.set_index_for(key)
        return self.set_index_of(key)

    def observation_set(self, key: SetKey) -> obs.SingleObservationSet:
        """Return the observation set named by ``key``.

        Parameters
        ----------
        key : SetKey
            The integer set index, the set identifier, or the set object itself.

        Returns
        -------
        SingleObservationSet
            The underlying Tudat set.
        """
        return self.collection.get_single_observation_sets()[self._resolve_set_index(key)]

    def set_index_of(self, observation_set: obs.SingleObservationSet) -> int:
        """Return the frame set index of an observation set, by identity.

        Parameters
        ----------
        observation_set : SingleObservationSet
            A set from the same collection.

        Returns
        -------
        int
            The index ``observation_set`` occupies in
            ``collection.get_single_observation_sets()``.

        Raises
        ------
        ValueError
            If the set is not part of this collection.
        """
        for index, candidate in enumerate(self.collection.get_single_observation_sets()):
            if candidate is observation_set:
                return index
        raise ValueError("Set is not part of this access layer's collection.")

    def basis_transform(
        self,
        key: SetKey,
        kind: str = "scan",
        *,
        de_projected: bool = False,
    ) -> BasisTransform:
        """Build the basis transform of one set.

        Parameters
        ----------
        key : SetKey
            The integer set index, the set identifier, or the set object itself.
        kind : str
            The target basis.  ``"scan"`` builds the along/across-scan rotation
            from the set's ``scan_angle`` frame column; ``"canonical"`` returns
            the identity.  Other kinds raise :class:`ValueError`.
        de_projected : bool
            Whether the first component already carries the ``cos(δ)`` factor
            (i.e. it holds ``α·cosδ`` rather than ``α``).  Defaults to ``False``
            because the pipeline's observations and residuals are in **raw**
            ``(α, δ)``: the Gaia archive stores true right ascension ``α`` (its
            ``ra_error`` is the de-projected ``σ_{α cosδ}``, which the bindings
            divide back out), and the observable is built from that ``α``.  With
            ``False`` the ``cos(δ)`` factor is applied from the observed
            declinations, so the rotation is non-orthogonal and
            :meth:`BasisTransform.weight` uses the correct dual.  Pass ``True``
            only when the components were de-projected upstream.

        Returns
        -------
        BasisTransform
            The block-diagonal transform over the set's observations.

        Raises
        ------
        ValueError
            If ``kind`` is unknown, or if ``"scan"`` is requested for a set whose
            rows carry no ``scan_angle`` (there is no basis to build).
        """
        observation_set = self.observation_set(key)
        n_obs = observation_set.number_of_observables

        if kind == "canonical":
            return BasisTransform.identity(2 * n_obs if n_obs else 2, name="canonical")
        if kind != "scan":
            raise ValueError(f"Unknown basis kind '{kind}'; expected 'canonical' or 'scan'.")

        psi = self._scan_angles_in_set_order(key)
        if psi is None or len(psi) != n_obs:
            raise ValueError(
                f"Set '{get_set_identifier(observation_set)}' has no scan_angle in its "
                "frame rows, so no scan basis can be built."
            )
        dec = None
        if not de_projected:
            observed = np.asarray(
                observation_set.concatenated_observations, dtype="float64"
            ).reshape(-1, 2)
            dec = observed[:, 1]
        return BasisTransform.scan(psi, dec=dec)

    def _scan_angles_in_set_order(self, key: SetKey) -> np.ndarray | None:
        """Return the set's ``scan_angle`` values in observation order, or ``None``.

        The scan angle lives on the standard frame (``scan_angle``), joined here
        through ``observation_set_index``.  Rows are matched to the set's
        observations by epoch - the same rule :meth:`refresh` uses - so the angles
        line up with ``concatenated_residuals``.  ``None`` means the set has no
        rows, or they carry no finite angle.
        """
        observation_set = self.observation_set(key)
        n_obs = observation_set.number_of_observables
        if n_obs == 0:
            return None
        rows = self.set_frame(self._resolve_set_index(key))
        if rows.empty or "scan_angle" not in rows.columns:
            return None

        frame_epochs = pd.to_numeric(rows["epoch_tdb"], errors="coerce").to_numpy("float64")
        positions = np.arange(len(rows))
        set_epochs = [float(t) for t in observation_set.observation_times]
        matched = _match_rows_by_epoch(frame_epochs, positions, set_epochs)
        if matched is None or matched.size != n_obs:
            return None

        angles = rows["scan_angle"].to_numpy("float64")[matched]
        if not np.all(np.isfinite(angles)):
            return None
        return angles

    def weight_matrix(
        self, key: SetKey, *, basis: str | BasisTransform = "canonical"
    ) -> np.ndarray:
        """Return the applied weight matrix of one set as a dense array.

        For a set with any off-diagonal (full or block-diagonal) contribution the
        matrix is the full ``2n x 2n`` array, **including** the diagonal base
        weights Tudat adds on top (``weight_matrix = diag(tabulated) +
        full_weight_matrix``).  For a purely diagonal set the matrix is the
        ``2n x 2n`` diagonal, which is generally large; prefer
        :meth:`weight_diagonal` or the per-row ``weight_first`` / ``weight_second``
        columns of :attr:`frame` for such sets.

        Parameters
        ----------
        key : SetKey
            The integer set index, the set identifier (e.g. ``"Gaia"``), or the
            set object itself.
        basis : str | BasisTransform
            Basis in which to express the weight matrix.  ``"canonical"`` (the
            default) returns it as stored; ``"scan"`` applies the correct dual
            transform ``T^{-T} W T^{-1}``; a :class:`BasisTransform` may be
            passed to reuse one already built.

        Returns
        -------
        np.ndarray
            The dense ``(2n, 2n)`` weight matrix in the requested basis.
        """
        weight_matrix = np.asarray(self.observation_set(key).weight_matrix, dtype="float64")
        basis_transform = self._resolve_basis(key, basis)
        if basis_transform is None:
            return weight_matrix
        return basis_transform.weight(weight_matrix)

    def _resolve_basis(self, key: SetKey, basis: str | BasisTransform) -> BasisTransform | None:
        """Resolve a ``basis`` argument to a transform, or ``None`` for canonical."""
        if isinstance(basis, BasisTransform):
            return basis
        if basis == "canonical":
            return None
        return self.basis_transform(key, kind=basis)

    def weight_diagonal(
        self, key: SetKey, *, basis: str | BasisTransform = "canonical"
    ) -> np.ndarray:
        """Return the diagonal of the applied weight matrix of one set.

        Parameters
        ----------
        key : SetKey
            The integer set index, the set identifier, or the set object itself.
        basis : str | BasisTransform
            Basis in which to express the weights (see :meth:`weight_matrix`).

        Returns
        -------
        np.ndarray
            ``(2n,)`` array of the diagonal weights.
        """
        return np.diag(self.weight_matrix(key, basis=basis))

    def covariance_matrix(
        self, key: SetKey, *, basis: str | BasisTransform = "canonical"
    ) -> np.ndarray:
        """Return the covariance (inverse weight matrix) of one set.

        The inverse is taken of the full ``2n x 2n`` weight matrix, so it keeps
        the cross-observation correlations a full weight matrix carries.  A
        rank-deficient matrix (e.g. a one-dimensional relative measurement) is
        inverted with the pseudo-inverse.

        Parameters
        ----------
        key : SetKey
            The integer set index, the set identifier, or the set object itself.
        basis : str | BasisTransform
            Basis in which to express the covariance.  ``"canonical"`` (default)
            returns it as stored; ``"scan"`` applies ``T C T^T``.

        Returns
        -------
        np.ndarray
            The dense ``(2n, 2n)`` covariance matrix in the requested basis.
        """
        weight_matrix = np.asarray(self.observation_set(key).weight_matrix, dtype="float64")
        try:
            covariance = np.linalg.inv(weight_matrix)
        except np.linalg.LinAlgError:
            covariance = np.linalg.pinv(weight_matrix)
        basis_transform = self._resolve_basis(key, basis)
        if basis_transform is None:
            return covariance
        return basis_transform.covariance(covariance)

    def has_full_weight_matrix(self, key: SetKey) -> bool:
        """Whether one set carries an off-diagonal (full or block-diagonal) matrix.

        Parameters
        ----------
        key : SetKey
            The integer set index, the set identifier, or the set object itself.

        Returns
        -------
        bool
            ``True`` when the set's weight matrix has off-diagonal terms.
        """
        observation_set = self.observation_set(key)
        return _as_bool(
            getattr(observation_set, "has_full_weight_matrix_contribution", False)
        ) or _as_bool(getattr(observation_set, "has_off_diagonal_weights", False))

    def residuals(self, key: SetKey) -> np.ndarray:
        """Return the set's concatenated residual vector (raw, canonical frame).

        This is the fetched input to the pure statistics in
        :mod:`orbitdet.analysis.statistics`: the residuals, paired with
        :meth:`weight_matrix` (or :meth:`covariance_matrix`), compose into
        :func:`~orbitdet.analysis.statistics.chi2_from_matrix`,
        :func:`~orbitdet.analysis.statistics.mahalanobis` and
        :class:`~orbitdet.analysis.whitening.Whitening`.  The calculation lives
        in the statistics layer; this accessor only reads the data.

        Rejected observations leave ``NaN`` entries, exactly as the set holds
        them; the statistics functions drop them.

        Parameters
        ----------
        key : SetKey
            The integer set index, the set identifier, or the set object itself.

        Returns
        -------
        np.ndarray
            The ``(2n,)`` interleaved residual vector ``[first_1, second_1, ...]``
            in the canonical frame (empty for a set with no observations).
        """
        observation_set = self.observation_set(key)
        return np.asarray(observation_set.concatenated_residuals, dtype="float64").ravel()

    def weight_matrices(self) -> dict[str, np.ndarray]:
        """Return the applied weight matrix of every set, keyed by set identifier.

        A set identifier (an observatory code, a body name, ...) is not
        guaranteed unique across sets, so duplicate identifiers are disambiguated
        with a ``#<index>`` suffix.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from a unique key to each set's dense weight matrix.
        """
        matrices: dict[str, np.ndarray] = {}
        for set_index in range(len(self.collection.get_single_observation_sets())):
            key = get_set_identifier(self.observation_set(set_index))
            if key in matrices:
                key = f"{key}#{set_index}"
            matrices[key] = self.weight_matrix(set_index)
        return matrices

    def accepted(self) -> pd.DataFrame:
        """Return the derived frame without outlier-rejected rows.

        Returns
        -------
        pd.DataFrame
            Rows where :data:`REJECTED_COLUMN` is false (the fit's actual input
            set).  Empty when the frame was never joined to a collection.
        """
        if REJECTED_COLUMN not in self.frame.columns:
            return self.frame.iloc[0:0]
        return self.frame.loc[~self.frame[REJECTED_COLUMN]]

    def rejected(self) -> pd.DataFrame:
        """Return only the outlier-rejected rows of the derived frame.

        Returns
        -------
        pd.DataFrame
            Rows where :data:`REJECTED_COLUMN` is true (possibly empty).
        """
        if REJECTED_COLUMN not in self.frame.columns:
            return self.frame.iloc[0:0]
        return self.frame.loc[self.frame[REJECTED_COLUMN]]


def _with_empty_join_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns to a frame that has no set index, all null."""
    out = frame.copy()
    _ensure_join_columns(out)
    out["has_full_weight_matrix"] = False
    return out


def _ensure_join_columns(frame: pd.DataFrame) -> None:
    """Add any missing derived column with a dtype that accepts the joined values.

    The identifier columns are object-typed (they hold strings) and the numeric
    columns are float64.  Getting the dtype right up front avoids pandas'
    ``Invalid value ... for dtype 'float64'`` when a later ``.loc`` assignment
    writes a string into a column that was initialised as float.
    """
    object_columns = {
        SET_INDEX_COLUMN: "float64",
        SET_ID_COLUMN: "object",
        OBSERVABLE_TYPE_COLUMN: "object",
    }
    numeric_columns = (
        "residual_first_rad",
        "residual_second_rad",
        "weight_first",
        "weight_second",
        "sigma_first_rad",
        "sigma_second_rad",
    )
    for column, dtype in object_columns.items():
        if column not in frame.columns:
            fill = np.nan if dtype == "float64" else None
            frame[column] = pd.Series(fill, index=frame.index, dtype=dtype)
    for column in numeric_columns:
        if column not in frame.columns:
            frame[column] = np.nan
    if "has_full_weight_matrix" not in frame.columns:
        frame["has_full_weight_matrix"] = False
    if REJECTED_COLUMN not in frame.columns:
        frame[REJECTED_COLUMN] = False

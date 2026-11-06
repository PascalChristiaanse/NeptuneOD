"""Pure statistics over a derived observation frame.

Every function here takes a dataframe (or plain arrays) and returns a value or a
new tidy dataframe.  Nothing reads config or touches the file system - that is
the job of the table layer (:mod:`orbitdet.analysis.tables`).
Keeping the maths pure makes it unit-testable without a collection, a system of
bodies, or an active Hydra run, and makes the assumptions explicit.

Conventions
-----------
* Residuals and sigmas are in **radians** in the derived frame.  Aggregations
  report radians; convert with a view (:mod:`orbitdet.analysis.views`) or the
  ``to_unit`` parameter of :func:`aggregate`.
* ``first`` / ``second`` are the canonical scalar components and are *not*
  assumed to be RA/Dec - the frame records what they are in ``coordinate_kind``.
  Functions that need RA/Dec semantics say so; the aggregate columns are named
  ``first`` / ``second`` for that reason.
* Degrees of freedom follow ``nu = N - p`` with ``N`` the number of observations
  (one per row; the two components of a pair are one observation, matching the
  weighting subsystem's convention) and ``p`` the number of fitted parameters.

Calibration caveat
------------------
For residual-derived weight schemes the weights are self-normalising (a
single-observation set's sigma *is* its residual RMS), so a normalized-residual
mean of unity is unfalsifiable there.  ``chi2`` and ``reduced_chi2`` are
therefore reported as descriptive statistics, not as a goodness-of-fit test
unless the weights come from an external sigma.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "rms",
    "weighted_rms",
    "mean",
    "sample_std",
    "median",
    "minimum",
    "maximum",
    "quadrature",
    "pooled_within_group_rms",
    "normalized_residuals",
    "chi2",
    "chi2_from_matrix",
    "mahalanobis",
    "reduced_chi2",
    "aggregate",
]

#: Canonical residual columns (radians) in component order.
_RESIDUAL_COLUMNS: tuple[str, str] = ("residual_first_rad", "residual_second_rad")

#: Canonical sigma columns (radians) in component order.
_SIGMA_COLUMNS: tuple[str, str] = ("sigma_first_rad", "sigma_second_rad")

#: Unit-name -> radians-to-unit factor, for :func:`aggregate`'s ``to_unit``.
_UNIT_FACTOR: dict[str, float] = {
    "rad": 1.0,
    "arcsec": float(np.rad2deg(1.0) * 3600.0),
    "mas": float(np.rad2deg(1.0) * 3600.0) * 1e3,
}


def _finite(values) -> np.ndarray:
    """Return the finite entries of an array-like as float64."""
    array = np.asarray(values, dtype="float64").ravel()
    return array[np.isfinite(array)]


def rms(values) -> float:
    """Root-mean-square of the finite entries; ``NaN`` if there are none.

    Parameters
    ----------
    values : array-like
        Residuals, in whatever unit the caller is reporting.

    Returns
    -------
    float
        ``sqrt(mean(values**2))`` over the finite entries.
    """
    finite = _finite(values)
    if finite.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(finite**2)))


def weighted_rms(residuals, sigmas) -> float:
    """Weighted RMS ``sqrt(Σ w r² / Σ w)`` with ``w = 1/σ²``, over finite pairs.

    Parameters
    ----------
    residuals, sigmas : array-like
        Residuals and 1σ uncertainties in the same unit.

    Returns
    -------
    float
        The weighted RMS, or ``NaN`` when no pair is finite with ``σ > 0``.
    """
    residuals = np.asarray(residuals, dtype="float64").ravel()
    sigmas = np.asarray(sigmas, dtype="float64").ravel()
    good = np.isfinite(residuals) & np.isfinite(sigmas) & (sigmas > 0.0)
    if not np.any(good):
        return np.nan
    r = residuals[good]
    w = 1.0 / sigmas[good] ** 2
    return float(np.sqrt(np.sum(w * r**2) / np.sum(w)))


def mean(values) -> float:
    """Arithmetic mean of the finite entries; ``NaN`` if there are none.

    Parameters
    ----------
    values : array-like
        Values in whatever unit the caller is reporting.

    Returns
    -------
    float
        ``mean(values)`` over the finite entries.
    """
    finite = _finite(values)
    if finite.size == 0:
        return np.nan
    return float(np.mean(finite))


def sample_std(values, *, ddof: int = 1) -> float:
    """Standard deviation of the finite entries; ``NaN`` if there are too few.

    Parameters
    ----------
    values : array-like
        Values in whatever unit the caller is reporting.
    ddof : int
        Delta degrees of freedom.  Defaults to ``1`` (the sample standard
        deviation, matching the plots); pass ``0`` for the population value.

    Returns
    -------
    float
        ``std(values, ddof=ddof)`` over the finite entries, or ``NaN`` when
        fewer than ``ddof + 1`` finite entries remain.
    """
    finite = _finite(values)
    if finite.size <= ddof:
        return np.nan
    return float(np.std(finite, ddof=ddof))


def median(values) -> float:
    """Median of the finite entries; ``NaN`` if there are none.

    Parameters
    ----------
    values : array-like
        Values in whatever unit the caller is reporting.

    Returns
    -------
    float
        ``median(values)`` over the finite entries.
    """
    finite = _finite(values)
    if finite.size == 0:
        return np.nan
    return float(np.median(finite))


def minimum(values) -> float:
    """Smallest finite entry; ``NaN`` if there are none.

    Parameters
    ----------
    values : array-like
        Values in whatever unit the caller is reporting.

    Returns
    -------
    float
        ``min(values)`` over the finite entries.
    """
    finite = _finite(values)
    if finite.size == 0:
        return np.nan
    return float(np.min(finite))


def maximum(values) -> float:
    """Largest finite entry; ``NaN`` if there are none.

    Parameters
    ----------
    values : array-like
        Values in whatever unit the caller is reporting.

    Returns
    -------
    float
        ``max(values)`` over the finite entries.
    """
    finite = _finite(values)
    if finite.size == 0:
        return np.nan
    return float(np.max(finite))


def quadrature(values) -> float:
    """Quadrature sum ``sqrt(Σ xᵢ²)`` of the finite entries.

    This is how independent 1σ uncertainties combine (e.g. random and systematic
    errors, or the per-axis sigmas of a vector norm).  It equals
    ``rms(values) * sqrt(n)`` for ``n`` finite entries.

    Parameters
    ----------
    values : array-like
        The components to combine (typically 1σ uncertainties).

    Returns
    -------
    float
        ``sqrt(Σ xᵢ²)`` over the finite entries, or ``NaN`` when there are none.
    """
    finite = _finite(values)
    if finite.size == 0:
        return np.nan
    return float(np.sqrt(np.sum(finite**2)))


def pooled_within_group_rms(values, labels) -> float:
    """Pooled within-cluster RMS about each cluster's own mean.

    This is the "random" component of a scatter that has been split into clusters
    (e.g. transits or nights): the RMS of the residuals about their cluster means,
    with the cluster means removed from the degrees of freedom::

        sqrt( Σ_g Σ_i (x_gi − x̄_g)² / (N − M) )

    for ``M`` clusters and ``N`` finite observations.  It is the natural estimator
    of the within-cluster noise when the cluster means themselves vary (the
    between-cluster spread is the *systematic* component, see
    :func:`sample_std` of the cluster means).

    Parameters
    ----------
    values : array-like
        The observations.
    labels : array-like
        A cluster label per observation (any hashable, e.g. an integer transit
        id).  Must have the same length as ``values``.

    Returns
    -------
    float
        The pooled within-cluster RMS, or ``NaN`` when there are no finite
        observations or no residual degrees of freedom (``N <= M``).

    Raises
    ------
    ValueError
        If ``values`` and ``labels`` have different lengths.
    """
    values = np.asarray(values, dtype="float64").ravel()
    labels = np.asarray(labels).ravel()
    if values.size != labels.size:
        raise ValueError(
            f"values and labels must have matching lengths, got {values.size} and {labels.size}."
        )
    finite = np.isfinite(values)
    values, labels = values[finite], labels[finite]
    if values.size == 0:
        return np.nan

    # Cluster means, broadcast back to each observation.
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    n_groups = unique_labels.size
    sums = np.bincount(inverse, weights=values, minlength=n_groups)
    counts = np.bincount(inverse, minlength=n_groups)
    group_means = sums / counts
    within = values - group_means[inverse]

    dof = values.size - n_groups
    if dof <= 0:
        return np.nan
    return float(np.sqrt(np.sum(within**2) / dof))


def normalized_residuals(df: pd.DataFrame, prefix: str = "z") -> pd.DataFrame:
    """Return a copy with per-component normalized residual columns added.

    The normalized residual is ``r / σ`` for each component, using the sigma
    implied by the *applied* weight matrix (the ``sigma_*_rad`` columns the
    access layer fills).  For a full (off-diagonal) weight matrix this is the
    componentwise normalization and ignores cross-observation correlation; use
    the Cholesky-whitened residuals in :mod:`orbitdet.analysis.whitening` when
    correlation matters.

    Parameters
    ----------
    df : pd.DataFrame
        A derived observation frame.
    prefix : str
        Base name for the added columns.  Defaults to ``"z"``.

    Returns
    -------
    pd.DataFrame
        A copy with ``<prefix>_first`` and ``<prefix>_second`` added.
    """
    out = df.copy()
    for suffix, residual_col, sigma_col in zip(
        ("first", "second"), _RESIDUAL_COLUMNS, _SIGMA_COLUMNS, strict=True
    ):
        sigma = out[sigma_col].to_numpy("float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"{prefix}_{suffix}"] = np.where(
                np.isfinite(sigma) & (sigma > 0.0),
                out[residual_col].to_numpy("float64") / sigma,
                np.nan,
            )
    return out


def chi2(df: pd.DataFrame, weight_matrix: np.ndarray | None = None) -> float:
    """Return the chi-squared of a derived frame.

    With ``weight_matrix=None`` (the default) this is the descriptive
    ``Σ (r/σ)²`` over both components, using the per-component sigmas in the
    ``sigma_*_rad`` columns.  That ignores any cross-observation correlation a
    full weight matrix carries.  Pass the set's ``(2n, 2n)`` weight matrix to
    get the general ``r^T W r`` instead, which reduces to ``Σ (r/σ)²`` when the
    matrix is diagonal and is the exact statistic otherwise.  ``r`` is the
    interleaved ``[first_1, second_1, first_2, ...]`` residual vector, matching
    the ordering the access layer joins.

    Parameters
    ----------
    df : pd.DataFrame
        A derived observation frame (or one set's rows).
    weight_matrix : np.ndarray | None
        Optional ``(2n, 2n)`` weight matrix covering ``df``'s rows in frame
        order.  Requires ``df`` to be a whole set's rows with no gaps.

    Returns
    -------
    float
        The chi-squared sum (``NaN`` when there are no usable components).

    Raises
    ------
    ValueError
        If ``weight_matrix`` is supplied and its size does not match the
        residual vector built for ``df``.
    """
    if weight_matrix is None:
        z = normalized_residuals(df)
        values = np.concatenate(
            [z["z_first"].to_numpy("float64"), z["z_second"].to_numpy("float64")]
        )
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.nan
        return float(np.sum(finite**2))

    first = df["residual_first_rad"].to_numpy("float64")
    second = df["residual_second_rad"].to_numpy("float64")
    residuals = np.stack([first, second], axis=1).ravel()
    return chi2_from_matrix(residuals, weight_matrix)


def _drop_nonfinite(
    residuals: np.ndarray, weight_matrix: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Drop rows whose residual is non-finite, keeping ``W`` conformable.

    Rejected observations leave a ``NaN`` residual; the matching rows and columns
    of the weight matrix are removed so the quadratic form stays well defined.

    Parameters
    ----------
    residuals : np.ndarray
        The ``(n,)`` residual vector.
    weight_matrix : np.ndarray
        The ``(n, n)`` weight matrix.

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        ``(residuals, weight_matrix)`` on the surviving entries, or
        ``(None, None)`` when none survive.

    Raises
    ------
    ValueError
        If the residual vector and the weight matrix disagree in size once a
        non-finite entry is present.
    """
    keep = np.isfinite(residuals)
    if not np.any(keep):
        return None, None
    if not np.all(keep):
        if keep.size != weight_matrix.shape[0]:
            raise ValueError(
                f"Residual vector size {keep.size} and weight matrix "
                f"{weight_matrix.shape} disagree."
            )
        return residuals[keep], weight_matrix[np.ix_(keep, keep)]
    if weight_matrix.shape != (residuals.size, residuals.size):
        raise ValueError(
            f"Weight matrix shape {weight_matrix.shape} does not match residual "
            f"vector size {residuals.size}."
        )
    return residuals, weight_matrix


def chi2_from_matrix(residuals, weight_matrix) -> float:
    """Return the general quadratic form ``r^T W r`` for one observation set.

    This is the exact chi-squared for correlated observations: with a diagonal
    weight matrix it equals ``Σ (r/σ)²``, and with a full (or block-diagonal)
    matrix it includes the cross-observation correlations that a componentwise
    sum ignores.  The form is invariant under a change of basis, so the weight
    matrix and residuals must be expressed in the *same* basis (e.g. both
    canonical, or both scan).

    Rows whose residual is ``NaN`` (rejected observations) are dropped, keeping
    the matching rows and columns of ``weight_matrix`` so the form is conformable.

    Parameters
    ----------
    residuals : array-like
        The ``(n,)`` interleaved residual vector ``[first_1, second_1, ...]``.
    weight_matrix : array-like
        The ``(n, n)`` weight matrix ``W`` covering ``residuals``.

    Returns
    -------
    float
        ``r^T W r``, or ``NaN`` when no usable residual remains.

    Raises
    ------
    ValueError
        If the residual vector and the weight matrix have mismatched sizes.
    """
    residuals = np.asarray(residuals, dtype="float64").ravel()
    weight_matrix = np.asarray(weight_matrix, dtype="float64")
    residuals, weight_matrix = _drop_nonfinite(residuals, weight_matrix)
    if residuals is None:
        return np.nan
    return float(residuals @ weight_matrix @ residuals)


def mahalanobis(residuals, *, weight_matrix=None, sigmas=None) -> float:
    """Return the Mahalanobis distance (the whitened norm) of a residual vector.

    The Mahalanobis distance generalizes the normalized residual to correlated
    observations: it is ``||z||`` with ``z = L^T r`` the Cholesky-whitened
    residual (``W = L L^T``), and its square is the chi-squared ``r^T W r``.  For
    independent components it reduces to ``sqrt(Σ (r/σ)²)``.

    Exactly one of ``weight_matrix`` and ``sigmas`` must be given:

    * ``weight_matrix`` - the full ``(n, n)`` matrix ``W``; the exact distance,
      including cross-observation correlation.
    * ``sigmas`` - the per-component 1σ vector; the componentwise distance that
      ignores correlation (equal to the matrix form when ``W`` is diagonal).

    Non-finite residuals are dropped.

    Parameters
    ----------
    residuals : array-like
        The ``(n,)`` residual vector.
    weight_matrix : array-like | None
        The ``(n, n)`` weight matrix ``W``.
    sigmas : array-like | None
        The ``(n,)`` per-component 1σ uncertainties.

    Returns
    -------
    float
        The Mahalanobis distance, or ``NaN`` when no usable component remains.

    Raises
    ------
    ValueError
        If neither or both of ``weight_matrix`` and ``sigmas`` are given, or
        ``sigmas`` does not match ``residuals`` in length.
    """
    if (weight_matrix is None) == (sigmas is None):
        raise ValueError("Provide exactly one of weight_matrix or sigmas.")

    residuals = np.asarray(residuals, dtype="float64").ravel()

    if weight_matrix is not None:
        value = chi2_from_matrix(residuals, weight_matrix)
        return float(np.sqrt(value)) if np.isfinite(value) else value

    sigmas = np.asarray(sigmas, dtype="float64").ravel()
    if sigmas.size != residuals.size:
        raise ValueError(
            f"Sigma vector size {sigmas.size} does not match residual vector size "
            f"{residuals.size}."
        )
    good = np.isfinite(residuals) & np.isfinite(sigmas) & (sigmas > 0.0)
    if not np.any(good):
        return np.nan
    return float(np.sqrt(np.sum((residuals[good] / sigmas[good]) ** 2)))


def reduced_chi2(n_observations: int, n_parameters: int, chi_squared: float) -> float:
    """``chi2 / (N - p)`` with ``N`` observations and ``p`` fitted parameters.

    Parameters
    ----------
    n_observations : int
        Number of observations (rows used in the fit).
    n_parameters : int
        Number of estimated parameters.
    chi_squared : float
        The chi-squared sum (see :func:`chi2`).

    Returns
    -------
    float
        The reduced chi-squared, or ``NaN`` when ``N <= p``.
    """
    dof = n_observations - n_parameters
    if dof <= 0:
        return np.nan
    return float(chi_squared / dof)


def aggregate(
    df: pd.DataFrame,
    by: str | list[str] | None = None,
    *,
    to_unit: str = "arcsec",
) -> pd.DataFrame:
    """Aggregate residuals into a tidy one-row-per-cluster summary table.

    This is the generic replacement for the per-observatory / per-source /
    per-catalog / per-scan-angle reductions the analysis work needs: cluster the
    derived frame on any columns and read the standard residual statistics off
    in one call.

    Parameters
    ----------
    df : pd.DataFrame
        A derived observation frame.
    by : str | list[str] | None
        Grouping column(s).  ``None`` produces a single all-observations row.
        Typical choices: ``"observatory_code"``, ``"source_set_id"``,
        ``"original_reduction"``, ``["observatory_code", "coordinate_kind"]``.
    to_unit : str
        Reporting unit for the residual columns: ``"arcsec"`` (default),
        ``"mas"`` or ``"rad"``.

    Returns
    -------
    pd.DataFrame
        One row per cluster with columns ``n_obs``, ``n_components``,
        ``mean_first``, ``mean_second``, ``std_first``, ``std_second``,
        ``median_first``, ``median_second``, ``min_first``, ``min_second``,
        ``max_first``, ``max_second``, ``rms_first``, ``rms_second``,
        ``wrms`` (weighted RMS of both components) and ``chi2``, in ``to_unit``
        for the angular columns.  ``std_*`` is the sample standard deviation
        (``ddof=1``).

    Raises
    ------
    ValueError
        If ``to_unit`` is not one of ``"rad"``, ``"arcsec"`` or ``"mas"``.
    """
    if to_unit not in _UNIT_FACTOR:
        raise ValueError(f"Unknown unit '{to_unit}'; expected one of {sorted(_UNIT_FACTOR)}.")
    factor = _UNIT_FACTOR[to_unit]

    if by is None:
        clusters = [((), df)]
    else:
        keys = [by] if isinstance(by, str) else list(by)
        clusters = list(df.groupby(keys, dropna=False, sort=True))
        if not clusters:
            clusters = [((), df)]

    records: list[dict] = []
    for key, cluster in clusters:
        first = cluster["residual_first_rad"].to_numpy("float64") * factor
        second = cluster["residual_second_rad"].to_numpy("float64") * factor
        combined = np.concatenate([first, second])
        finite = combined[np.isfinite(combined)]

        sigmas = np.concatenate(
            [
                cluster["sigma_first_rad"].to_numpy("float64") * factor,
                cluster["sigma_second_rad"].to_numpy("float64") * factor,
            ]
        )

        record: dict = {}
        if isinstance(key, tuple):
            for name, value in zip(_key_names(by), key, strict=True):
                record[name] = value
        else:
            record[_key_names(by)[0]] = key

        record.update(
            {
                "n_obs": int(len(cluster)),
                "n_components": int(finite.size),
                "mean_first": mean(first),
                "mean_second": mean(second),
                "std_first": sample_std(first),
                "std_second": sample_std(second),
                "median_first": median(first),
                "median_second": median(second),
                "min_first": minimum(first),
                "min_second": minimum(second),
                "max_first": maximum(first),
                "max_second": maximum(second),
                "rms_first": rms(first),
                "rms_second": rms(second),
                "wrms": weighted_rms(combined, sigmas),
                "chi2": chi2(cluster),
            }
        )
        records.append(record)

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.reset_index(drop=True)
    return result


def _key_names(by: str | list[str] | None) -> list[str]:
    """Return the clustering column names as a list (``["all"]`` when ``by`` is None)."""
    if by is None:
        return ["all"]
    return [by] if isinstance(by, str) else list(by)

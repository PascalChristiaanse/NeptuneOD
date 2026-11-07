import logging

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tudatpy.estimation import observations as obs

from orbitdet.analysis import Whitening, rms
from orbitdet.visualization.base import _make_hover_formatter
from orbitdet.visualization.residual_base import ResidualPlot, _cfg_get

logger = logging.getLogger(__name__)


def _scan_frame_sigmas(
    ra_err: np.ndarray,
    dec_err: np.ndarray,
    corr: np.ndarray,
    psi: np.ndarray,
    dec_obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a per-observation ``(RA, Dec)`` covariance into the scan frame.

    The scan plot works in the tangent plane with ``east = ΔRA·cos(dec)`` and
    ``north = ΔDec``, then rotates into the along-scan / across-scan frame::

        AL = east·sin(psi) + north·cos(psi)
        AC = east·cos(psi) - north·sin(psi)

    The archived covariance is expressed in ``(ΔRA, ΔDec)``, so the rotation must
    be applied to the *whole* covariance (the RA/Dec correlation survives the
    rotation) rather than to the variances alone::

        C_scan = T C T^T,  T = [[sin(psi)·cos(dec), cos(psi)],
                                [cos(psi)·cos(dec), -sin(psi)]]

    Args:
        ra_err: RA error (radians), already de-projected by ``cos(dec)``.
        dec_err: Dec error (radians).
        corr: RA/Dec correlation coefficient.
        psi: Position angle of scan (radians), measured from North towards East.
        dec_obs: Observed declination (radians), for the ``cos(dec)`` factor.

    Returns:
        Tuple ``(sigma_al, sigma_ac)`` in radians.
    """
    ra_err = np.asarray(ra_err, dtype=float)
    dec_err = np.asarray(dec_err, dtype=float)
    corr = np.asarray(corr, dtype=float)
    psi = np.asarray(psi, dtype=float)
    dec_obs = np.asarray(dec_obs, dtype=float)

    cos_dec = np.cos(dec_obs)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    # Covariance entries in (ΔRA, ΔDec).
    c_ra = ra_err**2
    c_dec = dec_err**2
    c_radec = corr * ra_err * dec_err

    # T C T^T diagonal entries (T rows are the AL and AC unit vectors expressed
    # in (ΔRA, ΔDec) coordinates).
    t_al_ra = sin_psi * cos_dec
    t_al_dec = cos_psi
    t_ac_ra = cos_psi * cos_dec
    t_ac_dec = -sin_psi

    var_al = t_al_ra**2 * c_ra + 2.0 * t_al_ra * t_al_dec * c_radec + t_al_dec**2 * c_dec
    var_ac = t_ac_ra**2 * c_ra + 2.0 * t_ac_ra * t_ac_dec * c_radec + t_ac_dec**2 * c_dec

    return np.sqrt(np.clip(var_al, 0.0, None)), np.sqrt(np.clip(var_ac, 0.0, None))


def _scan_frame_components(
    residuals: np.ndarray,
    dec_obs: np.ndarray,
    psi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate ``(ΔRA, ΔDec)`` residuals into the along/across-scan frame.

    Uses the same tangent-plane convention as :func:`_scan_frame_sigmas`::

        east  = ΔRA·cos(dec)
        north = ΔDec
        AL = east·sin(psi) + north·cos(psi)
        AC = east·cos(psi) - north·sin(psi)

    Args:
        residuals: ``(n, 2)`` array of ``[ΔRA, ΔDec]`` residuals in radians.
        dec_obs: Observed declinations (radians).
        psi: Position angle of scan (radians), measured from North towards East.

    Returns:
        Tuple ``(al, ac)`` in radians.
    """
    residuals = np.asarray(residuals, dtype=float)
    dec_obs = np.asarray(dec_obs, dtype=float)
    psi = np.asarray(psi, dtype=float)

    east = residuals[:, 0] * np.cos(dec_obs)
    north = residuals[:, 1]
    al = east * np.sin(psi) + north * np.cos(psi)
    ac = east * np.cos(psi) - north * np.sin(psi)
    return al, ac


def _applied_scan_sigmas(
    observation_set: obs.SingleObservationSet,
    psi: np.ndarray,
    dec_obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scan-frame 1σ uncertainties from the *applied* weight matrix.

    Unlike :func:`_scan_frame_sigmas` (which uses the archived Gaia errors), this
    reads the weight matrix actually assigned by the weighting strategy.  For each
    observation the 2x2 ``(RA, Dec)`` diagonal block of the applied weight matrix
    is inverted to a local covariance and rotated into the scan frame::

        C_i = inv(W[2i:2i+2, 2i:2i+2])
        C_scan = T C_i T^T,  sigma = sqrt(diag(C_scan))

    This is exact for diagonal and block-diagonal weights (e.g. the ``fixed_scan``
    strategy) and a local approximation for full (off-diagonal) weight matrices,
    where it ignores cross-observation correlations.

    Args:
        observation_set: The observation set whose applied weights are read.
        psi: Position angle of scan (radians), one per observation.
        dec_obs: Observed declinations (radians), one per observation.

    Returns:
        Tuple ``(sigma_al, sigma_ac)`` in radians.
    """
    weight_matrix = np.asarray(observation_set.weight_matrix, dtype=float)
    n_obs = len(psi)

    ra_err = np.full(n_obs, np.nan)
    dec_err = np.full(n_obs, np.nan)
    corr = np.zeros(n_obs)

    for i in range(n_obs):
        block = weight_matrix[2 * i : 2 * i + 2, 2 * i : 2 * i + 2]
        try:
            covariance = np.linalg.inv(block)
        except np.linalg.LinAlgError:
            covariance = np.linalg.pinv(block)
        c_ra = max(float(covariance[0, 0]), 0.0)
        c_dec = max(float(covariance[1, 1]), 0.0)
        ra_err[i] = np.sqrt(c_ra)
        dec_err[i] = np.sqrt(c_dec)
        denom = ra_err[i] * dec_err[i]
        corr[i] = float(covariance[0, 1]) / denom if denom > 0.0 else 0.0

    return _scan_frame_sigmas(ra_err, dec_err, corr, psi, dec_obs)


def _applied_radec_sigmas(
    observation_set: obs.SingleObservationSet,
    n_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-observation ``(RA, Dec)`` 1σ uncertainties from the *applied* weight matrix.

    This is the non-rotated counterpart of :func:`_applied_scan_sigmas`: for each
    observation the 2x2 ``(RA, Dec)`` diagonal block of the applied weight matrix
    is inverted to a local covariance, and the marginal sigmas are read off the
    diagonal::

        C_i = inv(W[2i:2i+2, 2i:2i+2])
        sigma_RA  = sqrt(C_i[0, 0])
        sigma_Dec = sqrt(C_i[1, 1])

    This is exact for diagonal and block-diagonal weights and a local
    approximation for full (off-diagonal) weight matrices, where it ignores
    cross-observation correlations.

    Args:
        observation_set: The observation set whose applied weights are read.
        n_obs: Number of observations in the set.

    Returns:
        Tuple ``(sigma_ra, sigma_dec)`` in radians.

    Raises:
        ValueError: If the applied weight matrix does not have the expected
            ``(2 * n_obs, 2 * n_obs)`` shape (e.g. no weights were applied).
    """
    weight_matrix = np.asarray(observation_set.weight_matrix, dtype=float)
    if weight_matrix.shape[0] != 2 * n_obs:
        raise ValueError(
            f"Applied weight matrix has shape {weight_matrix.shape}, expected "
            f"({2 * n_obs}, {2 * n_obs})."
        )

    sigma_ra = np.full(n_obs, np.nan)
    sigma_dec = np.full(n_obs, np.nan)

    for i in range(n_obs):
        block = weight_matrix[2 * i : 2 * i + 2, 2 * i : 2 * i + 2]
        try:
            covariance = np.linalg.inv(block)
        except np.linalg.LinAlgError:
            covariance = np.linalg.pinv(block)
        sigma_ra[i] = np.sqrt(max(float(covariance[0, 0]), 0.0))
        sigma_dec[i] = np.sqrt(max(float(covariance[1, 1]), 0.0))

    return sigma_ra, sigma_dec


def _archive_radec_sigmas(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Combine archived Gaia random and systematic errors in quadrature.

    Args:
        errors: ``(n, 6)`` array in :data:`orbitdet.data.gaia.GAIA_ERROR_COLUMNS`
            order (radians, RA errors already de-projected by ``cos(dec)``).

    Returns:
        Tuple ``(sigma_ra, sigma_dec)`` in radians.
    """
    errors = np.asarray(errors, dtype=float)
    ra_err_random, dec_err_random, _corr_random, ra_err_sys, dec_err_sys, _corr_sys = errors.T
    sigma_ra = np.sqrt(ra_err_random**2 + ra_err_sys**2)
    sigma_dec = np.sqrt(dec_err_random**2 + dec_err_sys**2)
    return sigma_ra, sigma_dec


def _rad_to_arcsec(angle_rad: np.ndarray) -> np.ndarray:
    return np.rad2deg(angle_rad) * 3600.0


def _unit_factor(unit: str) -> float:
    """Number of the chosen unit in one radian.

    Supported units are ``"arcsec"`` (default) and ``"mas"``.

    Args:
        unit: Display unit for the residuals. One of ``"arcsec"`` or ``"mas"``.

    Returns:
        Conversion factor from radians to the requested unit.
    """
    if unit == "mas":
        return np.rad2deg(1.0) * 3600.0 * 1e3
    # Default: arcseconds
    return np.rad2deg(1.0) * 3600.0


def _rad_to_unit(angle_rad: np.ndarray, unit: str) -> np.ndarray:
    """Convert angles from radians to the requested display unit."""
    return np.asarray(angle_rad) * _unit_factor(unit)


def _principal_angle_rad(angle_rad: np.ndarray) -> np.ndarray:
    return np.remainder(angle_rad + np.pi, 2.0 * np.pi) - np.pi


def _rms_in_unit(values_in_unit: np.ndarray, unit: str) -> float | None:
    finite_values = values_in_unit[np.isfinite(values_in_unit)]
    if finite_values.size == 0:
        return None

    rms = float(np.sqrt(np.mean(np.square(finite_values))))
    # Report very small RMS values (e.g. in mas) without losing precision.
    return rms


def _seconds_since_j2000_to_datetimes(seconds_since_j2000: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(
        seconds_since_j2000,
        unit="s",
        origin=pd.Timestamp("2000-01-01T12:00:00"),
    )


def _configure_datetime_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


class Residuals(ResidualPlot):
    """Plot residuals versus epoch, grouped by a standard-frame column.

    Consumes an :class:`~orbitdet.analysis.access.ObservationAccess` - the
    standard observation frame joined to the live observation sets - and draws
    one scatter series per cluster of a chosen column.  By default the clustering is
    ``observatory_code``, so the same plot covers "per observatory", "per set
    id", "per source", "per coordinate kind" or any other standard-frame column
    without new code: set ``group_by`` in the figure config.  Each cluster's
    legend entry reports its per-component RMS, read from
    :func:`orbitdet.analysis.statistics.rms`.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (``conf/figures/<name>.yaml``).
        ``group_by``, ``axes.unit`` and ``include_rejected`` are read from here.
    access : ObservationAccess
        The access layer whose derived frame is plotted.
    """

    def _make_figure(self):
        plot_cfg = self.cfg
        fig_w, fig_h = self._figure_size()
        fig, axs = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)

        unit = self.unit

        # One tidy frame: standard columns + residual/weight/sigma + unit columns.
        frame = self._frame(unit=unit, scan=False, normalized=False)

        first_col = f"residual_first_rad_{unit}"
        second_col = f"residual_second_rad_{unit}"
        if first_col not in frame.columns or second_col not in frame.columns:
            raise ValueError(
                f"Analysis frame has no '{first_col}'/'{second_col}' column; "
                f"cannot plot residuals in unit '{unit}'."
            )

        group_cols = self._grouping()
        self._require_columns(frame, group_cols)

        colors, marker_size, marker_types = self._styling()

        for index, (key, cluster) in enumerate(frame.groupby(group_cols, dropna=False, sort=True)):
            color = colors(index % colors.N)
            marker = marker_types[index % len(marker_types)]
            label = self._group_label(key, group_cols)

            epochs = _seconds_since_j2000_to_datetimes(cluster["epoch_tdb"].to_numpy("float64"))
            first = cluster[first_col].to_numpy("float64")
            second = cluster[second_col].to_numpy("float64")

            axs[0].scatter(
                epochs,
                first,
                marker=marker,
                s=marker_size,
                color=color,
                alpha=0.5,
                label=self._rms_label(label, rms(first), unit),
            )
            axs[1].scatter(
                epochs,
                second,
                marker=marker,
                s=marker_size,
                color=color,
                alpha=0.5,
                label=self._rms_label(label, rms(second), unit),
            )

        axs[0].set_title(_cfg_get(plot_cfg, "titles", "ra", default="Right Ascension"))
        axs[1].set_title(_cfg_get(plot_cfg, "titles", "dec", default="Declination"))
        y_label = _cfg_get(plot_cfg, "axes", "y_label", default=f"Residual [{unit}]")
        axs[0].set_ylabel(y_label)
        axs[1].set_ylabel(y_label)
        x_label = _cfg_get(plot_cfg, "axes", "x_label", default="Epoch")
        axs[0].set_xlabel(x_label)
        axs[1].set_xlabel(x_label)
        _configure_datetime_axis(axs[0])
        _configure_datetime_axis(axs[1])

        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default="Epoch")
        hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default=f"Residual [{unit}]")
        fmt = _make_hover_formatter(hover_x_label, hover_y_label)
        axs[0].format_coord = fmt
        axs[1].format_coord = fmt

        self._legend(axs[0])
        self._legend(axs[1])
        fig.suptitle(self._suptitle(frame, "suptitle", "Residuals for {target_name}"))
        fig.set_tight_layout(True)

        return fig, axs


class ResidualsScan(ResidualPlot):
    """Plot along-scan (AL) and across-scan (AC) residuals, grouped by a column.

    Consumes an :class:`~orbitdet.analysis.access.ObservationAccess` and plots the
    AL/AC residuals already rotated into the Gaia scan frame by
    :func:`orbitdet.analysis.frames.analysis_frame`.  The scan angle is read from
    the standard frame's ``scan_angle`` column (populated on ingest from the Gaia
    archive's ``position_angle_scan``), so this plot no longer walks the
    observation sets itself: it reads ``residual_along_scan_rad`` /
    ``residual_across_scan_rad`` (and their unit columns) straight from the
    analysis frame.

    Only scanning-instrument rows carry a scan angle; frames without one cannot
    be plotted in the scan frame and raise a :class:`ValueError`.  Rows grouped by
    ``group_by`` (default ``"observatory_code"``), one scatter series per cluster,
    with each cluster's RMS in the legend, exactly like :class:`Residuals`.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (``conf/figures/<name>.yaml``).
        ``group_by``, ``axes.unit`` and ``include_rejected`` are read from here.
    access : ObservationAccess
        The access layer whose derived frame is plotted.
    """

    def _make_figure(self):
        plot_cfg = self.cfg
        fig_w, fig_h = self._figure_size()
        fig, axs = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharex=True)

        unit = self.unit

        # One tidy frame with the scan-frame columns added by the access layer.
        frame = self._frame(unit=unit, scan=True, normalized=False)
        al_col = f"residual_along_scan_rad_{unit}"
        ac_col = f"residual_across_scan_rad_{unit}"
        has_angle = "scan_angle" in frame.columns and frame["scan_angle"].notna().any()
        if not has_angle or al_col not in frame.columns or ac_col not in frame.columns:
            raise ValueError(
                "ResidualsScan requires Gaia scan angles, but the frame carries no "
                "scan_angle (the source did not record a position_angle_scan)."
            )
        # Only rows with a scan angle belong in the scan frame; a mixed collection
        # (Gaia plus other observatories) keeps the Gaia rows and drops the rest.
        frame = frame.loc[frame["scan_angle"].notna()]

        group_cols = self._grouping()
        self._require_columns(frame, group_cols)

        colors, marker_size, marker_types = self._styling()

        for index, (key, cluster) in enumerate(frame.groupby(group_cols, dropna=False, sort=True)):
            color = colors(index % colors.N)
            marker = marker_types[index % len(marker_types)]
            label = self._group_label(key, group_cols)

            epochs = _seconds_since_j2000_to_datetimes(cluster["epoch_tdb"].to_numpy("float64"))
            along = cluster[al_col].to_numpy("float64")
            across = cluster[ac_col].to_numpy("float64")

            axs[0].scatter(
                epochs,
                along,
                marker=marker,
                s=marker_size,
                color=color,
                alpha=0.5,
                label=self._rms_label(label, rms(along), unit),
            )
            axs[1].scatter(
                epochs,
                across,
                marker=marker,
                s=marker_size,
                color=color,
                alpha=0.5,
                label=self._rms_label(label, rms(across), unit),
            )

        axs[0].set_title(_cfg_get(plot_cfg, "titles", "al", default="Along-scan"))
        axs[1].set_title(_cfg_get(plot_cfg, "titles", "ac", default="Across-scan"))
        y_label = _cfg_get(plot_cfg, "axes", "y_label", default=f"Residual [{unit}]")
        axs[0].set_ylabel(y_label)
        axs[1].set_ylabel(y_label)
        x_label = _cfg_get(plot_cfg, "axes", "x_label", default="Epoch")
        axs[0].set_xlabel(x_label)
        axs[1].set_xlabel(x_label)
        _configure_datetime_axis(axs[0])
        _configure_datetime_axis(axs[1])

        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default="Epoch")
        hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default=f"Residual [{unit}]")
        fmt = _make_hover_formatter(hover_x_label, hover_y_label)
        axs[0].format_coord = fmt
        axs[1].format_coord = fmt

        self._legend(axs[0])
        self._legend(axs[1])
        fig.suptitle(self._suptitle(frame, "suptitle_scan", "Scan Residuals for {target_name}"))
        fig.set_tight_layout(True)

        return fig, axs


class NormalizedResiduals(ResidualPlot):
    """Plot per-observation normalized residuals, grouped by a column.

    Consumes an :class:`~orbitdet.analysis.access.ObservationAccess` and plots the
    normalized residuals ``z = r / σ`` computed by
    :func:`orbitdet.analysis.statistics.normalized_residuals` - each residual
    component divided by the 1σ its own applied weight matrix implies.  Because no
    scan-frame rotation is required this works for *every* observation type
    (NSDB astrometry, Voyager, Gaia, ...), and the normalized columns already come
    out of :func:`orbitdet.analysis.frames.analysis_frame`.

    A well-calibrated solution has ``RMS(z) ~ 1``; reference lines are drawn at 0
    and ±1.  Rows are grouped by ``group_by`` (default ``"observatory_code"``), one
    scatter series per cluster, with each cluster's ``RMS(z)`` in the legend.

    The normalization is the *applied-weight* sigma (the ``sigma_*_rad`` columns
    the access layer fills).  For a full (off-diagonal) weight matrix this is the
    componentwise normalization and ignores cross-observation correlation - which
    is why the scan-frame figure (:class:`NormalizedResidualScan`) whitens
    instead.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (``conf/figures/<name>.yaml``).
        ``group_by`` and ``include_rejected`` are read from here.
    access : ObservationAccess
        The access layer whose derived frame is plotted.
    """

    def _make_figure(self):
        plot_cfg = self.cfg
        fig_w, fig_h = self._figure_size()
        fig, axs = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)

        # The normalized columns come from the analysis frame.
        frame = self._frame(unit=None, scan=False, normalized=True)
        # Scanning-instrument rows (a finite scan_angle, e.g. Gaia) are not shown
        # here: their RA/Dec normalization ignores the intra-scan correlation and
        # is better handled by NormalizedResidualScan, which whitens.
        if "scan_angle" in frame.columns:
            frame = frame.loc[frame["scan_angle"].isna()]
        if frame.empty:
            raise ValueError(
                "NormalizedResiduals has no rows to plot: every observation carries a "
                "scan_angle, so use NormalizedResidualScan for a scanning instrument."
            )

        for column in ("z_first", "z_second"):
            if column not in frame.columns:
                raise ValueError(f"Analysis frame has no '{column}' column.")

        group_cols = self._grouping()
        self._require_columns(frame, group_cols)

        colors, marker_size, marker_types = self._styling()

        for index, (key, cluster) in enumerate(frame.groupby(group_cols, dropna=False, sort=True)):
            color = colors(index % colors.N)
            marker = marker_types[index % len(marker_types)]
            label = self._group_label(key, group_cols)

            epochs = _seconds_since_j2000_to_datetimes(cluster["epoch_tdb"].to_numpy("float64"))
            z_first = cluster["z_first"].to_numpy("float64")
            z_second = cluster["z_second"].to_numpy("float64")

            axs[0].scatter(
                epochs,
                z_first,
                marker=marker,
                s=marker_size,
                color=color,
                alpha=0.5,
                label=self._rms_label(label, rms(z_first)),
            )
            axs[1].scatter(
                epochs,
                z_second,
                marker=marker,
                s=marker_size,
                color=color,
                alpha=0.5,
                label=self._rms_label(label, rms(z_second)),
            )

        # Reference lines at 0 and +/-1 sigma.
        for ax in axs:
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
            ax.axhline(-1.0, color="grey", linestyle="--", linewidth=0.8)

        axs[0].set_title(_cfg_get(plot_cfg, "titles", "ra", default="Right Ascension"))
        axs[1].set_title(_cfg_get(plot_cfg, "titles", "dec", default="Declination"))
        y_label = _cfg_get(plot_cfg, "axes", "y_label", default="Normalized residual [$\\sigma$]")
        axs[0].set_ylabel(y_label)
        axs[1].set_ylabel(y_label)
        x_label = _cfg_get(plot_cfg, "axes", "x_label", default="Epoch")
        axs[0].set_xlabel(x_label)
        axs[1].set_xlabel(x_label)
        _configure_datetime_axis(axs[0])
        _configure_datetime_axis(axs[1])

        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default="Epoch")
        hover_y_label = _cfg_get(
            plot_cfg, "axes", "hover_y_label", default="Normalized residual [$\\sigma$]"
        )
        fmt = _make_hover_formatter(hover_x_label, hover_y_label)
        axs[0].format_coord = fmt
        axs[1].format_coord = fmt

        self._legend(axs[0])
        self._legend(axs[1])
        fig.suptitle(self._suptitle(frame, "suptitle", "Normalized Residuals for {target_name}"))
        fig.set_tight_layout(True)

        return fig, axs


class NormalizedResidualScan(ResidualPlot):
    """Plot per-set Cholesky-whitened along/across-scan residuals for Gaia.

    Consumes an :class:`~orbitdet.analysis.access.ObservationAccess` and, for each
    Gaia set, whitens the scan-frame residual vector through the Cholesky factor of
    the set's applied weight matrix::

        W_scan = L L^T,   z = L^T r_scan,   Cov(z) = I,

    then plots the two interleaved components of ``z`` (``z[0::2]`` as along-scan,
    ``z[1::2]`` as across-scan) against the observation epochs.  For a
    block-diagonal (per-observation) weight matrix ``L`` is block-diagonal too, so
    the two columns are exactly the per-observation whitened AL/AC; for a dense
    (cross-observation) matrix the components mix across the set, which is the
    point - the correlation is removed rather than ignored.  A rank-deficient
    weight matrix (a near-rank-1 archive transit covariance) has no lower
    Cholesky factor, so the per-observation 2x2 blocks are whitened instead.

    This is the correct reduction for correlated observations, replacing the
    marginal-σ normalization: dividing each component by its marginal σ leaves the
    RA/Dec correlation in and inflates the variance.  A well-calibrated solution
    has ``RMS(z) ~ 1``; reference lines are at 0 and ±1.

    Only scanning-instrument rows carry a scan angle; a frame without one raises a
    :class:`ValueError`.  Rows are grouped by ``group_by`` (default
    ``"observatory_code"``), one scatter series per cluster.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (``conf/figures/<name>.yaml``).
        ``group_by`` and ``include_rejected`` are read from here.
    access : ObservationAccess
        The access layer whose derived frame is plotted.
    """

    def _whitened_set(
        self, set_index: int, rows: pd.DataFrame, al_col: str, ac_col: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return ``(epochs, z_along, z_across)`` for one set, whitened, or ``None``.

        The scan-frame residual vector is whitened through the Cholesky factor of
        the set's scan-basis weight matrix.  When that factor is rank-deficient the
        per-observation 2x2 blocks are whitened instead, so the AL/AC columns stay
        defined.
        """
        if rows.empty:
            return None
        first = rows[al_col].to_numpy("float64")
        second = rows[ac_col].to_numpy("float64")
        keep = np.isfinite(first) & np.isfinite(second)
        if not keep.any():
            return None

        weight_matrix = self.access.weight_matrix(set_index, basis="scan")
        n_obs = len(rows)
        if weight_matrix.shape[0] == 2 * n_obs:
            index = np.repeat(keep, 2)
            weight_matrix = weight_matrix[np.ix_(index, index)]

        residual = np.stack([first[keep], second[keep]], axis=1).ravel()
        if weight_matrix.shape[0] != residual.size:
            return None

        epochs = pd.to_numeric(rows["epoch_tdb"], errors="coerce").to_numpy("float64")[keep]
        whitening = Whitening.from_weight_matrix(weight_matrix)
        z = whitening.apply(residual)
        if z.size == residual.size:
            return epochs, z[0::2], z[1::2]

        # Rank-deficient: whiten each observation's own 2x2 block.
        z_al = np.full(residual.size // 2, np.nan)
        z_ac = np.full(residual.size // 2, np.nan)
        for i in range(residual.size // 2):
            block = weight_matrix[2 * i : 2 * i + 2, 2 * i : 2 * i + 2]
            block_whitening = Whitening.from_weight_matrix(block)
            z_pair = block_whitening.apply(residual[2 * i : 2 * i + 2])
            z_al[i] = z_pair[0] if z_pair.size > 0 else np.nan
            z_ac[i] = z_pair[1] if z_pair.size > 1 else np.nan
        return epochs, z_al, z_ac

    def _make_figure(self):
        plot_cfg = self.cfg
        fig_w, fig_h = self._figure_size()
        fig, axs = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)

        # Scan-frame residuals live in the analysis frame; whitening needs the
        # set weight matrix, so the residue columns are read per set below.
        frame = self._frame(unit=None, scan=True, normalized=False)
        al_col = "residual_along_scan_rad"
        ac_col = "residual_across_scan_rad"
        has_angle = "scan_angle" in frame.columns and frame["scan_angle"].notna().any()
        if not has_angle or al_col not in frame.columns or ac_col not in frame.columns:
            raise ValueError(
                "NormalizedResidualScan requires Gaia scan angles, but the frame carries no "
                "scan_angle (the source did not record a position_angle_scan)."
            )

        group_frame = frame.loc[frame["scan_angle"].notna()]

        group_cols = self._grouping()
        self._require_columns(group_frame, group_cols)

        colors, marker_size, marker_types = self._styling()

        for index, (key, cluster) in enumerate(
            group_frame.groupby(group_cols, dropna=False, sort=True)
        ):
            color = colors(index % colors.N)
            marker = marker_types[index % len(marker_types)]
            label = self._group_label(key, group_cols)

            set_indices = (
                cluster["observation_set_index"].dropna().astype(int).unique().tolist()
                if "observation_set_index" in cluster.columns
                else []
            )
            set_column = (
                cluster["observation_set_index"].astype(int)
                if "observation_set_index" in cluster.columns
                else None
            )
            epoch_chunks: list[np.ndarray] = []
            al_chunks: list[np.ndarray] = []
            ac_chunks: list[np.ndarray] = []
            for set_index in set_indices:
                if set_column is None:
                    rows = cluster
                else:
                    rows = cluster.loc[set_column == set_index]
                whitened = self._whitened_set(set_index, rows, al_col, ac_col)
                if whitened is None:
                    continue
                epochs, z_al, z_ac = whitened
                epoch_chunks.append(epochs)
                al_chunks.append(z_al)
                ac_chunks.append(z_ac)
            if not epoch_chunks:
                continue

            epochs = _seconds_since_j2000_to_datetimes(np.concatenate(epoch_chunks))
            z_al = np.concatenate(al_chunks)
            z_ac = np.concatenate(ac_chunks)

            axs[0].scatter(
                epochs,
                z_al,
                marker=marker,
                s=marker_size,
                color=color,
                alpha=0.5,
                label=self._rms_label(label, rms(z_al)),
            )
            axs[1].scatter(
                epochs,
                z_ac,
                marker=marker,
                s=marker_size,
                color=color,
                alpha=0.5,
                label=self._rms_label(label, rms(z_ac)),
            )

        # Reference lines at 0 and +/-1 sigma.
        for ax in axs:
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
            ax.axhline(-1.0, color="grey", linestyle="--", linewidth=0.8)

        axs[0].set_title(_cfg_get(plot_cfg, "titles", "al", default="Along-scan"))
        axs[1].set_title(_cfg_get(plot_cfg, "titles", "ac", default="Across-scan"))
        y_label = _cfg_get(plot_cfg, "axes", "y_label", default="Normalized residual [$\\sigma$]")
        axs[0].set_ylabel(y_label)
        axs[1].set_ylabel(y_label)
        x_label = _cfg_get(plot_cfg, "axes", "x_label", default="Epoch")
        axs[0].set_xlabel(x_label)
        axs[1].set_xlabel(x_label)
        _configure_datetime_axis(axs[0])
        _configure_datetime_axis(axs[1])

        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default="Epoch")
        hover_y_label = _cfg_get(
            plot_cfg, "axes", "hover_y_label", default="Normalized residual [$\\sigma$]"
        )
        fmt = _make_hover_formatter(hover_x_label, hover_y_label)
        axs[0].format_coord = fmt
        axs[1].format_coord = fmt

        self._legend(axs[0])
        self._legend(axs[1])
        fig.suptitle(
            self._suptitle(frame, "suptitle_scan", "Normalized Scan Residuals for {target_name}")
        )
        fig.set_tight_layout(True)

        return fig, axs

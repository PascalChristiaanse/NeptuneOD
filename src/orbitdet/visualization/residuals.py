import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tudatpy.estimation import observations as obs
from tudatpy.estimation.observable_models_setup import links
from tudatpy.estimation.observations import observations_processing as obs_proc

from orbitdet.data.gaia_data import get_scan_angles_for_epochs
from orbitdet.observations import get_observatory_info
from orbitdet.visualization.base import Plot


def _cfg_get(cfg: DictConfig | dict | None, *keys, default=None):
    cur = cfg
    for k in keys:
        if cur is None:
            return default
        try:
            # DictConfig supports get
            cur = cur.get(k)
        except Exception:
            try:
                cur = cur[k]
            except Exception:
                return default
    return default if cur is None else cur


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


def _make_hover_formatter(hover_x_label: str, hover_y_label: str):
    def _format(x, y):
        try:
            dt = mdates.num2date(x)
            xs = dt.isoformat(sep=" ")
        except Exception:
            xs = f"{x:.6g}"
        return f"{hover_x_label}: {xs}, {hover_y_label}: {y:.3e}"

    return _format


class Residuals(Plot):
    """Plot pre-fit and post-fit residuals for the orbit determination."""

    def __init__(
        self,
        cfg: DictConfig,
        observation_collection: obs.ObservationCollection,
        observation_parsers: list[obs_proc.ObservationParserType] | None = None,
    ):
        super().__init__(cfg)
        self.observation_collection = observation_collection
        self.observation_parsers = observation_parsers

    def _make_figure(self):
        cfg = self.cfg
        observation_collection = self.observation_collection
        observation_parsers = self.observation_parsers

        """Plot pre-fit and post-fit residuals for the orbit determination."""
        if observation_parsers is None:
            observation_sets: list[obs.SingleObservationSet] = (
                observation_collection.get_single_observation_sets()
            )
        else:
            observation_sets: list[obs.SingleObservationSet] = (
                observation_collection.get_single_observation_sets(observation_parsers)
            )

        # Load plotting configuration
        plot_cfg = _cfg_get(cfg, "residuals", default=None)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=8.27 * 2)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=8.27 * 2 / 2)
        unit = _cfg_get(plot_cfg, "axes", "unit", default="arcsec")

        fig, axs = plt.subplots(
            2,
            1,
            figsize=(fig_w, fig_h),
            sharex=True,
        )
        cmap = _cfg_get(plot_cfg, "styling", "cmap", default="tab10")
        colors = plt.get_cmap(cmap)
        marker_size = _cfg_get(plot_cfg, "styling", "marker_size", default=30)
        marker_types = [
            "o",
            "s",
            "D",
            "^",
            "v",
            "<",
            ">",
            "P",
            "X",
        ]  # cycle through marker types if more sets than colors
        for set_index, obs_set in enumerate(observation_sets):
            observatory_code = obs_set.link_definition.link_ends[links.receiver].reference_point
            if observatory_code == "":
                # Missing reference points imply spacecraft which use receiver name instead for info
                # lookup and labeling
                observatory_name = obs_set.link_definition.link_ends[links.receiver].body_name
                info = {"code": observatory_code}
                info["name"] = observatory_name
                info["region"] = "Spacecraft"
            else:
                info = get_observatory_info(cfg, observatory_code)
            target_name = obs_set.link_definition.link_ends[links.transmitter].body_name
            color = colors(set_index % colors.N)
            marker = marker_types[set_index % len(marker_types)]

            obs_times_sec_j2000 = np.array(
                [epoch.to_float() for epoch in obs_set.observation_times]
            )
            obs_times = _seconds_since_j2000_to_datetimes(obs_times_sec_j2000)
            residuals = np.array(obs_set.residuals)
            # n x 2 array of RA and DEC residuals in radians

            # Both RA and DEC residuals are circular; fold them to the principal interval
            # before converting to avoid wrapping artifacts near ±180° / 360°.
            # ra_residuals_arcsec = _rad_to_arcsec(_principal_angle_rad(residuals[:, 0]))
            # dec_residuals_arcsec = _rad_to_arcsec(_principal_angle_rad(residuals[:, 1]))

            # Dont use wrapping (TEST)
            ra_residuals_unit = _rad_to_unit(residuals[:, 0], unit)
            dec_residuals_unit = _rad_to_unit(residuals[:, 1], unit)

            ra_rms_unit = _rms_in_unit(ra_residuals_unit, unit)
            dec_rms_unit = _rms_in_unit(dec_residuals_unit, unit)
            ra_rms_label = f"{ra_rms_unit:.3e} {unit}" if ra_rms_unit is not None else None
            dec_rms_label = f"{dec_rms_unit:.3e} {unit}" if dec_rms_unit is not None else None

            # RA
            axs[0].scatter(
                obs_times,
                ra_residuals_unit,
                marker=marker,
                s=marker_size,
                label=f"{info['name']} - {info['region']} - RMS: {ra_rms_label}",
                color=color,
                alpha=0.5,
            )
            # DEC
            axs[1].scatter(
                obs_times,
                dec_residuals_unit,
                marker=marker,
                s=marker_size,
                label=f"{info['name']} - {info['region']} - RMS: {dec_rms_label}",
                color=color,
                alpha=0.5,
            )

        # Titles and labels (configurable)
        title_ra = _cfg_get(plot_cfg, "titles", "ra", default="Right Ascension")
        title_dec = _cfg_get(plot_cfg, "titles", "dec", default="Declination")
        suptitle = _cfg_get(
            plot_cfg, "titles", "suptitle", default=f"Pre-Fit Residuals for {target_name}"
        )
        # Allow templates like "Pre-Fit Residuals for {target_name}" in config
        try:
            if isinstance(suptitle, str):
                suptitle = suptitle.format(target_name=target_name)
        except Exception:
            # leave suptitle unchanged if formatting fails
            pass

        axs[0].set_title(title_ra)
        axs[1].set_title(title_dec)
        y_label = _cfg_get(plot_cfg, "axes", "y_label", default=f"Residual [{unit}]")
        axs[0].set_ylabel(y_label)
        axs[1].set_ylabel(y_label)
        x_label = _cfg_get(plot_cfg, "axes", "x_label", default="Epoch")
        axs[0].set_xlabel(x_label)
        _configure_datetime_axis(axs[0])
        axs[1].set_xlabel(x_label)
        _configure_datetime_axis(axs[1])

        # Hover formatter
        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default="Epoch")
        hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default="Residual [arcsec]")
        fmt = _make_hover_formatter(hover_x_label, hover_y_label)
        axs[0].format_coord = fmt
        axs[1].format_coord = fmt
        # axs[0].tick_params(axis="x", which="both", labelbottom=False)
        # add legend with observatory names and RMS values
        # Legend placement (configurable)
        # Ensure legend params have correct types
        try:
            legend_ncols = int(_cfg_get(plot_cfg, "legend", "ncols", default=2))
        except Exception:
            legend_ncols = 2

        bbox = _cfg_get(plot_cfg, "legend", "bbox_to_anchor", default={"x": 0.5, "y": -0.15})
        try:
            if isinstance(bbox, dict):
                bbox_tuple = (float(bbox.get("x", 0.5)), float(bbox.get("y", -0.15)))
            else:
                # coerce sequence values to floats
                bbox_tuple = tuple(float(x) for x in bbox)
        except Exception:
            bbox_tuple = (0.5, -0.15)

        axs[0].legend(ncols=legend_ncols, loc="upper center", bbox_to_anchor=bbox_tuple)
        axs[1].legend(ncols=legend_ncols, loc="upper center", bbox_to_anchor=bbox_tuple)
        fig.suptitle(suptitle)
        fig.set_tight_layout(True)

        # Optionally save to file
        out = _cfg_get(plot_cfg, "output_file", default=None)
        if out:
            fig.savefig(out)

        return fig, axs


class ResidualsScan(Plot):
    """Plot along-scan (AL) and across-scan (AC) residuals for Gaia observations.

    Rotates the RA/Dec residuals into the Gaia scan frame using the archived
    ``position_angle_scan``.  This is registered in :mod:`orbitdet.data.gaia_data`
    (see :func:`get_scan_angles_for_epochs`) when the observations are built via
    :meth:`GaiaQuery.to_tudat`.
    """

    def __init__(
        self,
        cfg: DictConfig,
        observation_collection: obs.ObservationCollection,
        observation_parsers: list[obs_proc.ObservationParserType] | None = None,
    ):
        super().__init__(cfg)
        self.observation_collection = observation_collection
        self.observation_parsers = observation_parsers

    def _make_figure(self):
        cfg = self.cfg
        observation_collection = self.observation_collection
        observation_parsers = self.observation_parsers

        if observation_parsers is None:
            observation_sets: list[obs.SingleObservationSet] = (
                observation_collection.get_single_observation_sets()
            )
        else:
            observation_sets: list[obs.SingleObservationSet] = (
                observation_collection.get_single_observation_sets(observation_parsers)
            )

        plot_cfg = _cfg_get(cfg, "residuals", default=None)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=8.27 * 2)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=8.27 * 2 / 2)
        unit = _cfg_get(plot_cfg, "axes", "unit", default="arcsec")

        fig, axs = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharex=True)
        cmap = _cfg_get(plot_cfg, "styling", "cmap", default="tab10")
        colors = plt.get_cmap(cmap)
        marker_size = _cfg_get(plot_cfg, "styling", "marker_size", default=30)
        marker_types = ["o", "s", "D", "^", "v", "<", ">", "P", "X"]

        al_labels = []
        ac_labels = []
        for set_index, obs_set in enumerate(observation_sets):
            observatory_code = obs_set.link_definition.link_ends[links.receiver].reference_point
            if observatory_code == "":
                # Missing reference points imply spacecraft which use receiver name instead
                observatory_name = obs_set.link_definition.link_ends[links.receiver].body_name
                info = {"code": observatory_code}
                info["name"] = observatory_name
                info["region"] = "Spacecraft"
            else:
                info = get_observatory_info(cfg, observatory_code)
            target_name = obs_set.link_definition.link_ends[links.transmitter].body_name
            color = colors(set_index % colors.N)
            marker = marker_types[set_index % len(marker_types)]

            obs_times_sec_j2000 = np.array(
                [epoch.to_float() for epoch in obs_set.observation_times]
            )
            obs_times = _seconds_since_j2000_to_datetimes(obs_times_sec_j2000)
            residuals = np.array(obs_set.residuals)  # (n,2) [ΔRA, ΔDec] in rad

            # Observed angles, used for the cos(dec) tangent-plane factor.
            observed = np.array(obs_set.concatenated_observations).reshape(-1, 2)
            dec_obs = observed[:, 1]

            # Physical angular offsets in the tangent plane (radians):
            #   east  = ΔRA * cos(Dec)   (direction of increasing RA)
            #   north = ΔDec             (direction of increasing Dec)
            east = residuals[:, 0] * np.cos(dec_obs)
            north = residuals[:, 1]

            # Scan angles (radians) for these epochs.
            scan_angles = get_scan_angles_for_epochs(obs_times_sec_j2000)
            if scan_angles is None:
                raise RuntimeError(
                    "Could not find scan angles for the observation epochs. Ensure the "
                    "observations were created via GaiaQuery.to_tudat(), which registers "
                    "the position_angle_scan for each observation."
                )
            psi = scan_angles  # measured from North towards East

            # Along-scan (AL) direction: unit vector in (east, north) coordinates.
            #   AL = (sin psi, cos psi)
            # Across-scan (AC) direction: perpendicular to AL.
            #   AC = (cos psi, -sin psi)
            al_in_unit = _rad_to_unit(east * np.sin(psi) + north * np.cos(psi), unit)
            ac_in_unit = _rad_to_unit(east * np.cos(psi) - north * np.sin(psi), unit)

            al_rms_unit = _rms_in_unit(al_in_unit, unit)
            ac_rms_unit = _rms_in_unit(ac_in_unit, unit)
            al_rms_label = f"{al_rms_unit:.3e} {unit}" if al_rms_unit is not None else None
            ac_rms_label = f"{ac_rms_unit:.3e} {unit}" if ac_rms_unit is not None else None

            al_labels.append(f"{info['name']} - {info['region']} - RMS: {al_rms_label}")
            ac_labels.append(f"{info['name']} - {info['region']} - RMS: {ac_rms_label}")

            # Along-scan
            axs[0].scatter(
                obs_times,
                al_in_unit,
                marker=marker,
                s=marker_size,
                label=al_labels[-1],
                color=color,
                alpha=0.5,
            )
            # Across-scan
            axs[1].scatter(
                obs_times,
                ac_in_unit,
                marker=marker,
                s=marker_size,
                label=ac_labels[-1],
                color=color,
                alpha=0.5,
            )

        # Titles and labels (configurable)
        title_al = _cfg_get(plot_cfg, "titles", "al", default="Along-scan")
        title_ac = _cfg_get(plot_cfg, "titles", "ac", default="Across-scan")
        suptitle = _cfg_get(
            plot_cfg, "titles", "suptitle_scan", default=f"Pre-Fit Scan Residuals for {target_name}"
        )
        try:
            if isinstance(suptitle, str):
                suptitle = suptitle.format(target_name=target_name)
        except Exception:
            pass

        axs[0].set_title(title_al)
        axs[1].set_title(title_ac)
        y_label = _cfg_get(plot_cfg, "axes", "y_label", default=f"Residual [{unit}]")
        axs[0].set_ylabel(y_label)
        axs[1].set_ylabel(y_label)
        x_label = _cfg_get(plot_cfg, "axes", "x_label", default="Epoch")
        axs[0].set_xlabel(x_label)
        _configure_datetime_axis(axs[0])
        axs[1].set_xlabel(x_label)
        _configure_datetime_axis(axs[1])

        # Hover formatter
        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default="Epoch")
        hover_y_label = _cfg_get(
            plot_cfg, "axes", "hover_y_label", default=f"Residual [{unit}]"
        )
        fmt = _make_hover_formatter(hover_x_label, hover_y_label)
        axs[0].format_coord = fmt
        axs[1].format_coord = fmt

        # Legend placement
        try:
            legend_ncols = int(_cfg_get(plot_cfg, "legend", "ncols", default=2))
        except Exception:
            legend_ncols = 2

        bbox = _cfg_get(plot_cfg, "legend", "bbox_to_anchor", default={"x": 0.5, "y": -0.15})
        try:
            if isinstance(bbox, dict):
                bbox_tuple = (float(bbox.get("x", 0.5)), float(bbox.get("y", -0.15)))
            else:
                bbox_tuple = tuple(float(x) for x in bbox)
        except Exception:
            bbox_tuple = (0.5, -0.15)

        axs[0].legend(ncols=legend_ncols, loc="upper center", bbox_to_anchor=bbox_tuple)
        axs[1].legend(ncols=legend_ncols, loc="upper center", bbox_to_anchor=bbox_tuple)
        fig.suptitle(suptitle)
        fig.set_tight_layout(True)

        # Optionally save to file
        out = _cfg_get(plot_cfg, "output_file", default=None)
        if out:
            fig.savefig(out)

        return fig, axs

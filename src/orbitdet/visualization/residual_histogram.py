import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from scipy import stats
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


def _principal_angle_rad(angle_rad: np.ndarray) -> np.ndarray:
    return np.remainder(angle_rad + np.pi, 2.0 * np.pi) - np.pi


def _shapiro_label(statistic: float, p_value: float) -> str:
    return f"Shapiro-Wilk W={statistic:.4f}, p={p_value:.3e}"


def _make_hover_formatter(hover_x_label: str, hover_y_label: str):
    def _format(x, y):
        return f"{hover_x_label}: {x:.3e}, {hover_y_label}: {y:.3e}"

    return _format


class ResidualHistogram(Plot):
    """Plot histograms of RA and DEC residuals with overlaid normal bell curves."""

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

        """Plot histograms of RA and DEC residuals with overlaid normal bell curves.

        Each subplot shows the residual distribution for one observatory group, with
        a fitted normal PDF and the Shapiro-Wilk normality test statistic annotated.
        """
        if observation_parsers is None:
            observation_sets: list[obs.SingleObservationSet] = (
                observation_collection.get_single_observation_sets()
            )
        else:
            observation_sets: list[obs.SingleObservationSet] = (
                observation_collection.get_single_observation_sets(observation_parsers)
            )

        # Load plotting configuration
        plot_cfg = _cfg_get(cfg, "residual_histogram", default=None)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=8.27 * 2)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=8.27 * 2 / 2)

        fig, axs = plt.subplots(
            2,
            1,
            figsize=(fig_w, fig_h),
            sharex=False,
        )
        cmap = _cfg_get(plot_cfg, "styling", "cmap", default="tab10")
        colors = plt.get_cmap(cmap)
        n_bins = int(_cfg_get(plot_cfg, "styling", "n_bins", default=30))
        bell_curve_color = _cfg_get(plot_cfg, "styling", "bell_curve_color", default="red")
        bell_curve_lw = float(_cfg_get(plot_cfg, "styling", "bell_curve_lw", default=1.5))

        for set_index, obs_set in enumerate(observation_sets):
            observatory_code = obs_set.link_definition.link_ends[links.receiver].reference_point
            if observatory_code == "":
                observatory_name = obs_set.link_definition.link_ends[links.receiver].body_name
                info = {"code": observatory_code}
                info["name"] = observatory_name
                info["region"] = "Spacecraft"
            else:
                info = get_observatory_info(cfg, observatory_code)
            target_name = obs_set.link_definition.link_ends[links.transmitter].body_name
            color = colors(set_index % colors.N)

            residuals = np.array(obs_set.residuals)
            # n x 2 array of RA and DEC residuals in radians

            # RA residuals are circular; fold them to the principal interval before converting.
            ra_residuals_arcsec = _rad_to_arcsec(_principal_angle_rad(residuals[:, 0]))
            dec_residuals_arcsec = _rad_to_arcsec(residuals[:, 1])

            finite_ra = ra_residuals_arcsec[np.isfinite(ra_residuals_arcsec)]
            finite_dec = dec_residuals_arcsec[np.isfinite(dec_residuals_arcsec)]

            # --- RA histogram ---
            if finite_ra.size > 0:
                n_ra, bins_ra, patches_ra = axs[0].hist(
                    finite_ra,
                    bins=n_bins,
                    density=True,
                    alpha=0.5,
                    color=color,
                    label=f"{info['name']} - {info['region']}",
                )
                # Overlay normal bell curve fitted to the data
                mu_ra, sigma_ra = np.mean(finite_ra), np.std(finite_ra, ddof=1)
                x_ra = np.linspace(bins_ra[0], bins_ra[-1], 300)
                axs[0].plot(
                    x_ra,
                    stats.norm.pdf(x_ra, mu_ra, sigma_ra),
                    color=bell_curve_color,
                    linewidth=bell_curve_lw,
                )
                # Shapiro-Wilk test
                if finite_ra.size >= 3:
                    w_ra, p_ra = stats.shapiro(finite_ra)
                    shapiro_label_ra = _shapiro_label(w_ra, p_ra)
                    # Annotate on the plot
                    axs[0].annotate(
                        shapiro_label_ra,
                        xy=(0.98, 0.95 - set_index * 0.08),
                        xycoords="axes fraction",
                        fontsize=7,
                        ha="right",
                        va="top",
                        color=color,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, edgecolor=color),
                    )

            # --- DEC histogram ---
            if finite_dec.size > 0:
                n_dec, bins_dec, patches_dec = axs[1].hist(
                    finite_dec,
                    bins=n_bins,
                    density=True,
                    alpha=0.5,
                    color=color,
                    label=f"{info['name']} - {info['region']}",
                )
                mu_dec, sigma_dec = np.mean(finite_dec), np.std(finite_dec, ddof=1)
                x_dec = np.linspace(bins_dec[0], bins_dec[-1], 300)
                axs[1].plot(
                    x_dec,
                    stats.norm.pdf(x_dec, mu_dec, sigma_dec),
                    color=bell_curve_color,
                    linewidth=bell_curve_lw,
                )
                if finite_dec.size >= 3:
                    w_dec, p_dec = stats.shapiro(finite_dec)
                    shapiro_label_dec = _shapiro_label(w_dec, p_dec)
                    axs[1].annotate(
                        shapiro_label_dec,
                        xy=(0.98, 0.95 - set_index * 0.08),
                        xycoords="axes fraction",
                        fontsize=7,
                        ha="right",
                        va="top",
                        color=color,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, edgecolor=color),
                    )

        # Titles and labels (configurable)
        title_ra = _cfg_get(plot_cfg, "titles", "ra", default="Right Ascension Residuals")
        title_dec = _cfg_get(plot_cfg, "titles", "dec", default="Declination Residuals")
        suptitle = _cfg_get(
            plot_cfg, "titles", "suptitle", default=f"Residual Histograms for {target_name}"
        )
        try:
            if isinstance(suptitle, str):
                suptitle = suptitle.format(target_name=target_name)
        except Exception:
            pass

        axs[0].set_title(title_ra)
        axs[1].set_title(title_dec)
        x_label = _cfg_get(plot_cfg, "axes", "x_label", default="Residual [arcsec]")
        y_label = _cfg_get(plot_cfg, "axes", "y_label", default="Probability Density")
        axs[0].set_xlabel(x_label)
        axs[0].set_ylabel(y_label)
        axs[1].set_xlabel(x_label)
        axs[1].set_ylabel(y_label)

        # Hover formatter
        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default="Residual [arcsec]")
        hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default="Probability Density")
        fmt = _make_hover_formatter(hover_x_label, hover_y_label)
        axs[0].format_coord = fmt
        axs[1].format_coord = fmt

        # Legend placement (configurable)
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


class ResidualScanHistogram(Plot):
    """Plot sideways histograms of Gaia along-scan (AL) and across-scan (AC) residuals.

    Extends the scan-frame rotation concept from :class:`ResidualsScan`: the
    RA/Dec residuals are rotated into the Gaia scan frame using the archived
    ``position_angle_scan``, then the AL and AC components are shown as
    horizontal histograms (``orientation="horizontal"``).  Each histogram is
    fitted with a normal curve, error bars are drawn on the bin counts, and the
    sample mean is marked with a vertical line.
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

        plot_cfg = _cfg_get(cfg, "residual_histogram", default=None)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=8.27 * 2)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=8.27 * 2 / 2)
        unit = _cfg_get(plot_cfg, "axes", "unit", default="arcsec")

        fig, axs = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=False)
        cmap = _cfg_get(plot_cfg, "styling", "cmap", default="tab10")
        colors = plt.get_cmap(cmap)
        n_bins = int(_cfg_get(plot_cfg, "styling", "n_bins", default=30))
        fit_color = _cfg_get(plot_cfg, "styling", "fit_color", default="red")
        fit_lw = float(_cfg_get(plot_cfg, "styling", "fit_lw", default=1.5))
        mean_color = _cfg_get(plot_cfg, "styling", "mean_color", default="black")
        mean_ls = _cfg_get(plot_cfg, "styling", "mean_ls", default="--")
        error_color = _cfg_get(plot_cfg, "styling", "error_color", default="black")
        error_lw = float(_cfg_get(plot_cfg, "styling", "error_lw", default=1.0))

        al_labels = []
        ac_labels = []
        for set_index, obs_set in enumerate(observation_sets):
            observatory_code = obs_set.link_definition.link_ends[links.receiver].reference_point
            if observatory_code == "":
                observatory_name = obs_set.link_definition.link_ends[links.receiver].body_name
                info = {"code": observatory_code}
                info["name"] = observatory_name
                info["region"] = "Spacecraft"
            else:
                info = get_observatory_info(cfg, observatory_code)
            target_name = obs_set.link_definition.link_ends[links.transmitter].body_name
            color = colors(set_index % colors.N)

            obs_times_sec_j2000 = np.array(
                [epoch.to_float() for epoch in obs_set.observation_times]
            )
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
            al_in_unit = _rad_to_arcsec(east * np.sin(psi) + north * np.cos(psi))
            ac_in_unit = _rad_to_arcsec(east * np.cos(psi) - north * np.sin(psi))

            finite_al = al_in_unit[np.isfinite(al_in_unit)]
            finite_ac = ac_in_unit[np.isfinite(ac_in_unit)]

            # --- Along-scan sideways histogram ---
            if finite_al.size > 0:
                n_al, bins_al, patches_al = axs[0].hist(
                    finite_al,
                    bins=n_bins,
                    orientation="horizontal",
                    density=True,
                    alpha=0.5,
                    color=color,
                    label=f"{info['name']} - {info['region']}",
                )
                # Error bars on the density bin values (Poisson: sqrt(count)).
                # For a density histogram, count = density * N * bin_width and the
                # density error = sqrt(count) / (N * bin_width).
                bin_widths_al = np.diff(bins_al)
                counts_al = n_al * finite_al.size * bin_widths_al
                density_err_al = np.sqrt(counts_al) / (finite_al.size * bin_widths_al)
                bin_centers_al = 0.5 * (bins_al[:-1] + bins_al[1:])
                axs[0].errorbar(
                    n_al,
                    bin_centers_al,
                    xerr=density_err_al,
                    fmt="none",
                    ecolor=error_color,
                    elinewidth=error_lw,
                    capsize=2,
                )
                # Fitted normal curve.
                mu_al, sigma_al = np.mean(finite_al), np.std(finite_al, ddof=1)
                y_al = np.linspace(bins_al[0], bins_al[-1], 300)
                axs[0].plot(
                    stats.norm.pdf(y_al, mu_al, sigma_al),
                    y_al,
                    color=fit_color,
                    linewidth=fit_lw,
                    label=f"Fit: μ={mu_al:.3e}, σ={sigma_al:.3e}",
                )
                # Sample mean line.
                axs[0].axhline(
                    mu_al,
                    color=mean_color,
                    linestyle=mean_ls,
                    linewidth=fit_lw,
                    label=f"Mean: {mu_al:.3e} {unit}",
                )
                al_labels.append(
                    f"{info['name']} - {info['region']} - μ={mu_al:.3e}, σ={sigma_al:.3e}"
                )

            # --- Across-scan sideways histogram ---
            if finite_ac.size > 0:
                n_ac, bins_ac, patches_ac = axs[1].hist(
                    finite_ac,
                    bins=n_bins,
                    orientation="horizontal",
                    density=True,
                    alpha=0.5,
                    color=color,
                    label=f"{info['name']} - {info['region']}",
                )
                bin_widths_ac = np.diff(bins_ac)
                counts_ac = n_ac * finite_ac.size * bin_widths_ac
                density_err_ac = np.sqrt(counts_ac) / (finite_ac.size * bin_widths_ac)
                bin_centers_ac = 0.5 * (bins_ac[:-1] + bins_ac[1:])
                axs[1].errorbar(
                    n_ac,
                    bin_centers_ac,
                    xerr=density_err_ac,
                    fmt="none",
                    ecolor=error_color,
                    elinewidth=error_lw,
                    capsize=2,
                )
                mu_ac, sigma_ac = np.mean(finite_ac), np.std(finite_ac, ddof=1)
                y_ac = np.linspace(bins_ac[0], bins_ac[-1], 300)
                axs[1].plot(
                    stats.norm.pdf(y_ac, mu_ac, sigma_ac),
                    y_ac,
                    color=fit_color,
                    linewidth=fit_lw,
                    label=f"Fit: μ={mu_ac:.3e}, σ={sigma_ac:.3e}",
                )
                axs[1].axhline(
                    mu_ac,
                    color=mean_color,
                    linestyle=mean_ls,
                    linewidth=fit_lw,
                    label=f"Mean: {mu_ac:.3e} {unit}",
                )
                ac_labels.append(
                    f"{info['name']} - {info['region']} - μ={mu_ac:.3e}, σ={sigma_ac:.3e}"
                )

        # Titles and labels (configurable)
        title_al = _cfg_get(plot_cfg, "titles", "al", default="Along-scan")
        title_ac = _cfg_get(plot_cfg, "titles", "ac", default="Across-scan")
        suptitle = _cfg_get(
            plot_cfg,
            "titles",
            "suptitle_scan_hist",
            default=f"Pre-Fit Scan Residual Histograms for {target_name}",
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
        x_label = _cfg_get(plot_cfg, "axes", "hist_x_label", default="Probability Density")
        axs[0].set_xlabel(x_label)
        axs[1].set_xlabel(x_label)

        # Hover formatter
        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default="Probability Density")
        hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default=f"Residual [{unit}]")
        fmt = _make_hover_formatter(hover_x_label, hover_y_label)
        axs[0].format_coord = fmt
        axs[1].format_coord = fmt

        # Legend placement (configurable)
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

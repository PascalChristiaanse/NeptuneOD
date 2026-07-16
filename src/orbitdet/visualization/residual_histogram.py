import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from scipy import stats
from tudatpy.estimation import observations as obs
from tudatpy.estimation.observable_models_setup import links
from tudatpy.estimation.observations import observations_processing as obs_proc

from orbitdet.observations import get_observatory_info


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


def plot_residual_histogram(
    cfg: DictConfig,
    observation_collection: obs.ObservationCollection,
    observation_parsers: list[obs_proc.ObservationParserType] = None,
) -> tuple[plt.Figure, np.ndarray]:
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

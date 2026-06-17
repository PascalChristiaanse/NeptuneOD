import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def _compute_qq_data(values_arcsec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute theoretical and sample quantiles for a Q-Q plot against a normal distribution."""
    finite_values = values_arcsec[np.isfinite(values_arcsec)]
    if finite_values.size == 0:
        return np.array([]), np.array([])

    standardized = (finite_values - np.mean(finite_values)) / np.std(finite_values, ddof=1)
    standardized = np.sort(standardized)
    n = standardized.size
    theoretical_quantiles = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    return theoretical_quantiles, standardized


def _make_hover_formatter(hover_x_label: str, hover_y_label: str):
    def _format(x, y):
        return f"{hover_x_label}: {x:.3e}, {hover_y_label}: {y:.3e}"

    return _format


def plot_residual_qq(
    cfg: DictConfig,
    observation_collection: obs.ObservationCollection,
    observation_parsers: list[obs_proc.ObservationParserType] = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot Q-Q plots of pre-fit and post-fit residuals against a normal distribution."""
    if observation_parsers is None:
        observation_sets: list[obs.SingleObservationSet] = (
            observation_collection.get_single_observation_sets()
        )
    else:
        observation_sets: list[obs.SingleObservationSet] = (
            observation_collection.get_single_observation_sets(observation_parsers)
        )

    # Load plotting configuration
    plot_cfg = _cfg_get(cfg, "residual_qq", default=None)
    fig_w = _cfg_get(plot_cfg, "figure", "width", default=8.27 * 2)
    fig_h = _cfg_get(plot_cfg, "figure", "height", default=8.27 * 2 / 2)

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

        residuals = np.array(obs_set.residuals)
        # n x 2 array of RA and DEC residuals in radians

        # RA residuals are circular; fold them to the principal interval before converting.
        ra_residuals_arcsec = _rad_to_arcsec(_principal_angle_rad(residuals[:, 0]))
        dec_residuals_arcsec = _rad_to_arcsec(residuals[:, 1])

        ra_theoretical, ra_sample = _compute_qq_data(ra_residuals_arcsec)
        dec_theoretical, dec_sample = _compute_qq_data(dec_residuals_arcsec)

        # RA Q-Q
        if ra_theoretical.size > 0:
            axs[0].scatter(
                ra_theoretical,
                ra_sample,
                marker=marker,
                s=marker_size,
                label=f"{info['name']} - {info['region']}",
                color=color,
                alpha=0.5,
            )

        # DEC Q-Q
        if dec_theoretical.size > 0:
            axs[1].scatter(
                dec_theoretical,
                dec_sample,
                marker=marker,
                s=marker_size,
                label=f"{info['name']} - {info['region']}",
                color=color,
                alpha=0.5,
            )

    # Titles and labels (configurable)
    title_ra = _cfg_get(plot_cfg, "titles", "ra", default="Right Ascension Q-Q")
    title_dec = _cfg_get(plot_cfg, "titles", "dec", default="Declination Q-Q")
    suptitle = _cfg_get(
        plot_cfg, "titles", "suptitle", default=f"Residual Q-Q Plot for {target_name}"
    )
    # Allow templates like "Residual Q-Q Plot for {target_name}" in config
    try:
        if isinstance(suptitle, str):
            suptitle = suptitle.format(target_name=target_name)
    except Exception:
        # leave suptitle unchanged if formatting fails
        pass

    axs[0].set_title(title_ra)
    axs[1].set_title(title_dec)
    y_label = _cfg_get(plot_cfg, "axes", "y_label", default="Sample Quantiles [std]")
    axs[0].set_ylabel(y_label)
    axs[1].set_ylabel(y_label)
    x_label = _cfg_get(plot_cfg, "axes", "x_label", default="Theoretical Quantiles [std]")
    axs[0].set_xlabel(x_label)
    axs[1].set_xlabel(x_label)

    # Reference line y=x
    ref_line_style = _cfg_get(plot_cfg, "styling", "ref_line", default={"ls": "--", "color": "k", "lw": 1.0})
    ref_ls = ref_line_style.get("ls", "--") if isinstance(ref_line_style, dict) else "--"
    ref_color = ref_line_style.get("color", "k") if isinstance(ref_line_style, dict) else "k"
    ref_lw = float(ref_line_style.get("lw", 1.0)) if isinstance(ref_line_style, dict) else 1.0

    for ax in axs:
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, ls=ref_ls, color=ref_color, lw=ref_lw, label="Normal reference")

    # Hover formatter
    hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default="Theoretical Quantiles [std]")
    hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default="Sample Quantiles [std]")
    fmt = _make_hover_formatter(hover_x_label, hover_y_label)
    axs[0].format_coord = fmt
    axs[1].format_coord = fmt

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

    # axs[0].legend(ncols=legend_ncols, loc="upper center", bbox_to_anchor=bbox_tuple)
    # axs[1].legend(ncols=legend_ncols, loc="upper center", bbox_to_anchor=bbox_tuple)
    fig.suptitle(suptitle)
    fig.set_tight_layout(True)

    # Optionally save to file
    out = _cfg_get(plot_cfg, "output_file", default=None)
    if out:
        fig.savefig(out)

    return fig, axs

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
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


def _seconds_since_j2000_to_datetimes(seconds_since_j2000: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(
        seconds_since_j2000,
        unit="s",
        origin=pd.Timestamp("2000-01-01T12:00:00"),
    )


def _make_hover_formatter(hover_x_label: str, hover_y_label: str):
    def _format(x, y):
        return f"{hover_x_label}: {x:.3e}, {hover_y_label}: {y:.3e}"

    return _format


def plot_residual_scatter(
    cfg: DictConfig,
    observation_collection: obs.ObservationCollection,
    observation_parsers: list[obs_proc.ObservationParserType] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot RA vs DEC residuals as a scatter plot with time encoded as color.

    Each observatory group uses a distinct marker shape. Points are colored
    on a rainbow gradient corresponding to observation time so that temporal
    trends are visible even though time is not an explicit axis.
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
    plot_cfg = _cfg_get(cfg, "residual_scatter", default=None)
    fig_w = _cfg_get(plot_cfg, "figure", "width", default=8.27)
    fig_h = _cfg_get(plot_cfg, "figure", "height", default=8.27)

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

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
    ]  # cycle through marker types for different observatories

    # First pass: collect all timestamps to determine global time range for color mapping
    all_times_sec = []
    for obs_set in observation_sets:
        obs_times_sec_j2000 = np.array([epoch.to_float() for epoch in obs_set.observation_times])
        all_times_sec.append(obs_times_sec_j2000)

    all_times_sec = np.concatenate(all_times_sec)
    global_t_min = float(np.min(all_times_sec))
    global_t_max = float(np.max(all_times_sec))
    time_range = global_t_max - global_t_min if global_t_max > global_t_min else 1.0

    cmap_name = _cfg_get(plot_cfg, "styling", "cmap", default="rainbow")
    cmap = plt.get_cmap(cmap_name)
    marker_size = _cfg_get(plot_cfg, "styling", "marker_size", default=30)

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
        marker = marker_types[set_index % len(marker_types)]

        obs_times_sec_j2000 = np.array([epoch.to_float() for epoch in obs_set.observation_times])
        residuals = np.array(obs_set.residuals)
        # n x 2 array of RA and DEC residuals in radians

        # RA residuals are circular; fold them to the principal interval before converting.
        ra_residuals_arcsec = _rad_to_arcsec(_principal_angle_rad(residuals[:, 0]))
        dec_residuals_arcsec = _rad_to_arcsec(residuals[:, 1])

        # Normalize time to [0, 1] for color mapping
        time_normalized = (obs_times_sec_j2000 - global_t_min) / time_range
        colors = cmap(time_normalized)

        ax.scatter(
            ra_residuals_arcsec,
            dec_residuals_arcsec,
            marker=marker,
            s=marker_size,
            c=colors,
            label=f"{info['name']} - {info['region']}",
            alpha=0.5,
            edgecolors="none",
        )

    # Titles and labels (configurable)
    title = _cfg_get(plot_cfg, "titles", "title", default=f"RA vs DEC Residuals for {target_name}")
    # Allow templates like "RA vs DEC Residuals for {target_name}" in config
    try:
        if isinstance(title, str):
            title = title.format(target_name=target_name)
    except Exception:
        pass

    x_label = _cfg_get(plot_cfg, "axes", "x_label", default="RA Residual [arcsec]")
    y_label = _cfg_get(plot_cfg, "axes", "y_label", default="DEC Residual [arcsec]")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Add a colorbar for time
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=global_t_min, vmax=global_t_max),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(_cfg_get(plot_cfg, "axes", "colorbar_label", default="Epoch"))

    # Format colorbar ticks as dates
    def _format_colorbar_tick(value, _position):
        try:
            dt = mdates.num2date(mdates.epoch2num(float(value)))
            return dt.strftime("%Y-%m")
        except Exception:
            return f"{value:.0f}"

    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(_format_colorbar_tick))

    # Hover formatter
    hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default="RA Residual [arcsec]")
    hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default="DEC Residual [arcsec]")
    fmt = _make_hover_formatter(hover_x_label, hover_y_label)
    ax.format_coord = fmt

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

    ax.legend(ncols=legend_ncols, loc="upper center", bbox_to_anchor=bbox_tuple)
    fig.set_tight_layout(True)

    # Optionally save to file
    out = _cfg_get(plot_cfg, "output_file", default=None)
    if out:
        fig.savefig(out)

    return fig, ax

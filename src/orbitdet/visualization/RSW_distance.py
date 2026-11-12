import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tudatpy.astro import frame_conversion as fc
from tudatpy.dynamics import propagation as prop
from tudatpy.dynamics import propagation_setup as prop_setup

from orbitdet.utility import create_dependent_variable_dictionary
from orbitdet.visualization.base import _make_hover_formatter
from orbitdet.visualization.palettes import palette
from orbitdet.visualization.simulation_base import SimulationPlot, _cfg_get


def _seconds_since_j2000_to_datetimes(seconds_since_j2000):
    return pd.to_datetime(
        seconds_since_j2000,
        unit="s",
        origin=pd.Timestamp("2000-01-01T12:00:00"),
    )


def _configure_datetime_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


#: Supported display units for the RSW components, as ``metres per unit``.
#:
#: The decomposition is always computed in metres (TudatPy's SI base unit); the
#: unit only affects what is drawn and labelled.
LENGTH_UNITS: dict[str, float] = {
    "m": 1.0,
    "km": 1e3,
}

#: Display unit used when a figure config names none.
DEFAULT_LENGTH_UNIT = "m"


def length_unit(plot_cfg) -> str:
    """Return the display unit named by ``axes.unit``, defaulting to metres.

    Parameters
    ----------
    plot_cfg : DictConfig
        The figure's own configuration block.

    Returns
    -------
    str
        One of the keys of :data:`LENGTH_UNITS`.

    Raises
    ------
    ValueError
        If the configured unit is not supported.
    """
    unit = str(_cfg_get(plot_cfg, "axes", "unit", default=DEFAULT_LENGTH_UNIT))
    if unit not in LENGTH_UNITS:
        known = ", ".join(sorted(LENGTH_UNITS))
        raise ValueError(f"Unknown length unit '{unit}'. Known: {known}.")
    return unit


def to_length_unit(values: np.ndarray, unit: str) -> np.ndarray:
    """Convert metres to the requested display unit."""
    return np.asarray(values) / LENGTH_UNITS[unit]


#: Colour collection used when a figure config names none.
#:
#: Black by default: the RSW panels are read as a set, and a single colour keeps
#: the eye on the shape of the curve rather than on the palette.  Set
#: ``styling.palette`` (or ``styling.colors``) in the figure config to override.
DEFAULT_COLORS: list[str] = ["black"]


def color_collection(plot_cfg) -> list[str]:
    """Return the line colours for this figure.

    Two ways to configure it, in precedence order::

        styling:
          palette: "okabe_ito"           # a named palette, cycled in order
          colors: ["#0072B2", "#D55E00"] # explicit colours, cycled in order

    ``palette`` is the preferred form: it keeps figures that should look alike
    consistent, and makes "is this colourblind-safe?" a question about one
    table.  ``colors`` remains for one-off overrides.

    A bare string is accepted for either key.

    Parameters
    ----------
    plot_cfg : DictConfig
        The figure's own configuration block.

    Returns
    -------
    list[str]
        The colours to cycle through, in order.

    Raises
    ------
    ValueError
        If the named palette is unknown, or ``colors`` is empty.
    """
    name = _cfg_get(plot_cfg, "styling", "palette", default=None)
    if name is not None:
        return palette(str(name))

    colors = _cfg_get(plot_cfg, "styling", "colors", default=None)
    if colors is None:
        return list(DEFAULT_COLORS)

    if isinstance(colors, str):
        colors = [colors]

    colors = [str(color) for color in colors]
    if not colors:
        raise ValueError("styling.colors must name at least one colour.")
    return colors


def color_for(colors: list[str], index: int) -> str:
    """Return the colour for series *index*, cycling through *colors*."""
    return colors[index % len(colors)]


#: Panel names for the 2x2 RSW grid, in ``axes`` order.
#:
#: ``axes`` is flattened row-major, so index 0 is top-left, 1 top-right,
#: 2 bottom-left and 3 bottom-right.
PANEL_NAMES: dict[str, int] = {
    "top_left": 0,
    "top_right": 1,
    "bottom_left": 2,
    "bottom_right": 3,
}

#: Corner names accepted by ``legend.corner``, mapped to matplotlib's ``loc``.
LEGEND_CORNERS: dict[str, str] = {
    "top_left": "upper left",
    "top_right": "upper right",
    "bottom_left": "lower left",
    "bottom_right": "lower right",
}

#: Where the legend goes when a figure config names no placement.
DEFAULT_LEGEND_PANEL = "top_right"
DEFAULT_LEGEND_CORNER = "top_right"


def legend_placement(plot_cfg) -> tuple[int, str]:
    """Resolve where the figure legend should be drawn.

    The legend is placed *inside* one of the four panels rather than below the
    figure, so it cannot collide with the x-axis tick labels.

    Configuration::

        legend:
          panel: "top_right"     # top_left | top_right | bottom_left | bottom_right
          corner: "top_right"    # corner of that panel to anchor to

    Parameters
    ----------
    plot_cfg : DictConfig
        The figure's own configuration block.

    Returns
    -------
    tuple[int, str]
        ``(panel_index, matplotlib_loc)`` — the flattened axes index to draw the
        legend on, and the ``loc`` string for :meth:`matplotlib.axes.Axes.legend`.

    Raises
    ------
    ValueError
        If the configured panel or corner is not recognised.
    """
    panel = str(_cfg_get(plot_cfg, "legend", "panel", default=DEFAULT_LEGEND_PANEL))
    corner = str(_cfg_get(plot_cfg, "legend", "corner", default=DEFAULT_LEGEND_CORNER))

    if panel not in PANEL_NAMES:
        known = ", ".join(PANEL_NAMES)
        raise ValueError(f"Unknown legend panel '{panel}'. Known: {known}.")
    if corner not in LEGEND_CORNERS:
        known = ", ".join(LEGEND_CORNERS)
        raise ValueError(f"Unknown legend corner '{corner}'. Known: {known}.")

    return PANEL_NAMES[panel], LEGEND_CORNERS[corner]


def add_panel_legend(axes, plot_cfg, n_entries: int) -> None:
    """Draw one legend inside the configured panel.

    A figure-level legend below the axes is what clipped the x-axis labels; an
    in-panel legend cannot, because it lives inside the axes box.

    The legend sits on a semi-transparent box by default, so the curves behind
    it stay partly visible instead of being hidden by an opaque patch.

    Parameters
    ----------
    axes : array-like
        The flattened axes of the 2x2 grid.
    plot_cfg : DictConfig
        The figure's own configuration block.
    n_entries : int
        Number of legend entries, used to pick a sensible column count.
    """
    panel_index, loc = legend_placement(plot_cfg)

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        return

    # Keep the legend compact: one column for a couple of series, two beyond.
    ncols = int(_cfg_get(plot_cfg, "legend", "ncols", default=1 if n_entries <= 3 else 2))

    axes[panel_index].legend(
        handles,
        labels,
        loc=loc,
        ncols=ncols,
        frameon=bool(_cfg_get(plot_cfg, "legend", "frameon", default=True)),
        framealpha=float(_cfg_get(plot_cfg, "legend", "framealpha", default=0.6)),
        fontsize=_cfg_get(plot_cfg, "legend", "fontsize", default=None),
    )


def compute_rsw_components(
    sim_results: prop.SimulationResults,
    position_dependent_variable: (
        prop_setup.dependent_variable.SingleDependentVariableSaveSettings
    ),
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Decompose a relative-position dependent variable into RSW components.

    Shared by :class:`RSWDistance` and :class:`RSWDistanceComparison` so the
    decomposition is written once.  The RSW frame is built from the propagated
    body's inertial state at each epoch.

    Parameters
    ----------
    sim_results : prop.SimulationResults
        The propagation result carrying both the state history and the
        dependent-variable history.
    position_dependent_variable : SingleDependentVariableSaveSettings
        The relative-position dependent variable to decompose.

    Returns
    -------
    tuple
        ``(times, rsw_components, rsw_norm)`` where ``times`` is a
        :class:`pandas.DatetimeIndex`, ``rsw_components`` has shape
        ``(n_epochs, 3)`` and ``rsw_norm`` has shape ``(n_epochs,)``.
    """
    dep_var_dict = create_dependent_variable_dictionary(sim_results)

    try:
        dep_var_dict[position_dependent_variable]
    except KeyError as e:
        raise ValueError(
            f"Dependent variable not found in results object. "
            f"Dependent variable: {position_dependent_variable}."
        ) from e

    pos_value_dict = dep_var_dict[position_dependent_variable]

    epochs = list(dep_var_dict.time_history)
    n_epochs = len(epochs)
    times = _seconds_since_j2000_to_datetimes(np.asarray(epochs))

    state_history = sim_results.state_history
    if len(state_history) != n_epochs:
        raise ValueError(
            f"Mismatch between state history ({len(state_history)} entries) and "
            f"dependent variable history ({n_epochs} entries)."
        )

    rsw_components = np.zeros((n_epochs, 3))
    skipped_epochs = 0
    for i, epoch in enumerate(epochs):
        inertial_state = np.asarray(state_history[epoch])
        # A degenerate state (position parallel to velocity) has no RSW frame.
        radius = inertial_state[:3]
        velocity = inertial_state[3:]
        cross_norm = np.linalg.norm(np.cross(radius, velocity))
        if cross_norm == 0.0:
            skipped_epochs += 1
            rsw_components[i, :] = np.nan
            continue
        rot = fc.inertial_to_rsw_rotation_matrix(inertial_state)
        rel_pos = np.asarray(pos_value_dict[epoch]).flatten()
        rsw_components[i, :] = rot @ rel_pos

    if skipped_epochs > 0:
        warnings.warn(
            f"Skipped {skipped_epochs} / {n_epochs} epoch(s) where "
            f"position and velocity are parallel (cannot form RSW frame)."
        )

    rsw_norm = np.linalg.norm(rsw_components, axis=1)
    return times, rsw_components, rsw_norm


class RSWDistance(SimulationPlot):
    """Decompose the relative position vector between two bodies into its Radial (R),
    Transverse (S), and Cross-track (W) components in the RSW frame.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (``conf/figures/<name>.yaml``).
        ``central_body`` is read from here (default ``"Neptune"``).
    result : prop.SimulationResults
        The propagation result to plot.
    position_dependent_variable : SingleDependentVariableSaveSettings
        The relative-position dependent variable to decompose.
    """

    def __init__(
        self,
        cfg: DictConfig,
        result: prop.SimulationResults,
        position_dependent_variable: (
            prop_setup.dependent_variable.SingleDependentVariableSaveSettings
        ),
    ):
        super().__init__(cfg, result)
        self.position_dependent_variable = position_dependent_variable

    def _make_figure(self):
        plot_cfg = self.cfg
        position_dependent_variable = self.position_dependent_variable
        central_body = _cfg_get(plot_cfg, "central_body", default="Neptune")

        times, rsw_components, rsw_norm = compute_rsw_components(
            self.result, position_dependent_variable
        )

        fig_w, fig_h = self._figure_size()

        titles = {
            "r": _cfg_get(plot_cfg, "titles", "r", default="Radial (R)"),
            "s": _cfg_get(plot_cfg, "titles", "s", default="Transverse (S)"),
            "w": _cfg_get(plot_cfg, "titles", "w", default="Cross-track (W)"),
            "norm": _cfg_get(plot_cfg, "titles", "norm", default="L2 Norm of RSW"),
        }
        suptitle = _cfg_get(
            plot_cfg,
            "titles",
            "suptitle",
            default=f"RSW Distance for {position_dependent_variable.associated_body}",
        )
        central = _cfg_get(plot_cfg, "central_body", default=central_body)
        try:
            if isinstance(suptitle, str):
                suptitle = suptitle.format(
                    body1=position_dependent_variable.associated_body,
                    body2=position_dependent_variable.secondary_body,
                    central_body=central,
                )
        except Exception:
            pass

        x_label = _cfg_get(plot_cfg, "axes", "x_label", default="Epoch")
        unit = length_unit(plot_cfg)
        y_label = _cfg_get(plot_cfg, "axes", "y_label", default=f"Distance [{unit}]")
        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default=x_label)
        hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default=y_label)

        fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), sharex=True)
        axes = np.asarray(axes).reshape(-1)

        colors = color_collection(plot_cfg)
        series = [
            (to_length_unit(rsw_components[:, 0], unit), titles["r"]),
            (to_length_unit(rsw_components[:, 1], unit), titles["s"]),
            (to_length_unit(rsw_components[:, 2], unit), titles["w"]),
            (to_length_unit(rsw_norm, unit), titles["norm"]),
        ]
        for index, (ax, (values, title)) in enumerate(zip(axes, series)):
            ax.plot(times, values, color=color_for(colors, index))
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.format_coord = _make_hover_formatter(hover_x_label, hover_y_label)

        for ax in axes[::2]:
            ax.set_ylabel("test")

        for ax in axes[2:]:
            _configure_datetime_axis(ax)
            ax.set_xlabel(x_label)

        fig.suptitle(
            suptitle,
            fontsize=14,
        )
        fig.align_ylabels(axes)
        fig.set_tight_layout(True)

        return fig, axes


class RSWDistanceComparison(SimulationPlot):
    """Overlay the RSW decomposition of several propagation results.

    Each result is one line per panel, labelled by its entry in
    ``labels``.  This is how two dynamical variants (e.g. two Neptune pole
    models) are compared on a single figure without a second plotter.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (``conf/figures/<name>.yaml``).
        ``central_body`` is read from here (default ``"Neptune"``).
    results : list[prop.SimulationResults]
        The propagation results to overlay, in plotting order.
    position_dependent_variable : SingleDependentVariableSaveSettings
        The relative-position dependent variable to decompose.  It must be
        present in every result, which is why the sweep runs the same config
        with only a value override.
    labels : list[str] | None
        Legend labels, one per result.  Defaults to ``Result 0``, ``Result 1``,
        ...
    """

    def __init__(
        self,
        cfg: DictConfig,
        results: list[prop.SimulationResults],
        position_dependent_variable: (
            prop_setup.dependent_variable.SingleDependentVariableSaveSettings
        ),
        labels: list[str] | None = None,
    ):
        if not results:
            raise ValueError("results must be a non-empty list of SimulationResults.")
        # The base class wants a single result; the first one supplies the
        # output name and the figure size.
        super().__init__(cfg, results[0])
        self.results = results
        self.position_dependent_variable = position_dependent_variable
        self.labels = labels or [f"Result {i}" for i in range(len(results))]

        if len(self.labels) != len(self.results):
            raise ValueError(
                f"labels has {len(self.labels)} entries but {len(self.results)} "
                "result(s) were given."
            )

    def _make_figure(self):
        plot_cfg = self.cfg
        position_dependent_variable = self.position_dependent_variable
        central_body = _cfg_get(plot_cfg, "central_body", default="Neptune")

        # Decompose every result once, up front.
        decomposed = [
            compute_rsw_components(result, position_dependent_variable) for result in self.results
        ]

        fig_w, fig_h = self._figure_size()

        titles = {
            "r": _cfg_get(plot_cfg, "titles", "r", default="Radial (R)"),
            "s": _cfg_get(plot_cfg, "titles", "s", default="Transverse (S)"),
            "w": _cfg_get(plot_cfg, "titles", "w", default="Cross-track (W)"),
            "norm": _cfg_get(plot_cfg, "titles", "norm", default="L2 Norm of RSW"),
        }
        suptitle = _cfg_get(
            plot_cfg,
            "titles",
            "suptitle",
            default=f"RSW Distance for {position_dependent_variable.associated_body}",
        )
        central = _cfg_get(plot_cfg, "central_body", default=central_body)
        try:
            if isinstance(suptitle, str):
                suptitle = suptitle.format(
                    body1=position_dependent_variable.associated_body,
                    body2=position_dependent_variable.secondary_body,
                    central_body=central,
                )
        except Exception:
            pass

        x_label = _cfg_get(plot_cfg, "axes", "x_label", default="Epoch")
        unit = length_unit(plot_cfg)
        y_label = _cfg_get(plot_cfg, "axes", "y_label", default=f"Distance [{unit}]")
        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default=x_label)
        hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default=y_label)

        fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), sharex=True)
        axes = np.asarray(axes).reshape(-1)

        colors = color_collection(plot_cfg)

        # Panel index -> (component column, title).  ``None`` selects the norm.
        panels = [
            (0, titles["r"]),
            (1, titles["s"]),
            (2, titles["w"]),
            (None, titles["norm"]),
        ]

        for ax, (column, title) in zip(axes, panels):
            for index, ((times, components, norm), label) in enumerate(
                zip(decomposed, self.labels)
            ):
                values = norm if column is None else components[:, column]
                ax.plot(
                    times,
                    to_length_unit(values, unit),
                    label=label,
                    color=color_for(colors, index),
                )
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.format_coord = _make_hover_formatter(hover_x_label, hover_y_label)

        for ax in axes[::2]:
            ax.set_ylabel(y_label)

        for ax in axes[2:]:
            _configure_datetime_axis(ax)
            ax.set_xlabel(x_label)

        # One legend for the whole figure, drawn inside a panel so it cannot
        # collide with the x-axis tick labels.
        add_panel_legend(axes, plot_cfg, len(self.labels))

        fig.suptitle(suptitle, fontsize=14)
        fig.align_ylabels(axes)
        fig.set_tight_layout(True)

        return fig, axes


class RSWDistanceDifference(SimulationPlot):
    """Plot the RSW difference between a reference result and comparison results.

    Where :class:`RSWDistanceComparison` overlays the absolute RSW components,
    this plots ``reference - comparison`` for each comparison result.  That is
    the figure that answers "how much do two dynamical variants disagree?",
    which the absolute overlay cannot show when the curves sit on top of each
    other.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (``conf/figures/<name>.yaml``).
        ``central_body`` is read from here (default ``"Neptune"``).
        config keys:
            - ``central_body``: the central body for the RSW frame (default ``"Neptune"``)
            - ``titles``: a mapping of panel names to titles (default ``"Radial (R)"``, ``"Transverse (S)"``, ``"Cross-track (W)"``, ``"L2 Norm of RSW"``)
            - ``axes``: a mapping of axis labels (default ``"Epoch"``, ``"Difference [unit]"``)
            - ``styling``: a mapping of styling options (default ``palette: "okabe_ito"``, ``colors: ["#0072B2", "#D55E00"]``)
            - ``legend``: a mapping of legend options (default ``panel: "top_right"``, ``corner: "top_right"``, ``ncols: 1``)
            - ``figure_size``: a mapping of figure size options (default ``width: 10``, ``height: 8``)
            - ``output``: a mapping of output options (default ``filename: "rsw_difference.pdf"``, ``dpi: 300``)
            - ``savefig``: a mapping of savefig options (default ``bbox_inches: "tight"``, ``pad_inches: 0.1``)
            - ``show``: a boolean indicating whether to show the figure (default ``False``)
            - ``close``: a boolean indicating whether to close the figure after saving (default ``True``)
    reference_result : prop.SimulationResults
        The result every comparison is subtracted from.
    comparison_results : list[prop.SimulationResults]
        The results to subtract from the reference, in plotting order.
    position_dependent_variable : SingleDependentVariableSaveSettings
        The relative-position dependent variable to decompose.  It must be
        present in every result.
    labels : list[str] | None
        Legend labels, one per comparison result.  Defaults to
        ``Comparison 0``, ``Comparison 1``, ...
    """

    def __init__(
        self,
        cfg: DictConfig,
        reference_result: prop.SimulationResults,
        comparison_results: list[prop.SimulationResults],
        position_dependent_variable: (
            prop_setup.dependent_variable.SingleDependentVariableSaveSettings
        ),
        labels: list[str] | None = None,
    ):
        if not comparison_results:
            raise ValueError("comparison_results must be a non-empty list of SimulationResults.")
        super().__init__(cfg, reference_result)
        self.reference_result = reference_result
        self.comparison_results = comparison_results
        self.position_dependent_variable = position_dependent_variable
        self.labels = labels or [f"Comparison {i}" for i in range(len(comparison_results))]

        if len(self.labels) != len(self.comparison_results):
            raise ValueError(
                f"labels has {len(self.labels)} entries but "
                f"{len(self.comparison_results)} comparison result(s) were given."
            )

    def _make_figure(self):
        plot_cfg = self.cfg
        position_dependent_variable = self.position_dependent_variable
        central_body = _cfg_get(plot_cfg, "central_body", default="Neptune")

        ref_times, ref_components, ref_norm = compute_rsw_components(
            self.reference_result, position_dependent_variable
        )

        # Decompose every comparison once, up front.
        decomposed = [
            compute_rsw_components(result, position_dependent_variable)
            for result in self.comparison_results
        ]

        fig_w, fig_h = self._figure_size()

        titles = {
            "r": _cfg_get(plot_cfg, "titles", "r", default="Radial (R)"),
            "s": _cfg_get(plot_cfg, "titles", "s", default="Transverse (S)"),
            "w": _cfg_get(plot_cfg, "titles", "w", default="Cross-track (W)"),
            "norm": _cfg_get(plot_cfg, "titles", "norm", default="L2 Norm of RSW"),
        }
        suptitle = _cfg_get(
            plot_cfg,
            "titles",
            "suptitle",
            default=f"RSW Difference for {position_dependent_variable.associated_body}",
        )
        central = _cfg_get(plot_cfg, "central_body", default=central_body)
        try:
            if isinstance(suptitle, str):
                suptitle = suptitle.format(
                    body1=position_dependent_variable.associated_body,
                    body2=position_dependent_variable.secondary_body,
                    central_body=central,
                )
        except Exception:
            pass

        x_label = _cfg_get(plot_cfg, "axes", "x_label", default="Epoch")
        unit = length_unit(plot_cfg)
        y_label = _cfg_get(plot_cfg, "axes", "y_label", default=f"Difference [{unit}]")
        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default=x_label)
        hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default=y_label)

        fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), sharex=True)
        axes = np.asarray(axes).reshape(-1)

        colors = color_collection(plot_cfg)

        # Panel index -> (component column, title).  ``None`` selects the norm.
        panels = [
            (0, titles["r"]),
            (1, titles["s"]),
            (2, titles["w"]),
            (None, titles["norm"]),
        ]

        for ax, (column, title) in zip(axes, panels):
            for index, ((times, components, norm), label) in enumerate(
                zip(decomposed, self.labels)
            ):
                if column is None:
                    values = ref_norm - norm
                else:
                    values = ref_components[:, column] - components[:, column]
                ax.plot(
                    times,
                    to_length_unit(values, unit),
                    label=label,
                    color=color_for(colors, index),
                )
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.format_coord = _make_hover_formatter(hover_x_label, hover_y_label)

        # Set only on the left panels
        for ax in axes[::2]:
            ax.set_ylabel(y_label)

        for ax in axes[2:]:
            _configure_datetime_axis(ax)
            ax.set_xlabel(x_label)

        # One legend for the whole figure, drawn inside a panel so it cannot
        # collide with the x-axis tick labels.
        add_panel_legend(axes, plot_cfg, len(self.labels))

        fig.suptitle(suptitle, fontsize=14)
        fig.align_ylabels(axes)
        fig.set_tight_layout(True)

        return fig, axes

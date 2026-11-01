"""Stage: comparison bar charts for a sweep.

One figure per metric, each bar being a sweep combination. This is the "which
combination won" view: the per-experiment reports say how one run went, these
say how they compare.

Design notes
------------
**One figure per metric, not one figure with everything.** The metrics differ by
orders of magnitude (a residual RMS versus a duration in seconds), and a shared
axis would flatten the informative one. Separate small figures also survive a
degenerate metric: if every covariance is NaN the ``covariance_is_finite`` chart
is still readable.

**Bars are sorted by the metric, not by combination id.** With 32 combinations
the ordering *is* the result -- an id-ordered chart makes the reader do the
sorting. The x tick labels carry the combination id so the ordering stays
traceable to ``sweep_manifest.json``.

**Colour encodes the varied parameters, not the value.** A single-hue ramp would
suggest the metric is ordered by the colour, which it is not. Instead the bars
are coloured by how many parameters the combination estimated, which is the
quantity this sweep actually varies -- so the chart shows *why* the metric
changes, not just that it does.

The figure is rendered with the project's ``Plot`` base class so it saves a PDF
and a pickled Figure like every other figure, and so it is picked up by the same
``report()`` mechanism.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from orbitdet.visualization.base import Plot

logger = logging.getLogger(__name__)

#: Metrics to chart, with their axis label and whether lower is better.
#: ``None`` for ``better`` means the metric has no direction (a count of
#: iterations, say), and the figure notes that instead of claiming a winner.
METRICS: dict[str, tuple[str, bool | None]] = {
    "residuals_rms": ("Residual RMS", True),
    "postfit_rms_best_iteration": ("Residual RMS at best iteration", True),
    "estimation_duration_s": ("Estimation duration [s]", True),
    "num_iterations": ("Least-squares iterations", None),
    "num_parameters": ("Estimated parameters", None),
}


def _load_table(csv_path: str | Path) -> list[dict]:
    """Read the telemetry CSV written by the telemetry stage."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"No sweep telemetry at {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: object) -> float:
    """Coerce a CSV cell to float, yielding NaN when it is not numeric.

    ``bool``-ish cells arrive as ``"True"``/``"False"``; anything unparseable
    becomes NaN so a chart still renders instead of the stage dying on one bad
    cell.
    """
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


class SweepComparison(Plot):
    """Bar chart comparing one metric across a sweep's combinations.

    Parameters
    ----------
    cfg:
        The ``figures`` config slice.
    telemetry:
        Path to ``sweep_telemetry.csv``.
    sweep:
        Sweep name, used for the title and the manifest lookup.
    metric:
        Column to chart. Must be a key of :data:`METRICS`.
    metrics_config:
        Optional per-metric configuration. Read from
        ``cfg.figures.sweep_comparison.metrics.<metric>`` when present, so a
        figure's title can be reworded without touching code.
    """

    def __init__(
        self,
        cfg: DictConfig,
        telemetry: str | Path,
        sweep: str,
        metric: str,
    ):
        super().__init__(cfg, name=f"sweep_comparison_{metric}")
        self.telemetry = Path(telemetry)
        self.sweep = sweep
        self.metric = metric

        if metric not in METRICS:
            raise ValueError(f"Unknown metric {metric!r}. Known: {sorted(METRICS)}")

        self.axis_label, self.lower_is_better = METRICS[metric]
        # Set by ``_colour_by_parameter_count`` when the parameter count varies.
        self._colour_range: tuple[float, float] | None = None

    def _make_figure(self):
        from matplotlib import pyplot as plt

        rows = _load_table(self.telemetry)
        if not rows:
            raise RuntimeError(
                f"Sweep telemetry at {self.telemetry} has no rows; nothing to compare."
            )

        values = np.array([_as_float(row.get(self.metric)) for row in rows])
        combos = [row["combo"] for row in rows]

        # Sort by the metric so the ordering carries the result. NaN sorts last
        # so a degenerate run does not sit in the middle looking average.
        order = np.argsort(np.where(np.isnan(values), np.inf, values), kind="stable")
        values = values[order]
        combos = [combos[i] for i in order]

        # Colour by how many parameters were estimated, when the sweep varies
        # that. It is the axis this sweep is actually about, so the colour
        # explains the bar heights rather than restating them.
        colours = self._colour_by_parameter_count(rows, order)

        styling = self._styling()
        fig, ax = plt.subplots(
            figsize=styling.get("figsize", (max(6.0, len(rows) * 0.42), 4.2))
        )

        positions = np.arange(len(rows))
        ax.bar(positions, values, color=colours if colours is not None else "tab:blue")

        ax.set_xticks(positions)
        ax.set_xticklabels(combos, rotation=90, fontsize=styling.get("tick_fontsize", 7))
        ax.set_xlabel("Sweep combination (see sweep_manifest.json)")
        ax.set_ylabel(styling.get("ylabel", self.axis_label))
        ax.set_title(f"{self.sweep}: {styling.get('title', self.axis_label)}")

        if self.lower_is_better is not None:
            best_position = 0 if self.lower_is_better else len(values) - 1
            if np.isfinite(values[best_position]):
                ax.annotate(
                    "best",
                    xy=(best_position, values[best_position]),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=styling.get("annotation_fontsize", 7),
                    color="black",
                )

        if np.isnan(values).any():
            n_nan = int(np.isnan(values).sum())
            ax.text(
                0.99,
                0.95,
                f"{n_nan} combination(s) not finite",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=styling.get("annotation_fontsize", 7),
                color="crimson",
            )

        if colours is not None:
            self._add_colour_legend(ax, rows, styling)

        fig.tight_layout()
        return fig, ax

    def _styling(self) -> dict:
        """Per-metric styling from the config, tolerating its absence.

        Reads ``figures.sweep_comparison`` for shared keys and
        ``figures.sweep_comparison.metrics.<metric>`` for overrides, so a
        figure's title or size can be changed without touching code.
        """
        section = OmegaConf.select(self.cfg, "figures.sweep_comparison", default=None)
        merged: dict = {}
        if section is not None:
            merged.update({k: v for k, v in section.items() if k != "metrics"})

        metric_section = OmegaConf.select(
            self.cfg, f"figures.sweep_comparison.metrics.{self.metric}", default=None
        )
        if metric_section is not None:
            merged.update(dict(metric_section.items()))
        return merged

    def _colour_by_parameter_count(self, rows: list[dict], order: np.ndarray):
        """Map each bar to a colour by estimated-parameter count, or ``None``.

        Returns ``None`` when the sweep does not vary the parameter count, in
        which case a single colour is used rather than a meaningless legend.
        """
        counts = [row.get("num_parameters", "") for row in rows]
        if len({c for c in counts if c not in ("", None)}) <= 1:
            return None

        numeric = np.array([_as_float(c) for c in counts])
        numeric = numeric[order]
        if not np.isfinite(numeric).any():
            return None

        from matplotlib import pyplot as plt

        finite = numeric[np.isfinite(numeric)]
        lo, hi = float(finite.min()), float(finite.max())
        if lo == hi:
            return None

        cmap = plt.get_cmap("viridis")
        colours = [
            cmap((v - lo) / (hi - lo)) if np.isfinite(v) else (0.6, 0.6, 0.6, 1.0)
            for v in numeric
        ]
        self._colour_range = (lo, hi)
        return colours

    def _add_colour_legend(self, ax, rows: list[dict], styling: dict) -> None:
        """Explain the colour ramp, which encodes the estimated-parameter count."""
        from matplotlib import pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        if self._colour_range is None:
            return
        lo, hi = self._colour_range
        mappable = ScalarMappable(norm=Normalize(lo, hi), cmap=plt.get_cmap("viridis"))
        mappable.set_array([])
        cbar = ax.figure.colorbar(mappable, ax=ax, pad=0.02)
        cbar.set_label("Estimated parameters", fontsize=styling.get("tick_fontsize", 7))


def run_sweep_comparison(
    cfg: DictConfig,
    telemetry_path: str | Path,
    sweep: str,
    output_dir: str | Path,
    metric: str,
) -> Path:
    """Render one sweep-comparison bar chart.

    Returns
    -------
    Path
        The written PDF.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plotter = SweepComparison(cfg, telemetry_path, sweep, metric)
    plotter.output_dir = output_dir
    plotter.plot()

    pdf_path = output_dir / f"{plotter.name}.pdf"
    if not pdf_path.exists():
        raise RuntimeError(f"Expected {pdf_path} after plotting, but it was not created.")
    logger.info("Wrote %s", pdf_path.name)
    return pdf_path

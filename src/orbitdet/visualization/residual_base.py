"""Shared base class for the observation-driven residual plots.

Every plot in this family is driven by exactly one :class:`ObservationAccess`
and exactly one figure configuration block.  The block is passed directly as
``cfg`` (the contents of ``conf/figures/<name>.yaml``), so there is no
``use_figure_name`` routing: a plotter always draws a single figure from a
single config.

The base class owns the plumbing that used to be copy-pasted into every
subclass:

* reading ``group_by``, ``unit`` and ``include_rejected`` from the config,
* resolving the clustering columns and rendering group labels,
* building the analysis frame and dropping rejected rows,
* the shared legend / styling / suptitle helpers.

Subclasses only implement :meth:`Plot._make_figure`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from orbitdet.analysis import ObservationAccess, analysis_frame
from orbitdet.visualization.base import Plot, _snake_case

#: Marker shapes cycled through for successive clusters.
MARKER_TYPES = ["o", "s", "D", "^", "v", "<", ">", "P", "X"]


def _cfg_get(cfg, *keys, default=None):
    """Read a nested config value, returning ``default`` when any key is missing."""
    cur = cfg
    for key in keys:
        if cur is None:
            return default
        try:
            cur = cur.get(key)
        except Exception:
            try:
                cur = cur[key]
            except Exception:
                return default
    return default if cur is None else cur


def _output_name(cfg) -> str | None:
    """Derive the output stem from the config's ``output_file``, if present."""
    output_file = _cfg_get(cfg, "output_file", default=None)
    if not output_file:
        return None
    return Path(str(output_file)).stem


class ResidualPlot(Plot):
    """Base class for plots driven by an :class:`ObservationAccess`.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (the contents of
        ``conf/figures/<name>.yaml``).  ``group_by``, ``unit`` and
        ``include_rejected`` are read from here; ``output_file`` names the
        output.
    access : ObservationAccess
        The access layer whose derived frame is plotted.
    """

    def __init__(self, cfg: DictConfig, access: ObservationAccess):
        super().__init__(cfg, name=_output_name(cfg))
        self.access = access

    # -- Configuration --------------------------------------------------

    @property
    def group_by(self) -> str | list[str] | None:
        """Standard-frame column(s) to cluster by, from the config."""
        return _cfg_get(self.cfg, "group_by", default=None)

    @property
    def unit(self) -> str:
        """Display unit for the residual axis, from the config."""
        return _cfg_get(self.cfg, "axes", "unit", default="arcsec")

    @property
    def include_rejected(self) -> bool:
        """Whether rows removed by outlier rejection are plotted."""
        return bool(_cfg_get(self.cfg, "include_rejected", default=False))

    def _grouping(self) -> list[str]:
        """Resolve the clustering columns: config, then ``observatory_code``."""
        group_by = self.group_by
        if group_by is None:
            group_by = "observatory_code"
        return [group_by] if isinstance(group_by, str) else list(group_by)

    def _figure_size(self, default_width: float, default_height: float) -> tuple[float, float]:
        """Return the configured ``(width, height)``, falling back to defaults."""
        width = float(_cfg_get(self.cfg, "figure", "width", default=default_width))
        height = float(_cfg_get(self.cfg, "figure", "height", default=default_height))
        return width, height

    # -- Frame ----------------------------------------------------------

    def _frame(self, *, unit: str | None, scan: bool, normalized: bool):
        """Return the analysis frame, filtered to the rows this plot should show."""
        frame = analysis_frame(self.access, unit=unit, scan=scan, normalized=normalized)
        if not self.include_rejected and "rejected" in frame.columns:
            frame = frame.loc[~frame["rejected"].astype(bool)]
        return frame

    def _require_columns(self, frame, columns: list[str]) -> None:
        """Raise if any of ``columns`` is missing from ``frame``."""
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(
                f"Grouping column(s) {missing} are not in the standard frame. "
                f"Available columns: {sorted(frame.columns)}."
            )

    # -- Labels ---------------------------------------------------------

    @staticmethod
    def _group_label(key, columns: list[str]) -> str:
        """Render a groupby key as a legend label."""
        values = key if isinstance(key, tuple) else (key,)
        if len(columns) == 1:
            return str(values[0])
        return ", ".join(f"{column}={value}" for column, value in zip(columns, values, strict=True))

    @staticmethod
    def _rms_label(label: str, value: float, unit: str | None = None) -> str:
        """Render a cluster's legend entry with its RMS."""
        if value is None or not np.isfinite(value):
            return label
        if unit is None:
            return f"{label} - RMS(z): {value:.3f}"
        return f"{label} - RMS: {value:.3e} {unit}"

    def _target_name(self, frame) -> str:
        """Return the observed object's name for the suptitle, or ``"target"``."""
        if "observed_object" in frame.columns and not frame["observed_object"].dropna().empty:
            return str(frame["observed_object"].dropna().iloc[0])
        return "target"

    def _suptitle(self, frame, key: str, default: str) -> str:
        """Resolve and format the suptitle."""
        suptitle = _cfg_get(self.cfg, "titles", key, default=default)
        try:
            if isinstance(suptitle, str):
                suptitle = suptitle.format(target_name=self._target_name(frame))
        except Exception:
            pass
        return suptitle

    # -- Styling --------------------------------------------------------

    def _styling(self) -> tuple:
        """Return ``(colors, marker_size, marker_types)`` from the config."""
        cmap = _cfg_get(self.cfg, "styling", "cmap", default="tab10")
        colors = plt.get_cmap(cmap)
        marker_size = _cfg_get(self.cfg, "styling", "marker_size", default=30)
        return colors, marker_size, MARKER_TYPES

    def _legend(self, ax: plt.Axes) -> None:
        """Apply the configured legend layout to an axes."""
        try:
            legend_ncols = int(_cfg_get(self.cfg, "legend", "ncols", default=2))
        except Exception:
            legend_ncols = 2
        bbox = _cfg_get(self.cfg, "legend", "bbox_to_anchor", default={"x": 0.5, "y": -0.15})
        try:
            if isinstance(bbox, dict):
                bbox_tuple = (float(bbox.get("x", 0.5)), float(bbox.get("y", -0.15)))
            else:
                bbox_tuple = tuple(float(x) for x in bbox)
        except Exception:
            bbox_tuple = (0.5, -0.15)
        ax.legend(ncols=legend_ncols, loc="upper center", bbox_to_anchor=bbox_tuple)


__all__ = ["ResidualPlot", "MARKER_TYPES", "_cfg_get", "_output_name", "_snake_case"]

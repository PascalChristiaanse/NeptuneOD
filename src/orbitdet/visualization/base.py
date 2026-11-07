"""Base class for plotter classes in the visualization module.

A plotter class encapsulates the construction of a matplotlib figure and the
bookkeeping every figure needs: saving it as a PDF to the Hydra output
directory and pickling the figure for interactive inspection.

Subclasses implement :meth:`Plot._make_figure`, which builds and returns the
``(fig, axes)`` tuple. The public :meth:`Plot.plot` method wraps that with the
bookkeeping above, so callers only need::

    fig, ax = Residuals(figure_cfg, access).plot()
"""

import functools
import pickle
import re
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.dates as mdates
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def _snake_case(name: str) -> str:
    """Convert a CamelCase class name to snake_case (e.g. ``ResidualsPSD`` → ``residuals_psd``)."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _datetime_hover_formatter(x, y, hover_x_label: str, hover_y_label: str) -> str:
    """Format ``ax.format_coord`` coordinates with a datetime x axis.

    Module-level (not a closure) so the figure can be pickled by
    :meth:`Plot._save_figure_pickle`.
    """
    try:
        dt = mdates.num2date(x)
        xs = dt.isoformat(sep=" ")
    except Exception:
        xs = f"{x:.6g}"
    return f"{hover_x_label}: {xs}, {hover_y_label}: {y:.3e}"


def _plain_hover_formatter(x, y, hover_x_label: str, hover_y_label: str) -> str:
    """Format ``ax.format_coord`` coordinates with numeric x and y axes.

    Module-level (not a closure) so the figure can be pickled by
    :meth:`Plot._save_figure_pickle`.
    """
    return f"{hover_x_label}: {x:.3e}, {hover_y_label}: {y:.3e}"


def _make_hover_formatter(hover_x_label: str, hover_y_label: str, datetime_x: bool = True):
    """Return a *picklable* ``ax.format_coord`` callable.

    A nested-function closure cannot be pickled, which breaks the ``.fig``
    serialization in :meth:`Plot.publish`. Returning a
    :class:`functools.partial` over a module-level function keeps the formatter
    serializable while preserving the labels and formatting behavior.

    Parameters
    ----------
    hover_x_label, hover_y_label : str
        Labels shown in the interactive coordinate readout.
    datetime_x : bool, optional
        Whether the x coordinate is a Matplotlib date number that should be
        rendered as an ISO timestamp. Defaults to ``True``.
    """
    formatter = _datetime_hover_formatter if datetime_x else _plain_hover_formatter
    return functools.partial(formatter, hover_x_label=hover_x_label, hover_y_label=hover_y_label)


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


#: A4 in inches (210 x 297 mm).
A4_WIDTH_INCHES = 8.27
A4_HEIGHT_INCHES = 11.69

#: Named portions of an A4 page, as ``(width, height)`` in inches.
#:
#: ``full``
#:     The whole page, portrait.
#: ``half_height``
#:     Full width, half the height — the usual shape for a figure that occupies
#:     the top or bottom half of a page.
#: ``half_width``
#:     Half the width, full height — a tall figure beside text.
#: ``quarter_corner``
#:     Half the width and half the height — a corner of the page.
PAPER_SIZES: dict[str, tuple[float, float]] = {
    "full": (A4_WIDTH_INCHES, A4_HEIGHT_INCHES),
    "half_height": (A4_WIDTH_INCHES, A4_HEIGHT_INCHES / 2),
    "half_width": (A4_WIDTH_INCHES / 2, A4_HEIGHT_INCHES),
    "quarter_corner": (A4_WIDTH_INCHES / 2, A4_HEIGHT_INCHES / 2),
}

#: Paper size used when a figure config names none.
DEFAULT_PAPER_SIZE = "half_height"


def paper_size_inches(name: str) -> tuple[float, float]:
    """Return the ``(width, height)`` in inches of a named A4 portion.

    Parameters
    ----------
    name : str
        One of the keys of :data:`PAPER_SIZES`.

    Returns
    -------
    tuple[float, float]
        The figure size in inches.

    Raises
    ------
    ValueError
        If *name* is not a known paper size.
    """
    try:
        return PAPER_SIZES[name]
    except KeyError:
        known = ", ".join(sorted(PAPER_SIZES))
        raise ValueError(f"Unknown paper size '{name}'. Known: {known}.") from None


class Plot(ABC):
    """Base class for all plotter classes.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra experiment configuration.
    name : str, optional
        Base name used for output files. Defaults to the snake_case class name.
    """

    def __init__(self, cfg: DictConfig, name: str | None = None):
        self.cfg = cfg
        self.name = name or _snake_case(self.__class__.__name__)

    @abstractmethod
    def _make_figure(self):
        """Build and return the (fig, axes) tuple for this plot."""

    def _figure_size(self, panels: int = 1, cfg=None) -> tuple[float, float]:
        """Resolve the figure size in inches.

        Resolution order:

        1. ``figure.paper_size`` — a named A4 portion (see :data:`PAPER_SIZES`).
        2. ``figure.width`` / ``figure.height`` — explicit inches.
        3. :data:`DEFAULT_PAPER_SIZE` — half an A4 page.

        There is deliberately no per-plotter default: a figure that names no
        size gets the default paper size, so every figure in the thesis starts
        from the same page geometry.

        Parameters
        ----------
        panels : int, optional
            Number of vertically stacked panels.  The height is multiplied by
            this, so a six-panel figure is six half-pages tall rather than six
            panels squeezed into one.  Defaults to ``1``.
        cfg : DictConfig, optional
            The configuration block to read from.  Defaults to ``self.cfg``.
            Plotters whose config nests the figure block under a key (the
            validation plotters read ``cfg.integrator_convergence``) pass that
            sub-block here.

        Returns
        -------
        tuple[float, float]
            The figure size in inches.
        """
        cfg = self.cfg if cfg is None else cfg

        paper_size = _cfg_get(cfg, "figure", "paper_size", default=None)
        if paper_size is not None:
            width, height = paper_size_inches(str(paper_size))
            return width, height * panels

        width = _cfg_get(cfg, "figure", "width", default=None)
        height = _cfg_get(cfg, "figure", "height", default=None)
        if width is not None or height is not None:
            fallback_width, fallback_height = PAPER_SIZES[DEFAULT_PAPER_SIZE]
            return (
                float(width) if width is not None else fallback_width,
                (float(height) if height is not None else fallback_height) * panels,
            )

        width, height = PAPER_SIZES[DEFAULT_PAPER_SIZE]
        return width, height * panels

    def plot(self):
        """Build the figure, publish it, and return (fig, axes)."""
        fig, ax = self._make_figure()
        return self.publish(fig, ax)

    @staticmethod
    def _save_figure_pickle(fig, path: Path) -> None:
        """Serialize a Matplotlib Figure for later restoration."""
        with path.open("wb") as f:
            pickle.dump(fig, f, protocol=pickle.HIGHEST_PROTOCOL)

    def publish(self, fig, ax):
        """Save the figure as PDF and pickle it.

        Returns
        -------
        tuple
            The (fig, ax) tuple that was passed in.
        """
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save PDF for reports and publication.
        pdf_path = output_dir / f"{self.name}.pdf"
        fig.savefig(pdf_path)

        # Save the Matplotlib Figure for interactive inspection in VS Code.
        fig_path = output_dir / f"{self.name}.fig"
        self._save_figure_pickle(fig, fig_path)

        return fig, ax

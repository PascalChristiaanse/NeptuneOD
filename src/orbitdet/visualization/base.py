"""Base class for plotter classes in the visualization module.

A plotter class encapsulates the construction of a matplotlib figure and the
bookkeeping every figure needs: saving it as a PDF to the Hydra output
directory, logging it to Aim (as a static image), and attaching the saved PDF
as an artifact reference.

Subclasses implement :meth:`Plot._make_figure`, which builds and returns the
``(fig, axes)`` tuple. The public :meth:`Plot.plot` method wraps that with the
bookkeeping above, so callers only need::

    fig, ax = Residuals(cfg, observations).plot()
"""

import functools
import pickle
import re
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.dates as mdates
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from orbitdet.reproducibility import aim_log_artifact_reference, aim_log_figure


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
    return functools.partial(
        formatter, hover_x_label=hover_x_label, hover_y_label=hover_y_label
    )


class Plot(ABC):
    """Base class for all plotter classes.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra experiment configuration.
    name : str, optional
        Base name used for output files and the Aim log entry.
        Defaults to the snake_case class name.
    """

    def __init__(self, cfg: DictConfig, name: str | None = None):
        self.cfg = cfg
        self.name = name or _snake_case(self.__class__.__name__)

    @abstractmethod
    def _make_figure(self):
        """Build and return the (fig, axes) tuple for this plot."""

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
        """Save the figure as PDF and pickle, log it to Aim, and
        attach the PDF as a reference.

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

        # Preserve existing Aim integration.
        aim_log_figure(fig, name=self.name)
        aim_log_artifact_reference(pdf_path)

        return fig, ax

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

import re
from abc import ABC, abstractmethod
from pathlib import Path

from hydra.core.hydra_config import HydraConfig

from orbitdet.reproducibility import aim_log_artifact_reference, aim_log_figure


def _snake_case(name: str) -> str:
    """Convert a CamelCase class name to snake_case (e.g. ``ResidualsPSD`` → ``residuals_psd``)."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


class Plot(ABC):
    """Base class for all plotter classes.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra experiment configuration.
    name : str, optional
        Base name used for the output PDF filename and the Aim log entry.
        Defaults to the snake_case class name.
    """

    def __init__(self, cfg, name: str | None = None):
        self.cfg = cfg
        self.name = name or _snake_case(self.__class__.__name__)

    @abstractmethod
    def _make_figure(self):
        """Build and return the ``(fig, axes)`` tuple for this plot."""

    def plot(self):
        """Build the figure, then save it, log it to Aim, and attach the PDF.

        Equivalent to ``self.publish(*self._make_figure())``.

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes | numpy.ndarray]
            The ``(fig, axes)`` tuple produced by :meth:`_make_figure`.
        """
        fig, ax = self._make_figure()
        return self.publish(fig, ax)

    def publish(self, fig, ax):
        """Save ``fig`` as a PDF, log it to Aim, and attach the PDF as a reference.

        Subclasses (or scripts) that need to modify the figure after it was
        built can call :meth:`_make_figure`, tweak the figure/axes, and then
        call this method to handle the bookkeeping.

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes | numpy.ndarray]
            The ``(fig, ax)`` tuple that was passed in.
        """
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / f"{self.name}.pdf"
        fig.savefig(pdf_path)

        aim_log_figure(fig, name=self.name)
        aim_log_artifact_reference(pdf_path)

        return fig, ax

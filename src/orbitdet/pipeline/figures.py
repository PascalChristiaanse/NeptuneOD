"""Stage: render figures drawn from an observation collection.

The payoff of persisting the collection in its native form: a figure needs no
live environment and no SPICE pool. It is a function of a collection artifact
plus a config slice, so editing a figure's styling re-runs this rule and nothing
else.

Why this is a table rather than a name-to-class lookup
-----------------------------------------------------
Different figures need *different subsets of the same collection*, and that is
load-bearing rather than cosmetic:

* ``prefit_residuals_spice``       -- every observation set
* ``prefit_residuals_spice_scan``  -- Gaia only, because the scan-frame rotation
  needs the archived ``position_angle_scan``, which only Gaia carries
* ``prefit_normalized_residuals``  -- **non-Gaia only**, because a Gaia residual
  is not meaningful without that same rotation

So a figure name selects a *combination* of plotter class and set filter, not
just a class. ``prefit_normalized_residuals`` and ``residual_qq`` both render a
distribution plot, but over different sets.

The name doubles as the styling key: the plotters look their configuration up
under ``cfg.figures.<name>``, so it must match the key the experiment config
declares.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig
from tudatpy.estimation import observations as obs

from orbitdet.pipeline.artifacts import load_collection, load_metadata
from orbitdet.utility import select_observation_collection_by_tag
from orbitdet.visualization.residual_histogram import ResidualHistogram
from orbitdet.visualization.residual_qq import ResidualQQ
from orbitdet.visualization.residual_scatter import ResidualScatter
from orbitdet.visualization.residuals import (
    NormalizedResiduals,
    NormalizedResidualScan,
    Residuals,
    ResidualsScan,
)

logger = logging.getLogger(__name__)

#: ``(collection, metadata) -> collection | None``. Returning ``None`` means no
#: set qualified, and the figure is skipped rather than drawn empty.
SetFilter = Callable[[obs.ObservationCollection, dict], obs.ObservationCollection | None]


def _all_sets(collection: obs.ObservationCollection, metadata: dict):
    """Every observation set."""
    return collection


def _gaia_only(collection: obs.ObservationCollection, metadata: dict):
    """Only the Gaia sets, which carry ``position_angle_scan``."""
    return select_observation_collection_by_tag(collection, metadata, "gaia")


def _non_gaia(collection: obs.ObservationCollection, metadata: dict):
    """Every set except Gaia, whose residuals need the scan rotation."""
    return select_observation_collection_by_tag(collection, metadata, "gaia", exclude=True)


@dataclass(frozen=True)
class FigureSpec:
    """How to render one figure from a collection.

    Attributes
    ----------
    plotter:
        The ``Plot`` subclass to instantiate.
    select:
        Chooses the subset of the collection this figure applies to.
    styled:
        Whether the plotter accepts ``use_figure_name``. The residual plotters
        do, and look their styling up under that key; the distribution plotters
        name themselves after their class, so ``name`` is assigned afterwards.
    """

    plotter: type
    select: SetFilter
    styled: bool = True


#: Figure name -> spec. The name doubles as the styling key and the PDF stem.
COLLECTION_FIGURES: dict[str, FigureSpec] = {
    # Ephemeris-referenced residuals: every set, the Gaia-only scan variants,
    # and the normalized non-Gaia view.
    "prefit_residuals_spice": FigureSpec(Residuals, _all_sets),
    "prefit_residuals_spice_scan": FigureSpec(ResidualsScan, _gaia_only),
    "prefit_normalized_residuals": FigureSpec(NormalizedResiduals, _non_gaia),
    "prefit_normalized_residuals_scan": FigureSpec(NormalizedResidualScan, _gaia_only),
    # Propagated-trajectory residuals, pre-fit.
    "prefit_residuals_prop": FigureSpec(Residuals, _all_sets),
    "prefit_residuals_prop_scan": FigureSpec(ResidualsScan, _gaia_only),
    # Propagated-trajectory residuals, post-fit.
    "postfit_residuals": FigureSpec(Residuals, _all_sets),
    "postfit_residuals_scan": FigureSpec(ResidualsScan, _gaia_only),
    "postfit_normalized_residuals": FigureSpec(NormalizedResiduals, _non_gaia),
    "postfit_normalized_residuals_scan": FigureSpec(NormalizedResidualScan, _gaia_only),
    # Distribution summaries over every set.
    "residual_qq": FigureSpec(ResidualQQ, _all_sets, styled=False),
    "residual_scatter": FigureSpec(ResidualScatter, _all_sets, styled=False),
    "residual_histogram": FigureSpec(ResidualHistogram, _all_sets, styled=False),
}


def _write_placeholder(pdf_path: Path, figure_name: str, reason: str) -> Path:
    """Write a one-page PDF explaining why a figure has no data.

    A figure whose filter matches nothing is a legitimate outcome -- a dataset
    with no Gaia observations has no scan-frame figure -- but Snakemake requires
    every declared output to exist. Writing an explicit placeholder keeps the
    DAG unconditional *and* tells the reader why the figure is empty, which a
    silently missing file would not.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(8, 4))
    fig.text(0.5, 0.62, figure_name, ha="center", va="center", fontsize=14, weight="bold")
    fig.text(0.5, 0.42, reason, ha="center", va="center", fontsize=10, wrap=True)
    fig.text(0.5, 0.2, "No figure produced.", ha="center", va="center", fontsize=9, style="italic")
    fig.savefig(pdf_path)
    plt.close(fig)
    logger.info("Wrote placeholder %s (%s)", pdf_path.name, reason)
    return pdf_path


def run_collection_figure(
    cfg: DictConfig,
    observations_path: str | Path,
    metadata_path: str | Path | None,
    output_dir: str | Path,
    figure_name: str,
) -> Path:
    """Render one collection-based figure.

    Parameters
    ----------
    cfg:
        The ``figures`` config slice.
    observations_path:
        Collection carrying residuals.
    metadata_path:
        ``dataset_metadata.json``, needed by the tag-filtered figures. May be
        ``None`` for figures that use every set.
    output_dir:
        Directory for the PDF.
    figure_name:
        Key under ``cfg.figures`` holding this figure's styling, and the base
        name of the output PDF.

    Returns
    -------
    Path
        The written PDF. When the figure's filter matches no observation set, a
        placeholder PDF is written instead of plotting, so the declared output
        always exists.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{figure_name}.pdf"

    spec = COLLECTION_FIGURES.get(figure_name)
    if spec is None:
        raise ValueError(
            f"Unknown collection figure {figure_name!r}. Known: {sorted(COLLECTION_FIGURES)}"
        )

    collection = load_collection(observations_path)
    metadata = load_metadata(metadata_path) if metadata_path else {}

    selected = spec.select(collection, metadata)
    if selected is None:
        return _write_placeholder(
            pdf_path,
            figure_name,
            f"No observation set matched this figure's filter "
            f"({spec.select.__name__.strip('_')}).",
        )

    if spec.styled:
        plotter = spec.plotter(cfg, use_figure_name=figure_name, observation_collection=selected)
    else:
        plotter = spec.plotter(cfg, selected)

    # Set the name explicitly so the output filename matches the DAG's declared
    # output regardless of how the plotter would have named itself.
    plotter.name = figure_name
    plotter.output_dir = output_dir
    plotter.plot()

    if not pdf_path.exists():
        raise RuntimeError(
            f"Expected {pdf_path} after plotting, but it was not created. "
            f"Directory contains: {sorted(p.name for p in output_dir.iterdir())}"
        )
    logger.info("Wrote %s", pdf_path.name)
    return pdf_path

"""Stage: residual figures from the pre-fit residuals CSV.

This is the payoff of persisting residuals to CSV. The figure needs no
``ObservationCollection``, no furnished SPICE pool, and no environment: it is a
pure function of a CSV plus a config slice. That makes it fast, parallelisable,
and re-runnable on its own — editing a figure title re-runs this rule and
nothing else.

The plotter classes in ``orbitdet.visualization`` expect a live
``ObservationCollection``, so this stage builds a minimal stand-in from the CSV
rows. Only the attributes the plotters actually read are provided.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import DictConfig
from tudatpy.astro.time_representation import Time

from orbitdet.visualization.residuals import Residuals

logger = logging.getLogger(__name__)

RAD_TO_ARCSEC = 180.0 / np.pi * 3600.0


def _load_residuals(csv_path: Path) -> dict[int, dict]:
    """Group CSV rows into per-set structures keyed by ``set_index``."""
    sets: dict[int, dict] = {}
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["set_index"])
            entry = sets.setdefault(
                index,
                {
                    "epochs": [],
                    "components": {},
                    "dataset_id": row["dataset_id"],
                    "observable_type": row["observable_type"],
                    "receiver_body": row.get("receiver_body", ""),
                    "receiver_ref_point": row.get("receiver_ref_point", ""),
                    "transmitter_body": row.get("transmitter_body", ""),
                },
            )
            epoch = float(row["epoch_sec_j2000"])
            if epoch not in entry["epochs"]:
                entry["epochs"].append(epoch)
            entry["components"].setdefault(epoch, {})[int(row["component"])] = float(
                row["residual_rad"]
            )
    return sets


def _link_definition(entry: dict) -> SimpleNamespace:
    """Rebuild the minimal link-end structure the plotters read for labels.

    ``Residuals`` inspects ``link_definition.link_ends`` to derive a series
    label: it looks for a ``receiver`` entry, reads its ``reference_point`` and
    ``body_name``, and falls back to ``observer`` or the first link end. Only
    those attributes are needed, so plain namespaces suffice.
    """
    from tudatpy.estimation.observable_models_setup import links

    receiver = SimpleNamespace(
        body_name=entry["receiver_body"],
        reference_point=entry["receiver_ref_point"],
    )
    transmitter = SimpleNamespace(
        body_name=entry["transmitter_body"],
        reference_point="",
    )
    return SimpleNamespace(link_ends={links.receiver: receiver, links.transmitter: transmitter})


def _build_collection(sets: dict[int, dict]) -> SimpleNamespace:
    """Build a minimal stand-in exposing what ``Residuals`` reads.

    ``Residuals`` calls ``get_single_observation_sets()`` and, per set, reads
    ``observation_times``, ``residuals``, ``observable_type`` and
    ``link_definition``.
    """
    observation_sets = []
    for index in sorted(sets):
        entry = sets[index]
        epochs = sorted(entry["epochs"])
        n_components = max(len(c) for c in entry["components"].values())
        residuals = np.array(
            [
                [entry["components"][epoch].get(k, 0.0) for k in range(n_components)]
                for epoch in epochs
            ]
        )
        observation_sets.append(
            SimpleNamespace(
                observation_times=[Time(epoch) for epoch in epochs],
                residuals=residuals,
                observable_type=entry["observable_type"],
                link_definition=_link_definition(entry),
            )
        )

    return SimpleNamespace(get_single_observation_sets=lambda *args: observation_sets)


def run_residuals_figure(
    cfg: DictConfig,
    residuals_csv: str | Path,
    output_dir: str | Path,
    figure_name: str = "residuals",
) -> Path:
    """Render the residual figure from the CSV.

    Parameters
    ----------
    cfg:
        The ``figures`` config slice.
    residuals_csv:
        ``residuals.csv`` from the prefit stage.
    output_dir:
        Directory for the PDF.
    figure_name:
        Key under ``cfg.figures`` holding this figure's styling. This must
        match the name the experiment script passes to ``Residuals``, because
        the plotter looks its styling up under exactly this key. It also
        determines the output filename (``<figure_name>.pdf``).

    Returns
    -------
    Path
        The written PDF.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sets = _load_residuals(Path(residuals_csv))
    logger.info("Loaded %d observation set(s) from %s", len(sets), Path(residuals_csv).name)

    collection = _build_collection(sets)
    plotter = Residuals(
        cfg,
        use_figure_name=figure_name,
        observation_collection=collection,
    )
    plotter.output_dir = output_dir
    plotter.plot()

    pdf_path = output_dir / f"{figure_name}.pdf"
    if not pdf_path.exists():
        raise RuntimeError(
            f"Expected {pdf_path} after plotting, but it was not created. "
            f"Directory contains: {sorted(p.name for p in output_dir.iterdir())}"
        )
    logger.info("Wrote %s", pdf_path.name)
    return pdf_path

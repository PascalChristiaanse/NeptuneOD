"""Plot observation clusters and print a summary table from a weights CSV.

Loads the per-observation weights CSV written by
:class:`~orbitdet.observations.weighting.WeightEngine` and:

1. Produces a two-panel figure showing RA/DEC residuals coloured by cluster
   with coloured vertical spans for each cluster's time range.
2. Prints a summary table with per-cluster statistics (source, site, type,
   number of observations, RA/DEC RMSE).

Usage
-----
::

    python scripts/plot_weight_groups.py weights_csv=<path_to_observation_weights.csv>

The script can also be used as a Hydra app that loads the CSV from a previous
run's output directory::

    python scripts/plot_weight_groups.py --config-name=sweep/params_x_data
"""

import logging
import os
from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from orbitdet.reproducibility import enforce_initialization, initialize
from orbitdet.visualization import WeightGroups

# ---------------------------------------------------------------------------
# Matplotlib backend setup
# ---------------------------------------------------------------------------
display = os.environ.get("DISPLAY")
is_headless_display = display == ":99" or display == "localhost:99" or display == "127.0.0.1:99"

matplotlib.rcParams["webagg.port"] = 8990
matplotlib.rcParams["webagg.open_in_browser"] = False

if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY") or is_headless_display:
    matplotlib.use("WebAgg", force=True)
elif display or os.environ.get("WAYLAND_DISPLAY"):
    try:
        matplotlib.use("QtAgg", force=True)
    except Exception:
        matplotlib.use("TkAgg", force=True)
else:
    matplotlib.use("WebAgg", force=True)

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="config",
)
@enforce_initialization
def main(cfg: DictConfig):
    initialize(cfg)

    # Determine the weights CSV path
    weights_csv = OmegaConf.select(cfg, "weights_csv", default=None)
    if weights_csv is None:
        # Fall back to the Hydra output dir of the current run
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        weights_csv = str(output_dir / "observation_weights.csv")

    weights_path = Path(weights_csv)
    if not weights_path.exists():
        logger.error("Weights CSV not found at %s", weights_path)
        raise FileNotFoundError(f"Weights CSV not found: {weights_path}")

    logger.info("Loading weights from %s", weights_path)
    weights_df = pd.read_csv(weights_path)

    if weights_df.empty:
        logger.warning("Weights DataFrame is empty — nothing to plot.")
        return

    logger.info(
        "Loaded %d observations across %d unique clusters (strategy: %s)",
        len(weights_df),
        weights_df["cluster_id"].nunique(),
        weights_df["strategy"].iloc[0] if "strategy" in weights_df.columns else "unknown",
    )

    # Build and publish the figure
    plotter = WeightGroups(cfg, weights_df)
    fig, axs = plotter.plot()

    logger.info("Weight clusters figure saved to %s", HydraConfig.get().runtime.output_dir)

    # Build and print the summary tables.  This script only has the weights CSV,
    # not the access layer, so it summarises the report directly: the CSV carries
    # the provenance columns the engine enriched it with.
    from orbitdet.analysis.tables import WeightSummary

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    for suffix, by in (
        ("source", ("source_name",)),
        ("cluster", ("source_name", "cluster_id")),
    ):
        summary = WeightSummary.from_weights(weights_df, by=by, name=f"weight_summary_{suffix}")
        table = summary.build()
        if table.empty:
            logger.info("Weight summary (%s) is empty; skipping.", suffix)
            continue
        print(f"\n=== {suffix.capitalize()} summary ===\n" + summary.to_string(table))
        (output_dir / f"weight_summary_{suffix}.tex").write_text(summary.to_latex(table))
    logger.info("Weight summary tables saved to %s", output_dir)

    # Show interactively if not headless
    if not is_headless_display:
        plt.show(block=True)


if __name__ == "__main__":
    main()

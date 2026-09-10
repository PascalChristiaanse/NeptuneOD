"""Plot observation groups and print a summary table from a weights CSV.

Loads the per-observation weights CSV written by
:class:`~orbitdet.observations.weighting.WeightEngine` and:

1. Produces a two-panel figure showing RA/DEC residuals coloured by group
   with coloured vertical spans for each group's time range.
2. Prints a summary table with per-group statistics (source, site, type,
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

from orbitdet.reproducibility import RuntimeContext, enforce_initialization, initialize
from orbitdet.visualization import WeightGroups, WeightSummaryTable

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
    ctx: RuntimeContext = initialize(cfg)

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
        "Loaded %d observations across %d unique groups (strategy: %s)",
        len(weights_df),
        weights_df["group_id"].nunique(),
        weights_df["strategy"].iloc[0] if "strategy" in weights_df.columns else "unknown",
    )

    # Build and publish the figure
    plotter = WeightGroups(cfg, weights_df)
    fig, axs = plotter.plot()

    logger.info("Weight groups figure saved to %s", HydraConfig.get().runtime.output_dir)

    # Build and print the summary tables
    table_builder = WeightSummaryTable(cfg, weights_df)
    src_table = table_builder.build_source()
    grp_table = table_builder.build_group()
    print("\n=== Per-source summary ===\n" + table_builder.to_string_source(src_table))
    print("\n=== Per-group summary ===\n" + table_builder.to_string_group(grp_table))

    # Save tables alongside the figure
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    src_table.to_csv(output_dir / "weight_summary_source.csv", index=False)
    grp_table.to_csv(output_dir / "weight_summary_group.csv", index=False)
    logger.info("Weight summary tables saved to %s", output_dir)

    with open(output_dir / "weight_summary_source.tex", "w") as f:
        f.write(table_builder.to_latex_source(src_table))
    with open(output_dir / "weight_summary_group.tex", "w") as f:
        f.write(table_builder.to_latex_group(grp_table))
    logger.info("LaTeX weight summary tables saved to %s", output_dir)

    # Save as Excel with two sheets
    with pd.ExcelWriter(output_dir / "weight_summary.xlsx") as writer:
        src_table.to_excel(writer, sheet_name="Per source", index=False)
        grp_table.to_excel(writer, sheet_name="Per group", index=False)
    logger.info("Excel weight summary tables saved to %s", output_dir)

    # Show interactively if not headless
    if not is_headless_display:
        plt.show(block=True)


if __name__ == "__main__":
    main()
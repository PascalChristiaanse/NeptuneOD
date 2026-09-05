"""Diagnostic script to visualise outlier rejection on pre-fit residuals.

Loads observations, computes pre-fit residuals, applies outlier rejection,
and plots the accepted vs. rejected observations with distinct markers so
you can visually verify that the filter is working correctly.
"""

import json
import logging
import os
from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tudatpy.astro.time_representation import iso_string_to_epoch_time_object
from tudatpy.estimation import observations as obs
from tudatpy.estimation.observations_setup import observations_simulation_settings as obs_sim_setup

from orbitdet.data import KernelManager
from orbitdet.observations import (
    OutlierEngine,
    create_observation_collection,
)
from orbitdet.reproducibility import RuntimeContext, enforce_initialization, initialize
from orbitdet.simulation import get_environment
from orbitdet.visualization import Residuals

# ---------------------------------------------------------------------------
# Matplotlib backend setup (same as generate_prefit_residuals.py)
# ---------------------------------------------------------------------------
display = os.environ.get("DISPLAY")
is_headless_display = display == ":99" or display == "localhost:99" or display == "127.0.0.1:99"

matplotlib.rcParams["webagg.port"] = 8988
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def _rad_to_arcsec(angle_rad: np.ndarray) -> np.ndarray:
    return np.rad2deg(angle_rad) * 3600.0


def _seconds_since_j2000_to_datetimes(seconds_since_j2000: np.ndarray):
    import pandas as pd

    return pd.to_datetime(
        seconds_since_j2000,
        unit="s",
        origin=pd.Timestamp("2000-01-01T12:00:00"),
    )


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="experiment/generate_prefit_residuals",
)
@enforce_initialization
def main(cfg: DictConfig):
    ctx: RuntimeContext = initialize(cfg)

    # Inject start and end epochs into the runtime context
    ctx.start_epoch = iso_string_to_epoch_time_object(cfg.start_date)
    ctx.end_epoch = iso_string_to_epoch_time_object(cfg.end_date)

    km: KernelManager = KernelManager(cfg)
    km.download_all_kernels()
    km.furnish()
    km.download_all_data_files()
    logger.info("Configuration loaded and runtime initialized successfully.")

    bodies = get_environment(cfg, ctx)
    logger.info("Environment created successfully.")

    # Create observations
    observations, observation_models = create_observation_collection(cfg, bodies)
    logger.info("Observations generated successfully.")

    # Create observation simulators for pre-fit residuals
    ephemeris_observation_simulators = obs_sim_setup.create_observation_simulators(
        observation_models, bodies
    )
    logger.info("Observation simulators created successfully.")

    # Populate residuals in SingleObservationSets
    obs.compute_residuals_and_dependent_variables(
        observations, ephemeris_observation_simulators, bodies
    )
    logger.info("Pre-fit residuals computed successfully.")

    # ------------------------------------------------------------------
    # Outlier rejection diagnostic
    # ------------------------------------------------------------------
    outlier_cfg = OmegaConf.select(cfg, "outlier_rejection")
    rejection_enabled = outlier_cfg is not None and outlier_cfg.get("enabled", False)

    if rejection_enabled:
        logger.info("Applying outlier rejection for diagnostic plot...")

        # Use the new apply_with_rejected API that returns both accepted and rejected collections
        outlier_engine = OutlierEngine.from_config(outlier_cfg)
        accepted_collection, rejected_collection, rejection_metadata = (
            outlier_engine.apply_with_rejected(observations, bodies)
        )

        # Log summary
        n_accepted = rejection_metadata["n_accepted"]
        n_rejected = rejection_metadata["n_rejected"]
        n_total = rejection_metadata["n_total_observations"]
        logger.info(
            "Outlier rejection: %d accepted, %d rejected out of %d",
            n_accepted,
            n_rejected,
            n_total,
        )

        # Save rejection metadata to JSON
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rejection_path = output_dir / "outlier_rejection_metadata.json"
        with open(rejection_path, "w") as f:
            json.dump(rejection_metadata, f, indent=2, default=str)
        logger.info("Outlier rejection metadata saved to %s", rejection_path)

        # ------------------------------------------------------------------
        # Plot: accepted vs rejected on the same axes
        # ------------------------------------------------------------------
        fig, axs = plt.subplots(
            2,
            1,
            figsize=(16.54, 8.27),
            sharex=True,
        )

        # Plot accepted observations using the standard Residuals plotter
        # (this handles all the per-set coloring and labeling)
        residuals_plot = Residuals(
            cfg,
            observation_collection=accepted_collection,
            fig=fig,
            ax=axs,
        )
        residuals_plot.plot()

        # Overlay rejected observations with a distinct marker
        for obs_set in rejected_collection.get_single_observation_sets():
            obs_times_sec = np.array(
                [epoch.to_float() for epoch in obs_set.observation_times]
            )
            if len(obs_times_sec) == 0:
                continue
            obs_times = _seconds_since_j2000_to_datetimes(obs_times_sec)
            residuals = np.array(obs_set.residuals)
            ra_resid_arcsec = _rad_to_arcsec(residuals[:, 0])
            dec_resid_arcsec = _rad_to_arcsec(residuals[:, 1])

            axs[0].scatter(
                obs_times,
                ra_resid_arcsec,
                marker="x",
                s=80,
                color="red",
                alpha=0.8,
                label="Rejected (RA)" if len(obs_times_sec) > 0 else "",
                zorder=5,
            )
            axs[1].scatter(
                obs_times,
                dec_resid_arcsec,
                marker="x",
                s=80,
                color="red",
                alpha=0.8,
                label="Rejected (DEC)" if len(obs_times_sec) > 0 else "",
                zorder=5,
            )

        # Add threshold lines if residual_threshold strategy was used
        for strategy_cfg in outlier_cfg.get("strategies", []):
            if strategy_cfg.get("type") == "residual_threshold":
                threshold_arcsec = strategy_cfg.get("threshold_arcsec", 1.5)
                for ax in axs:
                    ax.axhline(y=threshold_arcsec, color="red", linestyle="--", alpha=0.4)
                    ax.axhline(y=-threshold_arcsec, color="red", linestyle="--", alpha=0.4)
                break

        # Add legend entry for rejected
        # (Avoid duplicate legend entries by collecting unique labels)
        handles, labels = axs[0].get_legend_handles_labels()
        # Remove duplicate accepted entries from the legend for clarity
        # by keeping only the last occurrence of each label
        seen = set()
        unique_handles_labels = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                unique_handles_labels.append((h, l))
        # Add a single "Rejected" entry
        from matplotlib.lines import Line2D

        unique_handles_labels.append(
            (Line2D([0], [0], marker="x", color="red", linestyle="None", markersize=8), "Rejected")
        )
        axs[0].legend(
            [h for h, _ in unique_handles_labels],
            [l for _, l in unique_handles_labels],
            ncols=3,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
        )

        fig.suptitle(
            f"Pre-Fit Residuals — Outlier Rejection Diagnostic\n"
            f"Accepted: {n_accepted}, Rejected: {n_rejected}, Total: {n_total}"
        )
        fig.set_tight_layout(True)

        # Save the diagnostic figure
        diag_path = output_dir / "outlier_rejection_diagnostic.pdf"
        fig.savefig(diag_path)
        logger.info("Outlier rejection diagnostic plot saved to %s", diag_path)

    else:
        logger.info("Outlier rejection disabled — plotting standard pre-fit residuals only.")
        fig, ax = Residuals(cfg, observations).plot()

    # ------------------------------------------------------------------
    # Interactive display
    # ------------------------------------------------------------------
    backend = plt.get_backend().lower()
    if backend == "agg" or "inline" in backend:
        logger.info("Skipping interactive display because matplotlib backend is %s.", backend)
    else:
        if backend == "webagg":
            logger.info("Open the interactive plot at http://localhost:8988")
        plt.show(block=True)

    logger.info("Outlier rejection diagnostic script completed.")


if __name__ == "__main__":
    main()
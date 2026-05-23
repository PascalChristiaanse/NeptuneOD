import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from tudatpy.estimation import observations as obs
from tudatpy.estimation.observable_models_setup import links
from tudatpy.estimation.observations import observations_processing as obs_proc

from orbitdet.observations import get_observatory_info


def plot_residuals(
    cfg: DictConfig,
    observation_collection: obs.ObservationCollection,
    observation_parsers: list[obs_proc.ObservationParserType] = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot pre-fit and post-fit residuals for the orbit determination."""

    if observation_parsers is None:
        observation_parsers = obs_proc.observation_parser("Earth")

    observation_sets: list[obs.SingleObservationSet] = (
        observation_collection.get_single_observation_sets(observation_parsers)
    )

    fig, axs = plt.subplots(2, 1, figsize=(8.27*2, 8.27*2 / 2))  # A4 aspect ratio half page
    colors = plt.get_cmap("tab20")
    for set_index, obs_set in enumerate(observation_sets):
        observatory_code = obs_set.link_definition.link_ends[links.receiver].reference_point
        target_name = obs_set.link_definition.link_ends[links.transmitter].body_name
        info = get_observatory_info(cfg, observatory_code)
        color = colors(set_index % colors.N)

        obs_times_sec_j2000 = np.array([epoch.to_float() for epoch in obs_set.observation_times])
        obs_times = obs_times_sec_j2000 / 365.25 / 24 / 3600 + 2000  # Convert to years since J2000
        residuals = np.array(obs_set.computed_observations)
        # n x 2 array of RA and DEC residuals in radians

        # RA
        axs[0].scatter(
            obs_times,
            residuals[:, 0],
            marker=".",
            s=30,
            label=(
                f"{info['name']} - {info['region']} - RMS: "
                f"{np.std(residuals[:, 0]) * 1e6:.2f} µas"
            ),
            color=color,
        )
        # DEC
        axs[1].scatter(
            obs_times,
            residuals[:, 1],
            marker=".",
            s=30,
            label=(
                f"{info['name']} - {info['region']} - RMS: "
                f"{np.std(residuals[:, 1]) * 1e6:.2f} µas"
            ),
            color=color,
        )

    axs[0].set_title("Right Ascension")
    axs[1].set_title("Declination")
    # add legend with observatory names and RMS values
    axs[0].legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    axs[1].legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.suptitle(f"Pre-Fit Residuals for {target_name}")
    fig.set_tight_layout(True)

    return fig, axs

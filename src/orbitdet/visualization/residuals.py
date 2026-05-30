import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from tudatpy.estimation import observations as obs
from tudatpy.estimation.observable_models_setup import links
from tudatpy.estimation.observations import observations_processing as obs_proc

from orbitdet.observations import get_observatory_info


def _wrap_angle_rad(angle_rad: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle_rad), np.cos(angle_rad))


def _rad_to_arcsec(angle_rad: np.ndarray) -> np.ndarray:
    return np.rad2deg(angle_rad) * 3600.0


def _rms_arcsec(values_arcsec: np.ndarray) -> float | None:
    finite_values = values_arcsec[np.isfinite(values_arcsec)]
    if finite_values.size == 0:
        return None

    return float(np.sqrt(np.mean(np.square(finite_values))))


def plot_residuals(
    cfg: DictConfig,
    observation_collection: obs.ObservationCollection,
    observation_parsers: list[obs_proc.ObservationParserType] = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot pre-fit and post-fit residuals for the orbit determination."""
    if observation_parsers is None:
        observation_sets: list[obs.SingleObservationSet] = (
            observation_collection.get_single_observation_sets()
        )
    else:
        observation_sets: list[obs.SingleObservationSet] = (
            observation_collection.get_single_observation_sets(observation_parsers)
        )

    fig, axs = plt.subplots(2, 1, figsize=(8.27 * 2, 8.27 * 2 / 2))  # A4 aspect ratio half page
    colors = plt.get_cmap("tab20")
    for set_index, obs_set in enumerate(observation_sets):
        observatory_code = obs_set.link_definition.link_ends[links.receiver].reference_point
        target_name = obs_set.link_definition.link_ends[links.transmitter].body_name
        info = get_observatory_info(cfg, observatory_code)
        color = colors(set_index % colors.N)

        obs_times_sec_j2000 = np.array([epoch.to_float() for epoch in obs_set.observation_times])
        obs_times = obs_times_sec_j2000 / 365.25 / 24 / 3600 + 2000  # Convert to years since J2000
        residuals = np.array(obs_set.residuals)
        # n x 2 array of RA and DEC residuals in radians

        ra_residuals_arcsec = _rad_to_arcsec(_wrap_angle_rad(residuals[:, 0]))
        dec_residuals_arcsec = _rad_to_arcsec(residuals[:, 1])
        ra_rms_arcsec = _rms_arcsec(ra_residuals_arcsec)
        dec_rms_arcsec = _rms_arcsec(dec_residuals_arcsec)
        ra_rms_label = f"{ra_rms_arcsec:.3e} arcsec" if ra_rms_arcsec is not None else None
        dec_rms_label = f"{dec_rms_arcsec:.3e} arcsec" if dec_rms_arcsec is not None else None

        # RA
        axs[0].scatter(
            obs_times,
            ra_residuals_arcsec,
            marker=".",
            s=30,
            label=f"{info['name']} - {info['region']} - RMS: {ra_rms_label}",
            color=color,
        )
        # DEC
        axs[1].scatter(
            obs_times,
            dec_residuals_arcsec,
            marker=".",
            s=30,
            label=f"{info['name']} - {info['region']} - RMS: {dec_rms_label}",
            color=color,
        )

    axs[0].set_title("Right Ascension")
    axs[1].set_title("Declination")
    axs[0].set_ylabel("Residual [arcsec]")
    axs[1].set_ylabel("Residual [arcsec]")
    axs[1].set_xlabel("Epoch [year]")
    # add legend with observatory names and RMS values
    axs[0].legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    axs[1].legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.suptitle(f"Pre-Fit Residuals for {target_name}")
    fig.set_tight_layout(True)

    return fig, axs

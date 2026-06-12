import logging

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tudatpy.astro import time_representation as time_repr
from tudatpy.interface import spice

from orbitdet.transformations import convert_cartesian_frame

logger = logging.getLogger(__name__)


def load_voyager_table(file_path: str) -> pd.DataFrame:
    table = pd.read_csv(file_path, sep=",")
    if "PICNO" not in list(table.columns):
        raise ValueError(f"Expected a picno column in {file_path}")
    return table


def ephemeris_type(cfg: DictConfig, dataset_cfg: DictConfig) -> str:
    receiver_name = str(getattr(getattr(dataset_cfg, "observatory", None), "name", "Voyager 2"))
    bodies_to_create = getattr(cfg, "bodies_to_create", {})
    receiver_cfg = (
        bodies_to_create.get(receiver_name, {}) if hasattr(bodies_to_create, "get") else {}
    )
    ephemeris_cfg = getattr(receiver_cfg, "ephemeris", None)
    if ephemeris_cfg is None:
        ephemeris_cfg = getattr(dataset_cfg, "ephemeris", None)
    if ephemeris_cfg is None:
        return ""
    return str(getattr(ephemeris_cfg, "type", ephemeris_cfg))


def merge_voyager_tables(
    data: pd.DataFrame,
    ancillary_data: pd.DataFrame,
    merge_key: str,
    dataset_identifier: str,
    ephemeris_type: str,
) -> pd.DataFrame:
    merged_data = data.merge(
        ancillary_data,
        on=merge_key,
        how="left",
        suffixes=("_data", "_ancillary"),
        indicator=True,
    )

    if merged_data.empty:
        raise ValueError(
            f"Voyager merge on '{merge_key}' produced no rows for {dataset_identifier}."
        )

    if ephemeris_type == "tabulated_from_ancillary_file":
        unmatched_rows = merged_data[merged_data["_merge"] == "left_only"]
        for _, row in unmatched_rows.iterrows():
            logger.warning(
                "Voyager dataset %s has no ancillary match for %s=%s.",
                dataset_identifier,
                merge_key,
                row[merge_key],
            )

    return merged_data.drop(columns=["_merge"])


def build_voyager_tabulated_state_history(
    cfg: DictConfig,
    dataset_cfg: DictConfig,
) -> dict[float, np.ndarray]:
    if cfg.global_frame_origin != "SSB" and cfg.global_frame_orientation != "Neptune Barycenter":
        raise ValueError(
            "Voyager tabulated ephemeris requires global frame to be Neptune Barycenter or SSB"
        )

    merged_data = load_and_merge_voyager_tables(cfg, dataset_cfg)

    required_columns = ["jd", "x_km", "y_km", "z_km"]
    missing_columns = [column for column in required_columns if column not in merged_data.columns]
    if missing_columns:
        raise ValueError(
            "Voyager ancillary data is missing required columns for tabulated ephemeris: "
            + ", ".join(missing_columns)
        )

    # Correct for delta Tau if origin is SSB. Voyager ephemeris is in the modified
    # Neptune-centric frame; the paper defines r'(t, tau(t)) = r(t) + [b(t) - b(t-tau(t))].
    # Here, r is the spacecraft position in SSB, b is Neptune's position in SSB, and tau(t)
    # is the light time as a function of observation epoch.
    # t is the observation epoch, and r(t) is the tabulated ephemeris position.
    if cfg.global_frame_origin == "SSB":
        r_t_tau = merged_data[["x_km", "y_km", "z_km"]].to_numpy(dtype=float) * 1000.0
        light_time_seconds = merged_data["t_sec"].to_numpy(dtype=float)
        epochs = merged_data["epoch_TDB"].to_numpy(dtype=float)
        b_t = []
        b_t_tau = []
        for epoch, light_time in zip(epochs, light_time_seconds):
            b_t.append(
                spice.get_body_cartesian_position_at_epoch(
                    "Neptune Barycenter", "SSB", cfg.global_frame_orientation, "NONE", epoch
                )
            )
            b_t_tau.append(
                spice.get_body_cartesian_position_at_epoch(
                    "Neptune Barycenter",
                    "SSB",
                    cfg.global_frame_orientation,
                    "NONE",
                    epoch - light_time,
                )
            )
        b_t = np.asarray(b_t, dtype=float)
        b_t_tau = np.asarray(b_t_tau, dtype=float)
        r_t = r_t_tau - (b_t - b_t_tau)
        merged_data["x_km"] = r_t[:, 0] / 1000.0
        merged_data["y_km"] = r_t[:, 1] / 1000.0
        merged_data["z_km"] = r_t[:, 2] / 1000.0
        logger.info("Applied light time correction to Voyager tabulated ephemeris for SSB origin.")

    merged_data = merged_data.dropna(subset=["epoch_TDB", "jd", "x_km", "y_km", "z_km"])
    ordered_data = merged_data.sort_values("jd").drop_duplicates(subset="jd", keep="first")
    ordered_data = convert_cartesian_frame(
        data=ordered_data,
        x_column="x_km",
        y_column="y_km",
        z_column="z_km",
        input_frame=dataset_cfg.epoch_of_equinox,
        output_frame=cfg.global_frame_orientation,
        time_column="epoch_TDB",
    )
    logger.info(
        f"""Converted Voyager tabulated ephemeris from {dataset_cfg.epoch_of_equinox} """
        f"""to {cfg.global_frame_orientation}."""
    )

    time_seconds = ordered_data["epoch_TDB"].to_numpy(dtype=float)

    positions_km = ordered_data[["x_km", "y_km", "z_km"]].to_numpy(dtype=float)
    velocities_km_s = np.vstack(
        [np.gradient(positions_km[:, axis], time_seconds) for axis in range(3)]
    ).T

    state_history: dict[float, np.ndarray] = {}
    for index, time_value in enumerate(time_seconds):
        state = np.hstack((positions_km[index], velocities_km_s[index])).reshape(6, 1)
        state_history[float(time_value)] = state

    # convert to m and m/s for use in Tudat
    for time_value in state_history:
        state_history[time_value][:3] *= 1000.0
        state_history[time_value][3:] *= 1000.0

    logger.info(
        "Built Voyager tabulated ephemeris state history with %d samples.",
        len(state_history),
    )
    return state_history


def load_and_merge_voyager_tables(
    cfg: DictConfig,
    dataset_cfg: DictConfig,
) -> pd.DataFrame:
    data = load_voyager_table(dataset_cfg.data.file)
    ancillary_data = load_voyager_table(dataset_cfg.ancillary_files.file)

    merge_key = str(getattr(dataset_cfg, "merge_key", "picno"))
    merged_data = merge_voyager_tables(
        data,
        ancillary_data,
        merge_key,
        dataset_cfg.identifier,
        ephemeris_type(cfg, dataset_cfg),
    )

    if "date_jed" in merged_data.columns and "jd" not in merged_data.columns:
        merged_data = merged_data.rename(columns={"date_jed": "jd"})

    merged_data["epoch_TDB"] = merged_data["jd"].map(
        lambda value: (
            np.nan if pd.isna(value) else time_repr.julian_day_to_seconds_since_epoch(value)
        )
    )

    return merged_data

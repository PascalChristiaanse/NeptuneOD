import logging

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
from omegaconf import DictConfig

from orbitdet.transformations import convert_fk4_b1950_cartesian_to_icrs_j2000

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


def _jd_to_seconds_since_j2000_tdb(jd_values: np.ndarray) -> np.ndarray:
    reference_epoch = Time("2000-01-01T12:00:00", scale="tdb")
    times = Time(jd_values, format="jd", scale="tdb")
    return (times - reference_epoch).to_value(u.s)


def apply_voyager_time_offset_seconds(data: pd.DataFrame, time_offset_seconds) -> pd.DataFrame:
    if time_offset_seconds is None:
        return data

    offset_seconds = float(time_offset_seconds)
    if offset_seconds == 0.0:
        return data

    shifted_data = data.copy()
    offset_days = offset_seconds / 86400.0
    if "jd" in shifted_data.columns:
        shifted_data["jd"] = shifted_data["jd"].astype(float) + offset_days
    if "epoch_TDB" in shifted_data.columns:
        shifted_data["epoch_TDB"] = shifted_data["epoch_TDB"].astype(float) + offset_seconds

    logger.info("Applied Voyager time offset of %.3f seconds.", offset_seconds)
    return shifted_data


def build_voyager_tabulated_state_history(
    cfg: DictConfig,
    dataset_cfg: DictConfig,
) -> dict[float, np.ndarray]:
    merged_data = load_and_merge_voyager_tables(cfg, dataset_cfg)

    required_columns = ["jd", "x_km", "y_km", "z_km"]
    missing_columns = [column for column in required_columns if column not in merged_data.columns]
    if missing_columns:
        raise ValueError(
            "Voyager ancillary data is missing required columns for tabulated ephemeris: "
            + ", ".join(missing_columns)
        )

    epoch_of_equinox = getattr(dataset_cfg, "epoch_of_equinox", None)
    merged_data = convert_fk4_b1950_cartesian_to_icrs_j2000(
        merged_data,
        "x_km",
        "y_km",
        "z_km",
        epoch_of_equinox,
    )
    merged_data = apply_voyager_time_offset_seconds(
        merged_data, getattr(dataset_cfg, "time_offset_seconds", None)
    )

    # Filter all entries containing NaN in jd, x_km, y_km, or z_km, as these are
    # required for the tabulated ephemeris
    merged_data = merged_data.dropna(subset=["jd", "x_km", "y_km", "z_km"])

    ordered_data = merged_data.sort_values("jd").drop_duplicates(subset="jd", keep="first")
    time_seconds = _jd_to_seconds_since_j2000_tdb(ordered_data["jd"].to_numpy(dtype=float))
    if time_seconds.size < 2:
        raise ValueError(
            "Need at least two Voyager ephemeris samples to create a tabulated ephemeris."
        )

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

    merged_data = apply_voyager_time_offset_seconds(
        merged_data, getattr(dataset_cfg, "time_offset_seconds", None)
    )

    return merged_data

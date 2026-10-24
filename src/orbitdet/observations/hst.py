"""Hubble Space Telescope (HST) observation dataset factories.

Creates Tudat observation sets from HST astrometric observations of a
solar-system body (e.g. Triton).  Retrieval and normalisation are handled by
:class:`~orbitdet.data.hst_data.HSTQuery`, which combines two public sources:

* ``hst_mast``     — pointing RA/Dec from the MAST archive (astroquery);
* ``hst_springer`` — absolute RA/Dec reconstructed from the Showalter et al.
  (2019) Springer supplement's relative ``ΔRA``/``Δdec`` offsets.

Both factories build an absolute angular-position observation set with the
target body as transmitter and the HST spacecraft as receiver.  The HST body
must be present in the system of bodies (loaded from the standard ``hst.bsp``
SPICE kernel); no custom tabulated ephemeris is required.
"""

import logging

import numpy as np
import pandas as pd
import tudatpy.dynamics.environment as env
import tudatpy.estimation.observable_models_setup as obs_model_setup
import tudatpy.estimation.observations as obs
from omegaconf import DictConfig

from orbitdet.data.hst_data import HSTQuery
from orbitdet.transformations import convert_radec_frame

from .registry import register_dataset_factory

logger = logging.getLogger(__name__)


def _build_hst_observation_set(
    cfg: DictConfig,
    dataset_cfg: DictConfig,
    system_of_bodies: env.SystemOfBodies,
    table: pd.DataFrame,
) -> tuple[obs.SingleObservationSet, obs_model_setup.model_settings.ObservationModelSettings]:
    """Build a single absolute angular-position observation set from a normalised table."""
    target_name = str(dataset_cfg.target)
    receiver_name = str(getattr(dataset_cfg, "receiver", "HST"))
    epoch_of_equinox = str(getattr(dataset_cfg, "epoch_of_equinox", "J2000"))

    if not system_of_bodies.does_body_exist(target_name):
        raise ValueError(
            f"HST dataset {dataset_cfg.identifier}: target body '{target_name}' is not "
            "in the system of bodies."
        )
    if not system_of_bodies.does_body_exist(receiver_name):
        raise ValueError(
            f"HST dataset {dataset_cfg.identifier}: receiver body '{receiver_name}' is not "
            "in the system of bodies (add it to bodies_to_create and load the HST kernel)."
        )

    # Rotate the normalised RA/Dec (degrees) into the experiment frame.
    table = convert_radec_frame(
        table,
        "ra",
        "dec",
        input_frame=epoch_of_equinox,
        output_frame=cfg.global_frame_orientation,
        time_column="epoch_TDB",
    )

    valid_rows = table[["epoch_TDB", "ra", "dec"]].dropna()
    times = valid_rows["epoch_TDB"].to_numpy()
    data = [
        np.deg2rad(row[["ra", "dec"]].to_numpy(dtype=float)).reshape(2, 1)
        for _, row in valid_rows.iterrows()
    ]

    link_ends = dict()
    link_ends[obs_model_setup.links.transmitter] = obs_model_setup.links.body_origin_link_end_id(
        target_name
    )
    link_ends[obs_model_setup.links.receiver] = obs_model_setup.links.body_origin_link_end_id(
        receiver_name
    )
    link_definition = obs_model_setup.links.LinkDefinition(link_ends)
    observation_model = obs_model_setup.model_settings.angular_position(link_definition)

    observation_set = obs.create_single_observation_set(
        obs_model_setup.model_settings.angular_position_type,
        link_ends,
        data,
        times,
        obs_model_setup.links.LinkEndType.receiver,
    )

    return observation_set, observation_model


def _prepare_table(dataset_cfg: DictConfig, query: HSTQuery, source: str):
    """Retrieve and normalise the observation table for the configured source."""
    target = str(dataset_cfg.target)
    source = str(source).lower()

    if source == HSTQuery.MAST_SOURCE.lower():
        table = query.retrieve_mast_data(
            target=target,
            sci_aec=str(getattr(dataset_cfg, "sci_aec", "S")),
        )
    elif source == HSTQuery.SPRINGER_SOURCE.lower():
        table = query.retrieve_springer_data(
            dataset_cfg.file,
            moon=target,
        )
        if bool(getattr(dataset_cfg, "exclude_unused", True)):
            n_before = len(table)
            table = table[table["weight"] != 0]
            logger.info(
                f"HST dataset {dataset_cfg.identifier}: dropped {n_before - len(table)} "
                "measurement(s) flagged as unused (Weight == 0)."
            )
    else:
        raise ValueError(
            f"HST dataset {dataset_cfg.identifier}: unknown source '{source}'. "
            f"Expected '{HSTQuery.MAST_SOURCE}' or '{HSTQuery.SPRINGER_SOURCE}'."
        )

    if table.empty:
        raise ValueError(f"HST dataset {dataset_cfg.identifier} produced no observations.")
    return table


def _create_hst_dataset(
    cfg: DictConfig, dataset_cfg: DictConfig, source: str, system_of_bodies: env.SystemOfBodies
) -> tuple[obs.SingleObservationSet, obs_model_setup.model_settings.ObservationModelSettings]:
    """Shared implementation for the MAST and Springer HST dataset factories."""
    logger.info(f"Creating HST ({source}) observation dataset: {dataset_cfg.identifier}.")

    for required in ("target",):
        if not hasattr(dataset_cfg, required):
            raise ValueError(
                f"HST dataset {dataset_cfg.identifier} must specify '{required}'."
            )

    query = HSTQuery()
    table = _prepare_table(dataset_cfg, query, source)

    observation_set, observation_model = _build_hst_observation_set(
        cfg, dataset_cfg, system_of_bodies, table
    )

    logger.info(
        f"HST ({source}) observation dataset {dataset_cfg.identifier} created with "
        f"{len(observation_set.observation_times)} row(s)."
    )
    return observation_set, observation_model


@register_dataset_factory("hst_mast")
def create_hst_mast_dataset(
    cfg: DictConfig, dataset_cfg: DictConfig, system_of_bodies: env.SystemOfBodies
) -> tuple[obs.SingleObservationSet, obs_model_setup.model_settings.ObservationModelSettings]:
    """Create an HST dataset from MAST archive pointing coordinates.

    Args:
        cfg: Experiment configuration with necessary metadata.
        dataset_cfg: Dataset configuration; must contain ``target`` (and optionally
            ``receiver`` and ``sci_aec``).
        system_of_bodies: The environment containing the bodies for which to
            create the dataset.

    Returns:
        Tuple of (SingleObservationSet, ObservationModelSettings) for the MAST data.
    """
    return _create_hst_dataset(cfg, dataset_cfg, "MAST", system_of_bodies)


@register_dataset_factory("hst_springer")
def create_hst_springer_dataset(
    cfg: DictConfig, dataset_cfg: DictConfig, system_of_bodies: env.SystemOfBodies
) -> tuple[obs.SingleObservationSet, obs_model_setup.model_settings.ObservationModelSettings]:
    """Create an HST dataset from the Showalter et al. (2019) Springer supplement.

    Args:
        cfg: Experiment configuration with necessary metadata.
        dataset_cfg: Dataset configuration; must contain ``target`` and ``file``
            (path to the Excel workbook), and optionally ``receiver``.
        system_of_bodies: The environment containing the bodies for which to
            create the dataset.

    Returns:
        Tuple of (SingleObservationSet, ObservationModelSettings) for the Springer data.
    """
    if not hasattr(dataset_cfg, "file"):
        raise ValueError(
            f"HST dataset {dataset_cfg.identifier} must specify 'file' (path to the "
            "Showalter et al. Excel workbook)."
        )
    return _create_hst_dataset(cfg, dataset_cfg, "Springer", system_of_bodies)

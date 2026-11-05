"""Gaia observation dataset factory.

The measurements arrive as a standard observation frame, built by
:mod:`orbitdet.data.gaia_source` from the Gaia FPR archive.  Retrieval, unit
conversion, the astrometric corrections and the frame assembly all happen there,
so this factory does what every other factory does: turn a frame into link ends,
observation sets and a model.

Why this factory still exists at all
------------------------------------
Because a wide dataframe cannot carry Gaia's weights.  Gaia reports the
covariance of a whole *transit* - an ``(n, n)`` block across that transit's
observations - and ``set_full_weight_matrix`` consumes it as a matrix, not as a
column.  Dropping the off-diagonal terms and letting the generic frame-only path
handle Gaia is not neutral: the correlation between the two coordinates of one
transit is a real part of the error model, and discarding it understates
correlated uncertainty.

So the split is: the frame owns what is per-observation - including the marginal
uncertainties and the scan angle, both of which genuinely are per-observation -
and this factory owns what is per-transit.
"""

import logging

import numpy as np
import tudatpy.dynamics.environment as env
import tudatpy.estimation.observable_models_setup as obs_model_setup
import tudatpy.estimation.observations as obs
from omegaconf import DictConfig

from orbitdet.data.gaia_source import GaiaSource
from orbitdet.observations.registry import register_dataset_factory

logger = logging.getLogger(__name__)


@register_dataset_factory("gaia")
def create_gaia_dataset(
    cfg: DictConfig, dataset_cfg: DictConfig, system_of_bodies: env.SystemOfBodies
) -> tuple[obs.ObservationCollection, obs_model_setup.model_settings.ObservationModelSettings]:
    """Create a dataset from Gaia astrometric observations of a target body.

    Args:
        cfg: Experiment configuration.
        dataset_cfg: Dataset configuration.  Must contain a ``source_id`` (or
            ``mpc_numbers``).
        system_of_bodies: The environment, which supplies the Gaia body's
            ephemeris and the ephemerides the astrometric corrections need.

    Returns:
        Tuple of (ObservationCollection, ObservationModelSettings) for the Gaia
        dataset.
    """
    logger.info(f"Creating Gaia observation dataset: {dataset_cfg.identifier}.")

    target_name = str(getattr(dataset_cfg, "target", "Triton"))

    # The frame is built by the source, which also applies the astrometric
    # corrections - they need ephemerides, so they cannot happen later.  The
    # frame's RA/Dec are still in the archive's orientation (ICRS); the shared
    # conversion at the collection boundary rotates them.
    source = GaiaSource(cfg, system_of_bodies=system_of_bodies)
    frame = source.frame_from_config(dataset_cfg)

    link_ends = {}
    link_ends[obs_model_setup.links.transmitter] = obs_model_setup.links.body_origin_link_end_id(
        target_name
    )
    link_ends[obs_model_setup.links.receiver] = obs_model_setup.links.body_origin_link_end_id(
        "Gaia"
    )
    link_definition = obs_model_setup.links.LinkDefinition(link_ends)

    epochs = frame["epoch_tdb"].to_numpy(dtype="float64")
    ra = frame["first_value"].to_numpy(dtype="float64")
    dec = frame["second_value"].to_numpy(dtype="float64")

    observation_set = obs.create_single_observation_set(
        obs_model_setup.model_settings.angular_position_type,
        link_ends,
        _as_observation_data(ra, dec),
        epochs.tolist(),
        obs_model_setup.links.LinkEndType.receiver,
    )

    _attach_transit_weights(observation_set, source, dataset_cfg)
    _publish_scan_angles(epochs, frame)

    observation_model = obs_model_setup.model_settings.angular_position(link_definition)

    logger.info(
        f"Gaia observation dataset {dataset_cfg.identifier} created with {len(frame)} rows."
    )

    return (observation_set, observation_model)


def _as_observation_data(ra: np.ndarray, dec: np.ndarray) -> list[np.ndarray]:
    """Shape the RA/Dec pair into the ``(2, 1)`` columns Tudat expects."""
    return [np.array([[ra_value], [dec_value]]) for ra_value, dec_value in zip(ra, dec)]


def _attach_transit_weights(observation_set, source: GaiaSource, dataset_cfg) -> None:
    """Attach the archive's full transit weight matrix to the observation set.

    The covariance blocks are per transit and the set's observations are in table
    order, so the blocks concatenate into one block-diagonal matrix over all
    observations.  Each block is inverted and symmetrised, matching what
    ``GaiaAstrometry.to_observation_collection`` does.

    A build without full-matrix support falls back to the diagonal, which is
    degraded but not wrong; the fallback is logged so an unexpectedly
    under-weighted fit is traceable to here.
    """
    blocks = _covariance_blocks(source, dataset_cfg)
    if not blocks:
        logger.warning(
            "Gaia dataset %s: no covariance blocks available; leaving the default "
            "per-observation weights in place.",
            dataset_cfg.get("identifier", "?"),
        )
        return

    matrices = [np.linalg.inv(block) for block in blocks]
    # Force symmetry: the inversion introduces numerical asymmetry, and Tudat's
    # normal equations assume a symmetric weight matrix.
    matrices = [(matrix + matrix.T) / 2.0 for matrix in matrices]

    from scipy.linalg import block_diag

    weight_matrix = block_diag(*matrices)

    # Tudat *adds* the tabulated weights to the full matrix, and the
    # per-observation weights default to one, so they are zeroed first.  Without
    # this every observation carries an extra unit of weight on top of its
    # inverse covariance.
    zero_base_weights = np.zeros(weight_matrix.shape[0])
    try:
        observation_set.set_tabulated_weights(zero_base_weights)
        observation_set.set_full_weight_matrix(weight_matrix)
        logger.info(
            "Gaia dataset %s: attached full transit weight matrix (%d x %d).",
            dataset_cfg.get("identifier", "?"),
            *weight_matrix.shape,
        )
    except (AttributeError, RuntimeError, TypeError) as exc:
        logger.warning(
            "Gaia dataset %s: full weight matrix unavailable (%s); falling back to its diagonal.",
            dataset_cfg.get("identifier", "?"),
            exc,
        )
        observation_set.set_tabulated_weights(np.diag(weight_matrix).copy())


def _covariance_blocks(source: GaiaSource, dataset_cfg) -> list:
    """Return the covariance blocks for the dataset's target object.

    Gaia keys objects by MPC number for asteroids and by ``source_id`` for
    natural satellites, and a query may hold several; this picks the one the
    dataset asked for.
    """
    blocks = source.covariance_blocks
    if not blocks:
        return []

    selection = _selection_keys(dataset_cfg)
    for key in selection:
        if key in blocks:
            return blocks[key]

    # One object in the query: it is the target regardless of how it was keyed.
    if len(blocks) == 1:
        return next(iter(blocks.values()))

    raise ValueError(
        f"Gaia dataset {dataset_cfg.get('identifier', '?')}: no covariance blocks found for "
        f"{selection or 'the configured selection'}; the query holds "
        f"{sorted(blocks.keys())}."
    )


def _selection_keys(dataset_cfg) -> list[int]:
    """Return the object keys the dataset selects, as ints."""
    keys: list[int] = []
    for field in ("source_ids", "mpc_numbers"):
        values = dataset_cfg.get(field)
        if values is None:
            continue
        try:
            keys.extend(int(value) for value in values)
        except (TypeError, ValueError):
            continue
    return keys


def _publish_scan_angles(epochs: np.ndarray, frame) -> None:
    """Publish the frame's scan angles to the epoch-keyed registry.

    The frame is the source of truth, but the scan-frame residual plots read the
    registry in :mod:`orbitdet.data.gaia` because they hold only the observation
    collection and not the frame.  Pushing the frame's values across keeps that
    registry in step without a second archive read.
    """
    if "scan_angle" not in frame.columns:
        return
    angles = frame["scan_angle"].to_numpy(dtype="float64")
    if not np.all(np.isfinite(angles)):
        logger.warning(
            "Gaia: %d scan angle(s) are missing; the scan-frame plots will fall back to "
            "the (RA, Dec) frame for the affected observations.",
            int((~np.isfinite(angles)).sum()),
        )
        return

    from orbitdet.data.gaia import _register_scan_angles

    _register_scan_angles(epochs, angles)

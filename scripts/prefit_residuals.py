"""Pipeline stage: compute the residuals of an observation collection.

This stage does one thing: it evaluates the observation models of an existing
collection against the environment's ephemerides and writes the residuals back
into the collection.  It builds no observations and draws no figures, so a
change to a dataset selection or a figure title never re-runs it.

Because the residuals are computed against whichever ephemeris the environment
holds for the observed body, the same stage serves both the pre-fit and the
post-fit case.  ``observation_source`` selects it:

``spice`` (default)
    The body's ephemeris is the one the dynamics config builds, i.e. SPICE.
    The residuals are those of the *a priori* state.
``propagation``
    A saved ``SingleArcSimulationResults`` is loaded and its state history is
    installed as the observed body's ephemeris before the residuals are
    computed.  The residuals are those of a *propagated* state, which is how the
    post-fit residuals are produced without a second propagation stage.

The source is auto-detected: naming a ``propagation_result`` selects
``propagation``, leaving it unset selects ``spice``.  An explicit ``type``
overrides the detection.

Runnable in three ways:

* standalone, with the default config::

      python scripts/prefit_residuals.py

* standalone, against a different config::

      python scripts/prefit_residuals.py --config-name=experiment/prefit_residuals_toy

* as a Snakemake rule, which additionally pins the input and output paths::

      python scripts/prefit_residuals.py \\
          observations=<collection>/observations.tudat \\
          standard_frame=<collection>/standard_frame.parquet \\
          hydra.run.dir=results/<stage>

Artifacts written into the Hydra run directory:

``observations.tudat``
    The input collection with the residuals populated.  Every figure rule and
    every downstream stage consumes this.
``standard_frame.parquet``
    The input standard frame, copied through unchanged.  Residual computation
    does not touch it, but keeping it beside the collection means the two
    artifacts that rebuild an :class:`~orbitdet.analysis.ObservationAccess`
    always travel together.
``observation_source.json``
    Provenance: which source produced the residuals, and which result file and
    body were used when it was a propagation.

The stage is self-contained: it furnishes SPICE itself, because kernel state is
process-local and cannot be handed to another process.
"""

import json
import logging
import os
import shutil

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tudatpy.astro.time_representation import iso_string_to_epoch_time_object
from tudatpy.dynamics import environment_setup as env_setup
from tudatpy.dynamics import propagation as prop
from tudatpy.estimation import observations as obs
from tudatpy.estimation.observations_setup import observations_simulation_settings as obs_sim_setup
from tudatpy.interface import spice

from orbitdet.data import KernelManager
from orbitdet.observations import rebuild_observation_models
from orbitdet.observations.helpers import add_observatory_to_SOB
from orbitdet.reproducibility import RuntimeContext, enforce_initialization, initialize
from orbitdet.simulation import get_environment
from orbitdet.utility import load_tudat_object, save_tudat_object

logger = logging.getLogger(__name__)

SPICE_SOURCE = "spice"
PROPAGATION_SOURCE = "propagation"


def _resolve_source(cfg: DictConfig) -> tuple[str, str | None]:
    """Return the ``(source type, propagation result path)`` to use.

    An explicit ``observation_source.type`` wins; otherwise naming a
    ``propagation_result`` selects the propagation source and leaving it unset
    selects SPICE.  The path is ``None`` for the SPICE source.
    """
    source_cfg = OmegaConf.select(cfg, "observation_source", default={}) or {}
    result_path = source_cfg.get("propagation_result")
    source_type = source_cfg.get("type")

    if source_type is None:
        source_type = PROPAGATION_SOURCE if result_path else SPICE_SOURCE

    if source_type == PROPAGATION_SOURCE and not result_path:
        raise ValueError(
            "observation_source.type is 'propagation' but no "
            "observation_source.propagation_result was given."
        )
    if source_type not in (SPICE_SOURCE, PROPAGATION_SOURCE):
        raise ValueError(
            f"Unknown observation_source.type '{source_type}'. "
            f"Known: {SPICE_SOURCE}, {PROPAGATION_SOURCE}."
        )

    return source_type, (str(result_path) if result_path else None)


def _propagated_body(cfg: DictConfig) -> str:
    """Return the single body the propagation result describes.

    A single-arc result carries one body's state history, so a config that
    propagates several bodies cannot be resolved to one ephemeris.  The body is
    read from ``bodies_to_propagate`` rather than hard-coded, so the stage stays
    free of body names.
    """
    bodies = list(cfg.bodies_to_propagate.keys())
    if len(bodies) != 1:
        raise ValueError(
            "The propagation source needs exactly one propagated body to install "
            f"as an ephemeris, but bodies_to_propagate lists {bodies}."
        )
    return bodies[0]


def _shift_to_global_origin(cfg: DictConfig, state_history: dict, central_body: str) -> dict:
    """Translate a central-body-relative state history to the global origin.

    A propagation result holds the propagated body's state *with respect to its
    central body*, but the observation models evaluate every body's state with
    respect to the global frame origin.  A tabulated ephemeris is queried
    directly, without the origin translation that ``Body.state`` applies, so the
    history has to be shifted here: adding the central body's own state at each
    epoch is what makes the tabulated ephemeris interchangeable with the SPICE
    one the environment would otherwise hold.

    The central body's state is read from SPICE, which is the same source the
    environment builds it from, so the shift introduces no model of its own.
    """
    if central_body == cfg.global_frame_origin:
        return state_history

    shifted: dict = {}
    for epoch, state in state_history.items():
        offset = spice.get_body_cartesian_state_at_epoch(
            target_body_name=central_body,
            observer_body_name=cfg.global_frame_origin,
            reference_frame_name=cfg.global_frame_orientation,
            aberration_corrections="none",
            ephemeris_time=epoch,
        )
        shifted[epoch] = np.asarray(state).flatten() + np.asarray(offset).flatten()

    logger.info(
        "Shifted %d state(s) from %s to the global origin %s.",
        len(shifted),
        central_body,
        cfg.global_frame_origin,
    )
    return shifted


def _install_propagated_ephemeris(
    cfg: DictConfig, bodies, result_path: str
) -> tuple[str, str, int]:
    """Install a saved propagation result as the observed body's ephemeris.

    The result's ``state_history`` is the propagated body's state with respect to
    its central body, expressed in the global frame orientation - the same
    convention the propagator was given its initial state in.  It is shifted to
    the global frame origin and tabulated, which is what makes the ephemeris
    interchangeable with the SPICE one the environment would otherwise hold.

    Returns:
        ``(body, central_body, number of samples)``.
    """
    body = _propagated_body(cfg)
    central_body = cfg.bodies_to_propagate[body].central_body

    logger.info("Loading propagation result from %s", result_path)
    result = load_tudat_object(prop.SingleArcSimulationResults, result_path)

    state_history = result.state_history
    if not state_history:
        raise ValueError(
            f"The propagation result at {result_path} carries no state history, so "
            "it cannot be installed as an ephemeris. It was probably written with "
            "processing_settings.clear_numerical_solution enabled."
        )

    state_history = _shift_to_global_origin(cfg, state_history, central_body)

    settings = env_setup.ephemeris.tabulated(
        state_history, cfg.global_frame_origin, cfg.global_frame_orientation
    )
    bodies.get(body).ephemeris = env_setup.ephemeris.create_ephemeris(settings, body)

    logger.info(
        "Installed %d propagated state(s) of %s (w.r.t. %s, frame %s) as its ephemeris.",
        len(state_history),
        body,
        central_body,
        cfg.global_frame_orientation,
    )
    return body, central_body, len(state_history)


def _furnish_link_ends(cfg: DictConfig, bodies, collection) -> list[str]:
    """Add every ground station the collection's link ends refer to.

    The environment is rebuilt from the dynamics config, which knows nothing
    about the observatories a collection was built with - those were added while
    the collection was created.  A station that is missing from the environment
    cannot be resolved when the observation models are rebuilt, so the names are
    read back off the collection and added here.

    Returns:
        The station names that were added.
    """
    stations = list(collection.get_reference_points_in_link_ends())
    for station in stations:
        add_observatory_to_SOB(cfg, bodies, station)
    if stations:
        logger.info("Added %d ground station(s) from the collection: %s", len(stations), stations)
    return stations


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="prefit_residuals",
)
@enforce_initialization
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir

    ctx: RuntimeContext = initialize(cfg)

    # Inject the epochs the observation models need.
    ctx.start_epoch = iso_string_to_epoch_time_object(cfg.start_date)
    ctx.end_epoch = iso_string_to_epoch_time_object(cfg.end_date)
    ctx.initial_epoch = iso_string_to_epoch_time_object(cfg.initial_epoch)

    source_type, result_path = _resolve_source(cfg)
    logger.info("Computing residuals with respect to %s.", source_type)

    # Kernel state is process-local, so every stage furnishes SPICE itself.
    logger.info("Loading kernels...")
    km: KernelManager = KernelManager(cfg)
    km.download_all_kernels()
    km.furnish()
    km.download_all_data_files()
    logger.info("Kernels loaded and furnished successfully.")

    bodies = get_environment(cfg, ctx)
    logger.info("Environment built.")

    provenance = {"source": source_type}
    if source_type == PROPAGATION_SOURCE:
        body, central_body, n_samples = _install_propagated_ephemeris(cfg, bodies, result_path)
        provenance.update(
            {
                "propagation_result": result_path,
                "body": body,
                "central_body": central_body,
                "state_history_samples": n_samples,
            }
        )

    logger.info("Loading observation collection from %s", cfg.observations)
    observations = load_tudat_object(obs.ObservationCollection, cfg.observations)
    logger.info(
        "Observation collection loaded with %d observation(s) in %d set(s).",
        len(observations.concatenated_times),
        len(observations.get_single_observation_sets()),
    )

    # The collection carries the observable types, link ends and epochs, but not
    # the model settings - those are not serializable.  Rebuilding them from the
    # collection is what keeps this stage independent of the collection config.
    _furnish_link_ends(cfg, bodies, observations)
    models = rebuild_observation_models(observations)
    logger.info("Rebuilt %d observation model(s) from the collection.", len(models))

    logger.info("Computing residuals with respect to %s...", source_type)
    simulators = obs_sim_setup.create_observation_simulators(models, bodies)
    obs.compute_residuals_and_dependent_variables(observations, simulators, bodies)
    logger.info("Residuals computed successfully.")

    # -- Artifacts -----------------------------------------------------
    observations_path = save_tudat_object(observations, os.path.join(output_dir, "observations"))
    logger.info("Observations saved to %s", observations_path)

    # The frame is not modified here, but it is copied so the two artifacts that
    # rebuild an ObservationAccess always sit together.
    frame_path = os.path.join(output_dir, "standard_frame.parquet")
    shutil.copyfile(cfg.standard_frame, frame_path)
    logger.info("Standard frame copied to %s", frame_path)

    provenance_path = os.path.join(output_dir, "observation_source.json")
    with open(provenance_path, "w") as handle:
        json.dump(provenance, handle, indent=2)
    logger.info("Observation source saved to %s", provenance_path)

    logger.info("prefit_residuals stage completed.")


if __name__ == "__main__":
    main()

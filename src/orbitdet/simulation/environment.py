import logging

import numpy as np
from omegaconf import DictConfig
from tudatpy.dynamics import environment as env
from tudatpy.dynamics import environment_setup as env_setup
from tudatpy.interface import spice

from orbitdet.data.voyager_data import build_voyager_tabulated_state_history
from orbitdet.reproducibility.runtime import RuntimeContext

logger = logging.getLogger(__name__)

DEFAULT_BODIES = {
    "Sun",
    "Mercury",
    "Venus",
    "Earth",
    "Moon",
    "Mars",
    # Major Martian satellites
    "Phobos",
    "Deimos",
    "Jupiter",
    # Major Galilean satellites
    "Io",
    "Europa",
    "Ganymede",
    "Callisto",
    "Saturn",
    # Major Saturnian satellites
    "Mimas",
    "Enceladus",
    "Tethys",
    "Dione",
    "Rhea",
    "Titan",
    "Iapetus",
    "Uranus",
    "Neptune",
    "Triton",
}


def get_environment(cfg: DictConfig, ctx: RuntimeContext) -> env.SystemOfBodies:
    """Factory function to create Environment instance based on configuration."""
    logger.info("Starting environment setup...")

    body_settings = _setup_body_settings_from_config(cfg)
    _configure_body_models(cfg, ctx, body_settings)
    _setup_frame_origin_ephemeris(cfg, body_settings)
    _setup_neptune_default_ephemeris(cfg, body_settings)

    logger.info("Creating system of bodies...")
    bodies = env_setup.create_system_of_bodies(body_settings)

    logger.info("Environment setup completed successfully.")
    return bodies


def _setup_body_settings_from_config(cfg: DictConfig) -> env_setup.BodyListSettings:
    """Initialize body settings from configuration."""
    logger.info("Initializing body settings from configuration...")

    bodies_to_use = set(cfg.bodies_to_create.keys()) & set(DEFAULT_BODIES)
    logger.info(f"Creating {len(bodies_to_use)} default bodies.")

    body_settings = env_setup.get_default_body_settings(
        bodies_to_use,
        cfg.global_frame_origin,
        cfg.global_frame_orientation,
    )

    body_settings.add_empty_settings("Triton")

    # Add bodies not in default bodies but specified in config
    custom_bodies = set(cfg.bodies_to_create.keys()) - DEFAULT_BODIES
    if custom_bodies:
        logger.info(f"Adding {len(custom_bodies)} custom body/bodies: {custom_bodies}")
        for body_name in custom_bodies:
            body_settings.add_empty_settings(body_name)

    return body_settings


def _configure_body_models(
    cfg: DictConfig, ctx: RuntimeContext, body_settings: env_setup.BodyListSettings
) -> None:
    """Configure ephemeris, gravity, rotation, and shape models for all bodies."""
    logger.info(f"Configuring models for {len(cfg.bodies_to_create)} bodies...")

    for body_name, settings in cfg.bodies_to_create.items():
        _configure_ephemeris_model(cfg, ctx, body_name, settings, body_settings)
        _configure_gravity_model(cfg, body_name, settings, body_settings)
        _configure_rotation_model(cfg, body_name, settings, body_settings)
        _configure_shape_model(body_name, settings, body_settings)


def _configure_ephemeris_model(
    cfg: DictConfig,
    ctx: RuntimeContext,
    body_name: str,
    settings: DictConfig,
    body_settings: env_setup.BodyListSettings,
) -> None:
    """Configure ephemeris model for a body."""
    if "ephemeris" not in settings:
        return

    relative_to = (
        settings.ephemeris.relative_to
        if hasattr(settings.ephemeris, "relative_to")
        else cfg.global_frame_origin
    )

    ephemeris_type = settings.ephemeris.type
    logger.info(
        f"Setting ephemeris for {body_name} using {ephemeris_type} (relative to {relative_to})."
    )

    match ephemeris_type:
        case "direct_spice":
            body_settings.get(body_name).ephemeris_settings = env_setup.ephemeris.direct_spice(
                relative_to, cfg.global_frame_orientation
            )
        case "interpolated_spice":
            if not hasattr(settings.ephemeris, "interpolator_cadance"):
                raise ValueError(
                    f"Interpolator cadence must be specified for interpolated_spice "
                    f"ephemeris of {body_name}."
                )
            logger.info(
                f"Using interpolated SPICE for {body_name} with cadence "
                f"{settings.ephemeris.interpolator_cadance}s."
            )
            body_settings.get(
                body_name
            ).ephemeris_settings = env_setup.ephemeris.interpolated_spice(
                ctx.start_epoch - 3000,
                ctx.end_epoch + 3000,
                settings.ephemeris.interpolator_cadance,
                cfg.global_frame_origin,
                cfg.global_frame_orientation,
            )
        case "tabulated_from_ancillary_file":
            dataset_name = str(getattr(settings.ephemeris, "source_dataset", "voyager"))
            voyager_cfg = getattr(getattr(cfg, "datasets", None), dataset_name, None)
            if voyager_cfg is None:
                raise ValueError(
                    f"Voyager dataset configuration '{dataset_name}' is required "
                    f"to build a tabulated ephemeris for {body_name}."
                )
            logger.info(
                f"Building tabulated ephemeris for {body_name} from {dataset_name} dataset."
            )
            state_history = build_voyager_tabulated_state_history(cfg, voyager_cfg)

            relative_to = (
                settings.ephemeris.relative_to
                if hasattr(settings.ephemeris, "relative_to")
                else cfg.global_frame_origin
            )
            body_settings.get(body_name).ephemeris_settings = env_setup.ephemeris.tabulated(
                state_history,
                relative_to,
                cfg.global_frame_orientation,
            )
        case _:
            raise ValueError(f"Unsupported ephemeris type for {body_name}: {ephemeris_type}")


def _configure_gravity_model(
    cfg: DictConfig, body_name: str, settings: DictConfig, body_settings: env_setup.BodyListSettings
) -> None:
    """Configure gravity model for a body."""
    if "gravity" not in settings:
        return

    gravity_type = settings.gravity
    logger.info(f"Setting gravity model for {body_name}: {gravity_type}")

    match gravity_type:
        case "Jacobson2009":
            if body_name != "Neptune":
                raise ValueError(
                    f"Jacobson2009 gravity model is only supported for Neptune, not {body_name}."
                )
            logger.info("Configuring Jacobson 2009 spherical harmonic gravity field for Neptune.")
            _setup_gravity_neptune_jacobson2009(cfg, body_settings)
        case "central":
            logger.debug(f"Using default central gravity for {body_name}.")
        case _:
            raise ValueError(f"Unsupported gravity model for {body_name}: {gravity_type}")


def _configure_rotation_model(
    cfg: DictConfig, body_name: str, settings: DictConfig, body_settings: env_setup.BodyListSettings
) -> None:
    """Configure rotation model for a body."""
    if "rotation_model" not in settings:
        return

    original_frame = cfg.global_frame_orientation
    rotation_model_type = settings.rotation_model.type
    logger.info(f"Setting rotation model for {body_name}: {rotation_model_type}")

    match rotation_model_type:
        case "GCRS_to_ITRS":
            if not hasattr(settings.rotation_model, "nutation_model"):
                raise ValueError(
                    f"Nutation model must be specified for GCRS_to_ITRS rotation "
                    f"model of {body_name}."
                )
            nutation_model = settings.rotation_model.nutation_model
            match nutation_model:
                case "iau_2006":
                    logger.info(
                        f"Configuring GCRS_to_ITRS rotation with IAU 2006 nutation for {body_name}."
                    )
                    precession_nutation_theory = env_setup.rotation_model.IAUConventions.iau_2006
                case _:
                    raise ValueError(
                        f"Unsupported nutation model for {body_name}: {nutation_model}"
                    )
            body_settings.get(
                body_name
            ).rotation_model_settings = env_setup.rotation_model.gcrs_to_itrs(
                precession_nutation_theory, cfg.global_frame_orientation
            )
        case "simple_from_spice":
            if body_name != "Neptune":
                raise ValueError(
                    f"simple_from_spice rotation model is only supported for Neptune, "
                    f"not {body_name}."
                )
            logger.info("Configuring simple SPICE rotation model for Neptune.")
            body_settings.get(
                body_name
            ).rotation_model_settings = env_setup.rotation_model.simple_from_spice(
                original_frame, "IAU_Neptune", "IAU_Neptune", cfg.start_epoch
            )
        case "spice":
            logger.warning(
                f"Skipping explicit SPICE rotation model setup for {body_name}; "
                f"assuming it has been configured in get_default_body_settings."
            )
        case "IAU2015":
            if body_name != "Neptune":
                raise ValueError(
                    f"IAU2015 rotation model is only supported for Neptune, not {body_name}."
                )
            logger.info("Configuring IAU2015 rotation model for Neptune.")
            _setup_rotation_neptune_iau2015(cfg, body_settings, original_frame)
        case "Pole_Model_Jacobson2009":
            if body_name != "Neptune":
                raise ValueError(
                    f"Pole_Model_Jacobson2009 rotation model is only supported for Neptune, "
                    f"not {body_name}."
                )
            logger.info("Configuring Jacobson 2009 pole rotation model for Neptune.")
            _setup_rotation_neptune_jacobson2009(cfg, body_settings, original_frame)
        case _:
            raise ValueError(
                f"Unsupported rotation model type for {body_name}: {rotation_model_type}"
            )


def _configure_shape_model(
    body_name: str, settings: DictConfig, body_settings: env_setup.BodyListSettings
) -> None:
    """Configure shape model for a body."""
    if "shape_model" not in settings:
        return

    shape_model_type = settings.shape_model
    logger.info(f"Setting shape model for {body_name}: {shape_model_type}")

    match shape_model_type:
        case "oblate_spherical_spice":
            logger.info(f"Configuring oblate spherical shape model from SPICE for {body_name}.")
            body_settings.get(body_name).shape_settings = env_setup.shape.oblate_spherical_spice()
        case _:
            raise ValueError(f"Unsupported shape model for {body_name}: {shape_model_type}")


def _setup_frame_origin_ephemeris(
    cfg: DictConfig, body_settings: env_setup.BodyListSettings
) -> None:
    """Setup ephemeris for the global frame origin if needed."""
    if cfg.global_frame_origin == "SSB" or cfg.global_frame_origin not in cfg.bodies_to_create:
        return

    origin_settings = body_settings.get(cfg.global_frame_origin)
    if getattr(origin_settings, "ephemeris_settings", None) is not None:
        logger.debug(f"Ephemeris already configured for frame origin {cfg.global_frame_origin}.")
        return

    logger.info(
        f"Setting ephemeris for global frame origin {cfg.global_frame_origin} "
        f"using direct SPICE from SSB."
    )
    origin_settings.ephemeris_settings = env_setup.ephemeris.direct_spice(
        "SSB",
        cfg.global_frame_orientation,
        cfg.global_frame_origin,
    )


def _setup_neptune_default_ephemeris(
    cfg: DictConfig, body_settings: env_setup.BodyListSettings
) -> None:
    """Setup default ephemeris for Neptune if not already configured."""
    if "Neptune" not in cfg.bodies_to_create:
        return

    neptune_settings = body_settings.get("Neptune")
    if getattr(neptune_settings, "ephemeris_settings", None) is not None:
        logger.debug("Neptune ephemeris already configured.")
        return

    logger.info(
        f"Setting default ephemeris for Neptune using direct SPICE "
        f"(relative to {cfg.global_frame_origin})."
    )
    neptune_settings.ephemeris_settings = env_setup.ephemeris.direct_spice(
        cfg.global_frame_origin, cfg.global_frame_orientation
    )


def _setup_gravity_neptune_jacobson2009(
    cfg: DictConfig, body_settings: env_setup.BodyListSettings
) -> None:
    """Setup Neptune's Jacobson 2009 gravity field model."""
    # copied from Atanas Dzhurkov (2026)
    # Define spherical harmonics from Jacobson 2009
    J2 = 3408.428530717952e-6
    J4 = -33.398917590066e-6
    C20 = -J2 / np.sqrt(5.0)  # C̄20 = -J2 / sqrt(2*2+1)
    C40 = -J4 / 3.0  # C̄40 = -J4 / sqrt(2*4+1) = -J4/3

    # Build coefficient matrices (normalized)
    l_max = 4
    Cbar = np.zeros((l_max + 1, l_max + 1))
    Sbar = np.zeros_like(Cbar)
    Cbar[2, 0] = C20
    Cbar[4, 0] = C40

    # Get GM and radius from SPICE
    mu_N = spice.get_body_gravitational_parameter("Neptune")
    radii_km = spice.get_body_properties("Neptune", "RADII", 3)
    R_eq = radii_km[0] * 1e3  # meters (use equatorial as reference radius)

    body_settings.get(
        "Neptune"
    ).gravity_field_settings = env_setup.gravity_field.spherical_harmonic(
        gravitational_parameter=mu_N,
        reference_radius=R_eq,
        normalized_cosine_coefficients=Cbar,
        normalized_sine_coefficients=Sbar,
        associated_reference_frame="IAU_Neptune",
    )


def _setup_rotation_neptune_iau2015(
    cfg: DictConfig,
    body_settings: env_setup.BodyListSettings,
    original_frame: str,
) -> None:
    """Setup Neptune's IAU2015 rotation model."""
    target_frame = "IAU_Neptune"
    # Copied from Atanas Dzhurkov (2026), adapted to fit Hydra configuration
    nominal_meridian = np.deg2rad(249.978)  # W_0
    nominal_pole = np.deg2rad(np.array([299.36, 43.46]))  # alpha_0 and delta_0
    rotation_rate = np.deg2rad(
        541.1397757 / 24 / 3600
    )  # W_0_dot (in paper it's multipled by day so /24/3600 should align with tudat check!)
    pole_precession = np.array([0, 0])  # alpha_0_dot and delta_0_dot are 0
    merdian_periodic_terms = {
        np.deg2rad(52.316 / 36525 / 24 / 3600): (
            np.deg2rad(-0.48),
            np.deg2rad(357.85),
        )
    }  # w_N_i, W_i, Phi_N_i in that order

    # Values for alpha and delta from IAU 2015
    w_n_i = np.deg2rad(52.316 / 36525 / 24 / 3600)
    alpha_i = np.deg2rad(0.7)
    delta_i = np.deg2rad(-0.51)
    phi = np.deg2rad(357.85)

    # Create the numpy array for [alpha_i, delta_i] as a 2x1 column vector
    alpha_delta = np.array([alpha_i, delta_i])

    if "initial_Pole_Pos" in cfg.bodies_to_create["Neptune"]:
        nominal_pole = cfg.bodies_to_create["Neptune"]["initial_Pole_Pos"]
    if "initial_Pole_lib_deg1" in cfg.bodies_to_create["Neptune"]:
        alpha_delta = cfg.bodies_to_create["Neptune"]["initial_Pole_lib_deg1"]

    # Create the dictionary
    data = {w_n_i: (alpha_delta, phi)}
    pole_periodic_terms = data

    body_settings.get(
        "Neptune"
    ).rotation_model_settings = env_setup.rotation_model.iau_rotation_model(
        original_frame,
        target_frame,
        nominal_meridian,
        nominal_pole,
        rotation_rate,
        pole_precession,
        merdian_periodic_terms,
        pole_periodic_terms,
    )


def _setup_rotation_neptune_jacobson2009(
    cfg: DictConfig,
    body_settings: env_setup.BodyListSettings,
    original_frame: str,
) -> None:
    """Setup Neptune's Jacobson 2009 pole rotation model."""
    target_frame = "IAU_Neptune"
    # Copied from Atanas Dzhurkov (2026), adapted to fit Hydra
    alpha_r = np.deg2rad(
        299.4608612607558
    )  # as defined by Jacbson 2009 (check paper for uncertanties)
    delta_r = np.deg2rad(
        43.4048107907141
    )  # as defined by Jacbson 2009 (check paper for uncertanties)
    epsilon = np.deg2rad(0.4616274249865)
    omega_0 = np.deg2rad(352.1753923868973)  # 1989 August 25 needs to be adjusted to J2000
    omega_dot = np.deg2rad(52.3836218446110 / 36525 / 24 / 3600)  # rad/sec

    t0 = np.datetime64("1989-08-25T00:00:00")
    t1 = np.datetime64("2000-01-01T12:00:00")
    seconds_to_J2000 = (t1 - t0) / np.timedelta64(1, "s")
    omega_0 = omega_0 + omega_dot * seconds_to_J2000  # adjusted for J2000 !!

    alpha_0 = alpha_r
    alpha_1 = epsilon * (1 / np.cos(delta_r))
    alpha_2 = -1 / 2 * epsilon**2 * np.tan(delta_r) / np.cos(delta_r)

    delta_0 = delta_r - 1 / 4 * epsilon**2 * np.tan(delta_r)
    delta_1 = -epsilon
    delta_2 = 1 / 4 * epsilon**2 * np.tan(delta_r)

    alpha_delta_1 = np.array([alpha_1, delta_1])
    alpha_delta_2 = np.array([alpha_2, delta_2])

    # Order in a way Tudat accepts
    nominal_meridian = np.deg2rad(249.978)  # W_0 from IAU (not relevant for this study)
    nominal_pole = np.array([alpha_0, delta_0])  # alpha_0 and delta_0
    rotation_rate = np.deg2rad(
        541.1397757 / 24 / 3600
    )  # W_0_dot from IAU (not relevant for this study)
    pole_precession = np.array([0, 0])  # alpha_0_dot and delta_0_dot are 0
    merdian_periodic_terms = {
        np.deg2rad(52.316 / 36525 / 24 / 3600): (
            np.deg2rad(-0.48),
            np.deg2rad(357.85),
        )
    }  # w_N_i, W_i, Phi_N_i in that order from IAU (not relevant)

    data = {
        omega_dot: (alpha_delta_1, omega_0),
        2 * omega_dot: (alpha_delta_2, 2 * omega_0),
    }
    pole_periodic_terms = data

    # Assuming Jacboson 2009 is in J2000 frame and not ECLIPJ2000
    body_settings.get(
        "Neptune"
    ).rotation_model_settings = env_setup.rotation_model.iau_rotation_model(
        original_frame,
        target_frame,
        nominal_meridian,
        nominal_pole,
        rotation_rate,
        pole_precession,
        merdian_periodic_terms,
        pole_periodic_terms,
    )

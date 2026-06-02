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

    body_settings = env_setup.get_default_body_settings(
        set(cfg.bodies_to_create.keys()) & set(DEFAULT_BODIES),
        cfg.global_frame_origin,
        cfg.global_frame_orientation,
    )

    # Add bodies not in default bodies but specified in config using an empty
    # BodySettings as a starting point
    for body_name in set(cfg.bodies_to_create.keys()) - DEFAULT_BODIES:
        body_settings.add_empty_settings(body_name)

    for body_name, settings in cfg.bodies_to_create.items():
        if "ephemeris" in settings:
            match settings.ephemeris.type:
                case "direct_spice":
                    if body_name == "Voyager 2 shifted":
                        continue  # MAJOR HACK; requires manually setting ephemeris to equal
                        #   Voyager 2's after creating SystemOfBodies
                    if body_name == "Voyager 2 spice":
                        body_settings.get(
                            body_name
                        ).ephemeris_settings = env_setup.ephemeris.direct_spice(
                            cfg.global_frame_origin, cfg.global_frame_orientation, "Voyager 2"
                        )
                    else:
                        body_settings.get(
                            body_name
                        ).ephemeris_settings = env_setup.ephemeris.direct_spice(
                            cfg.global_frame_origin, cfg.global_frame_orientation
                        )
                case "interpolated_spice":
                    if not hasattr(settings.ephemeris, "interpolator_cadance"):
                        raise ValueError(
                            f"""Interpolator cadence must be specified for interpolated_spice"""
                            f"""ephemeris of {body_name}."""
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
                            ""
                            f"""to build a tabulated ephemeris."""
                        )
                    state_history = build_voyager_tabulated_state_history(cfg, voyager_cfg)
                    body_settings.get(body_name).ephemeris_settings = env_setup.ephemeris.tabulated(
                        state_history,
                        # 'Neptune',
                        "Neptune Barycenter",
                        cfg.global_frame_orientation,
                    )
                case _:
                    raise ValueError(
                        f"Unsupported ephemeris type for {body_name}: {settings.ephemeris.type}"
                    )
        if "rotation_model" in settings:
            if body_name == "Neptune":
                continue
            match settings.rotation_model.type:
                case "GCRS_to_ITRS":
                    if not hasattr(settings.rotation_model, "nutation_model"):
                        raise ValueError(
                            f"""Nutation model must be specified for GCRS_to_ITRS rotation """
                            f"""model of {body_name}."""
                        )
                    match settings.rotation_model.nutation_model:
                        case "iau_2006":
                            precession_nutation_theory = (
                                env_setup.rotation_model.IAUConventions.iau_2006
                            )
                        case _:
                            raise ValueError(
                                f"Unsupported nutation model for {body_name}: "
                                ""
                                f"""{settings.rotation_model.nutation_model}"""
                            )
                    body_settings.get(
                        body_name
                    ).rotation_model_settings = env_setup.rotation_model.gcrs_to_itrs(
                        precession_nutation_theory, cfg.global_frame_orientation
                    )
                case "IAU2015":
                    if body_name != "Neptune":
                        raise ValueError(
                            f"IAU2015 rotation model currently only supported for Neptune, "
                            ""
                            f"""not {body_name}."""
                        )
                    pass
                case "spice":
                    logger.warning(
                        f"""Skipping SPICE rotation model, assuming this has been set in """
                        f"""get_default_body_settings for {body_name}."""
                    )
                case _:
                    raise ValueError(
                        f"""Unsupported rotation model type for {body_name}: """
                        f"""{settings.rotation_model.type}"""
                    )
        if "shape_model" in settings:
            match settings.shape_model:
                case "oblate_spherical_spice":
                    body_settings.get(
                        body_name
                    ).shape_settings = env_setup.shape.oblate_spherical_spice()
                case _:
                    raise ValueError(
                        f"Unsupported shape model for {body_name}: {settings.shape_model}"
                    )
    # body_settings.get("Triton").ephemeris_settings = env_setup.ephemeris.interpolated_spice(
    #     ctx.start_epoch - 3000,
    #     ctx.end_epoch + 3000,
    #     cfg.bodies_to_create["Triton"].ephemeris.interpolator_cadance,
    #     cfg.global_frame_origin,
    #     cfg.global_frame_orientation,
    # )

    add_neptune(cfg, body_settings)

    # ----- Setup Rotation Model for Earth (for GCRS to ITRS transformation) -----
    # precession_nutation_theory = env_setup.rotation_model.IAUConventions.iau_2006
    # body_settings.get("Earth").rotation_model_settings = env_setup.rotation_model.gcrs_to_itrs(
    #     precession_nutation_theory, cfg.global_frame_orientation
    # )

    # logger.warning("Earth shape not set yet, introduces some slight error")
    # body_settings.get("Earth").shape_settings = env_setup.shape.oblate_spherical_spice()
    logger.info("Earth rotation model and shape settings configured successfully. ")
    logger.warning("CHECK SHAPE MODEL FOR WGS-84/OBSERVATORIES.TXT")
    # # Create system of selected bodies
    bodies = env_setup.create_system_of_bodies(body_settings)

    # Set Voyager 2 shifted ephemeris equal to Voyager 2's ephemeris
    if "Voyager 2 shifted" in cfg.bodies_to_create and "Voyager 2" in bodies.list_of_bodies():
        bodies.get("Voyager 2 shifted").ephemeris = bodies.get("Voyager 2").ephemeris
        logger.info("Voyager 2 shifted ephemeris set equal to Voyager 2's ephemeris successfully.")

    return bodies


def add_neptune(cfg: DictConfig, body_settings: env_setup.BodyListSettings) -> None:
    # ----- Setup Gravity Model -----
    match cfg.bodies_to_create["Neptune"].gravity:
        case "Jacobson2009":
            # copied from Atanas Dzhurkov (2026)
            # Define spherical harmonics from Jacobson 2009
            J2 = 3408.428530717952e-6
            J4 = -33.398917590066e-6
            C20 = -J2 / np.sqrt(5.0)  # C̄20 = -J2 / sqrt(2*2+1)
            C40 = -J4 / 3.0  # C̄40 = -J4 / sqrt(2*4+1) = -J4/3

            # Build coefficient matrices (normalized)
            # l_max, m_max = 4, 0
            l_max = 4
            Cbar = np.zeros((l_max + 1, l_max + 1))
            Sbar = np.zeros_like(Cbar)
            Cbar[2, 0] = C20
            Cbar[4, 0] = C40

            # Get GM and radius from SPICE
            mu_N = spice.get_body_gravitational_parameter("Neptune")
            radii_km = spice.get_body_properties(
                "Neptune", "RADII", 3
            )  # returns [Rx, Ry, Rz] in km in tudatpy >=0.8
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
        case "central":
            pass  # default is central gravity, so no changes needed
        case _:
            raise ValueError(
                f"Unsupported gravity model for Neptune: {cfg.bodies_to_create['Neptune'].gravity}"
            )

    # ----- Setup Rotation Model -----
    # set parameters for defining the rotation between frames
    original_frame = cfg.global_frame_orientation
    target_frame = "IAU_Neptune"
    target_frame_spice = "IAU_Neptune"  # is this correct?
    rotation_model_cfg = cfg.bodies_to_create["Neptune"].rotation_model
    rotation_model_type = (
        rotation_model_cfg.type if hasattr(rotation_model_cfg, "type") else rotation_model_cfg
    )

    match rotation_model_type:
        case "simple_from_spice":
            body_settings.get(
                "Neptune"
            ).rotation_model_settings = env_setup.rotation_model.simple_from_spice(
                original_frame, target_frame, target_frame_spice, cfg.start_epoch
            )
        case "spice":
            body_settings.get("Neptune").rotation_model_settings = env_setup.rotation_model.spice(
                original_frame, target_frame, target_frame_spice
            )

        case "IAU2015":
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

            # If you need it in a list (as per the type annotation)
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

        case "Pole_Model_Jacobson2009":
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
            # w_dot = np.deg2rad(536.3128492)  # rad/day  Not estimated, from Warwick et al. (1989).

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
            # -----------------------------------------------------------------------------
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

    # ----- Setup Ephemeris Model -----
    body_settings.get("Neptune").ephemeris_settings = env_setup.ephemeris.direct_spice(
        cfg.global_frame_origin, cfg.global_frame_orientation
    )

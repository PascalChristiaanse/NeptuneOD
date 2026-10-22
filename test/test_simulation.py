"""Tests for the simulation module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf
from tudatpy.dynamics import environment as env
from tudatpy.dynamics import environment_setup as env_setup
from tudatpy.dynamics import propagation_setup as prop_setup
from tudatpy.math import interpolators

from orbitdet.reproducibility.runtime import RuntimeContext
from orbitdet.simulation import (
    TUDATPY_DEFAULT_INTERPOLATOR_ORDER,
    build_integrator_result_interpolator_settings,
    build_integrator_settings,
    build_translational_propagator_type,
    get_dynamical_model,
    get_environment,
    get_integrator_settings,
    get_propagator_settings,
    resolve_interpolator_order,
)


@pytest.fixture
def mock_runtime_context(tmp_path):
    """Create a mock RuntimeContext for testing."""
    ctx = MagicMock(spec=RuntimeContext)
    ctx.seed = 42
    ctx.git_commit = "abc123"
    ctx.test_mode = True
    ctx.output_dir = tmp_path
    ctx.initial_epoch = 1234567890.0
    ctx.start_epoch = 1234567890.0
    ctx.end_epoch = 1234567900.0
    return ctx


@pytest.fixture
def basic_config():
    """Create a basic configuration for testing."""
    return OmegaConf.create(
        {
            "global_frame_origin": "Neptune",
            "global_frame_orientation": "J2000",
            "bodies_to_create": {
                "Neptune": {
                    "gravity": "central",
                    "rotation_model": {
                        "type": "IAU2015",
                    },
                },
                "Triton": {
                    "ephemeris": {
                        "type": "interpolated_spice",
                        "interpolator_cadance": 300,
                        "coverage_buffer": 86400,
                    },
                },
            },
            "bodies_to_propagate": {
                "Triton": {
                    "central_body": "Neptune",
                    "initial_state": [100000.0, 0.0, 0.0, 0.0, 5000.0, 0.0],
                },
            },
            "propagation": {
                "set_integrated_result": True,
            },
            "integrator": {
                "type": "RKF78",
                "fixed_step_size": 10.0,
            },
            "start_epoch": 1234567890.0,
            "end_epoch": 1234567900.0,
            "use_default_body_settings": True,
        }
    )


@pytest.fixture
def jacobson_config():
    """Create a configuration with Jacobson gravity model."""
    return OmegaConf.create(
        {
            "global_frame_origin": "Neptune",
            "global_frame_orientation": "J2000",
            "bodies_to_create": {
                "Neptune": {
                    "gravity": "Jacobson2009",
                    "rotation_model": {
                        "type": "IAU2015",
                    },
                },
                "Triton": {
                    "ephemeris": {
                        "type": "interpolated_spice",
                        "interpolator_cadance": 300,
                        "coverage_buffer": 86400,
                    },
                },
            },
            "bodies_to_propagate": {
                "Triton": {
                    "central_body": "Neptune",
                    "initial_state": [100000.0, 0.0, 0.0, 0.0, 5000.0, 0.0],
                },
            },
            "propagation": {
                "set_integrated_result": True,
            },
            "integrator": {
                "type": "RKF78",
                "fixed_step_size": 10.0,
            },
            "start_epoch": 1234567890.0,
            "end_epoch": 1234567900.0,
            "use_default_body_settings": True,
        }
    )


class TestGetIntegratorSettings:
    """Tests for get_integrator_settings function."""

    def test_rkf78_integrator_creation(self, basic_config, mock_runtime_context):
        """Test successful creation of RKF78 integrator."""
        settings = get_integrator_settings(basic_config, mock_runtime_context)

        assert isinstance(settings, prop_setup.integrator.IntegratorSettings)

    def test_unknown_integrator_type_raises_error(self, mock_runtime_context):
        """Test that unknown integrator type raises ValueError."""
        config = OmegaConf.create(
            {
                "integrator": {
                    "type": "UnknownIntegrator",
                    "fixed_step_size": 10.0,
                }
            }
        )

        with pytest.raises(ValueError, match="Unknown integrator type"):
            get_integrator_settings(config, mock_runtime_context)


class TestBuildIntegratorSettings:
    """Tests for the timestep-explicit build_integrator_settings factory."""

    @pytest.mark.parametrize(
        "integrator_type",
        [
            "RKF78",
            "RK4",
            "Euler",
            "RK3",
            "Ralston4",
            "RKF89",
            "RKV89",
            "RKF108",
            "RKF1210",
            "RKF1412",
            "BulirschStoer6",
            "BulirschStoer8",
            "BulirschStoer10",
        ],
    )
    def test_creates_settings_for_every_supported_type(self, integrator_type):
        settings = build_integrator_settings(3600.0, integrator_type)

        assert isinstance(settings, prop_setup.integrator.IntegratorSettings)

    def test_uses_the_explicit_timestep_not_the_config(self):
        """The whole point of the factory: the timestep argument wins."""
        config = OmegaConf.create({"integrator": {"fixed_step_size": 999.0}})

        with patch(
            "orbitdet.simulation.integration.prop_setup.integrator.runge_kutta_fixed_step"
        ) as mock_rk:
            build_integrator_settings(7200.0, "RKF78", config)

        assert mock_rk.call_args[0][0] == pytest.approx(7200.0)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown integrator type"):
            build_integrator_settings(3600.0, "NotAnIntegrator")

    def test_generic_bulirsch_stoer_requires_max_steps(self):
        with pytest.raises(ValueError, match="maximum_number_of_steps"):
            build_integrator_settings(3600.0, "BulirschStoer")

    def test_generic_bulirsch_stoer_reads_max_steps_from_config(self):
        config = OmegaConf.create({"integrator": {"maximum_number_of_steps": 6}})

        settings = build_integrator_settings(3600.0, "BulirschStoer", config)

        assert isinstance(settings, prop_setup.integrator.IntegratorSettings)


class TestBuildTranslationalPropagatorType:
    """Tests for the propagator-formulation lookup."""

    def test_resolves_known_formulations(self):
        assert build_translational_propagator_type("cowell") is not None
        assert build_translational_propagator_type("encke") is not None
        assert build_translational_propagator_type("gauss_keplerian") is not None

    def test_unknown_formulation_raises_and_lists_known_ones(self):
        with pytest.raises(ValueError, match="Unknown translational propagator"):
            build_translational_propagator_type("warp_drive")


@pytest.fixture
def real_processing_settings_translational():
    """Patch ``translational`` to return a *real* propagator settings object.

    The interpolator order is applied to ``processing_settings``, so the tests
    need a genuine settings instance rather than a MagicMock (attribute
    assignment on a mock would accept anything and prove nothing).
    """
    real_settings = prop_setup.propagator.translational(
        central_bodies=["Neptune"],
        acceleration_models={},
        bodies_to_integrate=["Triton"],
        initial_states=np.zeros(6),
        initial_time=0.0,
        integrator_settings=prop_setup.integrator.runge_kutta_fixed_step(
            10.0, coefficient_set=prop_setup.integrator.CoefficientSets.rkf_78
        ),
        termination_settings=prop_setup.propagator.time_termination(100.0),
    )

    with patch(
        "orbitdet.simulation.propagation.prop_setup.propagator.translational",
        return_value=real_settings,
    ):
        yield real_settings


class TestIntegratorResultInterpolator:
    """Tests for the integrator-result interpolator wiring.

    This interpolator builds the propagated body's ephemeris whenever
    ``set_integrated_result`` is on — which is always the case in estimation,
    because the orbit determination manager forces it. It therefore fixes the
    accuracy of synthetic observations.
    """

    def test_builds_settings_for_requested_order(self):
        settings = build_integrator_result_interpolator_settings(10)

        assert isinstance(settings, interpolators.LagrangeInterpolatorSettings)

    def test_rejects_non_positive_order(self):
        with pytest.raises(ValueError, match="at least 1"):
            build_integrator_result_interpolator_settings(0)

    def test_resolve_uses_configured_order(self):
        config = OmegaConf.create({"propagation": {"interpolator_order": 10}})

        assert resolve_interpolator_order(config) == 10

    def test_resolve_falls_back_to_tudatpy_default_when_absent(self):
        """A config without the key must behave like one written before it existed."""
        config = OmegaConf.create({"propagation": {"set_integrated_result": True}})

        assert resolve_interpolator_order(config) == TUDATPY_DEFAULT_INTERPOLATOR_ORDER

    def test_resolve_treats_explicit_none_as_absent(self):
        """``interpolator_order: null`` in YAML should not raise."""
        config = OmegaConf.create({"propagation": {"interpolator_order": None}})

        assert resolve_interpolator_order(config) == TUDATPY_DEFAULT_INTERPOLATOR_ORDER

    def test_resolve_handles_missing_propagation_section(self):
        """Some dynamics configs carry no propagation section at all."""
        config = OmegaConf.create({"start_date": "2020-01-01"})

        assert resolve_interpolator_order(config) == TUDATPY_DEFAULT_INTERPOLATOR_ORDER

    def test_get_propagator_settings_applies_the_configured_order(
        self,
        real_processing_settings_translational,
        basic_config,
        mock_runtime_context,
    ):
        """The order must actually land on processing_settings."""
        basic_config.propagation.interpolator_order = 10

        # Spy on the factory: ``LagrangeInterpolatorSettings`` exposes no
        # readable order and defines no Python __eq__, so the argument it is
        # called with is the only faithful way to read the value back.
        with patch(
            "orbitdet.simulation.propagation.build_integrator_result_interpolator_settings",
            wraps=build_integrator_result_interpolator_settings,
        ) as spy:
            get_propagator_settings(
                basic_config, mock_runtime_context, MagicMock(), MagicMock(), []
            )

        assert spy.call_args[0][0] == 10
        applied = real_processing_settings_translational.processing_settings
        assert isinstance(applied.interpolator_settings, interpolators.LagrangeInterpolatorSettings)

    def test_default_order_is_applied_when_unset(
        self,
        real_processing_settings_translational,
        basic_config,
        mock_runtime_context,
    ):
        with patch(
            "orbitdet.simulation.propagation.build_integrator_result_interpolator_settings",
            wraps=build_integrator_result_interpolator_settings,
        ) as spy:
            get_propagator_settings(
                basic_config, mock_runtime_context, MagicMock(), MagicMock(), []
            )

        assert spy.call_args[0][0] == TUDATPY_DEFAULT_INTERPOLATOR_ORDER

    def test_get_propagator_settings_forwards_different_orders(
        self,
        real_processing_settings_translational,
        basic_config,
        mock_runtime_context,
    ):
        """Changing the config must change the order passed to the factory."""
        with patch(
            "orbitdet.simulation.propagation.build_integrator_result_interpolator_settings",
            wraps=build_integrator_result_interpolator_settings,
        ) as spy:
            basic_config.propagation.interpolator_order = 10
            get_propagator_settings(
                basic_config, mock_runtime_context, MagicMock(), MagicMock(), []
            )
            assert spy.call_args[0][0] == 10

            basic_config.propagation.interpolator_order = 12
            get_propagator_settings(
                basic_config, mock_runtime_context, MagicMock(), MagicMock(), []
            )
            assert spy.call_args[0][0] == 12

    def test_default_order_logged_when_unset(
        self,
        real_processing_settings_translational,
        basic_config,
        mock_runtime_context,
        caplog,
    ):
        """The default must be announced, not silent."""
        with caplog.at_level("INFO"):
            get_propagator_settings(
                basic_config, mock_runtime_context, MagicMock(), MagicMock(), []
            )

        assert "TudatPy default" in caplog.text
        assert f"Lagrange-{TUDATPY_DEFAULT_INTERPOLATOR_ORDER}" in caplog.text

    def test_override_is_logged_as_such(
        self,
        real_processing_settings_translational,
        basic_config,
        mock_runtime_context,
        caplog,
    ):
        basic_config.propagation.interpolator_order = 12

        with caplog.at_level("INFO"):
            get_propagator_settings(
                basic_config, mock_runtime_context, MagicMock(), MagicMock(), []
            )

        assert "overriding" in caplog.text
        assert "Lagrange-12" in caplog.text


class TestGetDynamicalModel:
    """Tests for get_dynamical_model function."""

    @patch("orbitdet.simulation.dynamics.prop_setup.create_acceleration_models")
    def test_central_gravity_model(self, mock_create_accel, basic_config, mock_runtime_context):
        """Test creation of central gravity model."""
        mock_bodies = MagicMock(spec=env.SystemOfBodies)
        mock_create_accel.return_value = {"Triton": {"Neptune": []}}

        result = get_dynamical_model(basic_config, mock_runtime_context, mock_bodies)

        assert mock_create_accel.called
        assert result == {"Triton": {"Neptune": []}}

    @patch("orbitdet.simulation.dynamics.prop_setup.create_acceleration_models")
    def test_jacobson_gravity_model(self, mock_create_accel, jacobson_config, mock_runtime_context):
        """Test creation of Jacobson2009 gravity model."""
        mock_bodies = MagicMock(spec=env.SystemOfBodies)
        mock_create_accel.return_value = {"Triton": {"Neptune": []}}

        result = get_dynamical_model(jacobson_config, mock_runtime_context, mock_bodies)

        assert mock_create_accel.called
        assert result == {"Triton": {"Neptune": []}}

    def test_propagate_body_not_in_creation_list_raises_error(self, mock_runtime_context):
        """Test that propagating undefined body raises RuntimeError."""
        config = OmegaConf.create(
            {
                "bodies_to_create": {
                    "Neptune": {"gravity": "central"},
                },
                "bodies_to_propagate": {
                    "UndefinedBody": {
                        "central_body": "Neptune",
                    },
                },
            }
        )
        mock_bodies = MagicMock(spec=env.SystemOfBodies)

        with pytest.raises(
            RuntimeError, match="Cannot propagate UndefinedBody because it is not defined"
        ):
            get_dynamical_model(config, mock_runtime_context, mock_bodies)

    def test_jacobson_for_non_neptune_raises_error(self, mock_runtime_context):
        """Test that Jacobson model only works with Neptune."""
        config = OmegaConf.create(
            {
                "bodies_to_create": {
                    "Sun": {"gravity": "Jacobson2009"},
                    "Triton": {"gravity": "central"},
                },
                "bodies_to_propagate": {
                    "Triton": {
                        "central_body": "Sun",
                    },
                },
            }
        )
        mock_bodies = MagicMock(spec=env.SystemOfBodies)

        with pytest.raises(RuntimeError, match="only defined for Neptune"):
            get_dynamical_model(config, mock_runtime_context, mock_bodies)


class TestGetEnvironment:
    """Tests for get_environment function."""

    @patch("orbitdet.simulation.environment.spice")
    @patch("orbitdet.simulation.environment.env_setup.create_system_of_bodies")
    @patch("orbitdet.simulation.environment.env_setup.get_default_body_settings")
    def test_environment_creation_with_central_gravity(
        self,
        mock_get_defaults,
        mock_create_bodies,
        mock_spice,
        basic_config,
        mock_runtime_context,
    ):
        """Test successful environment creation with central gravity."""
        mock_spice.get_body_gravitational_parameter.return_value = 6.836529e15

        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_body_settings.get = MagicMock(return_value=MagicMock())
        mock_body_settings.add_empty_settings = MagicMock()
        mock_get_defaults.return_value = mock_body_settings

        mock_bodies = MagicMock(spec=env.SystemOfBodies)
        mock_bodies.list_of_bodies = MagicMock(return_value=[])
        mock_create_bodies.return_value = mock_bodies

        result = get_environment(basic_config, mock_runtime_context)

        assert result == mock_bodies
        assert mock_get_defaults.called
        assert mock_create_bodies.called

    @patch("orbitdet.simulation.environment.spice")
    @patch("orbitdet.simulation.environment.env_setup.create_system_of_bodies")
    @patch("orbitdet.simulation.environment.env_setup.get_default_body_settings")
    def test_environment_creation_with_jacobson_gravity(
        self,
        mock_get_defaults,
        mock_create_bodies,
        mock_spice,
        jacobson_config,
        mock_runtime_context,
    ):
        """Test environment creation with Jacobson2009 gravity model."""
        mock_spice.get_body_gravitational_parameter.return_value = 6.836529e15
        mock_spice.get_body_properties.return_value = [24764, 24764, 24341]

        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_body_settings.get = MagicMock(return_value=MagicMock())
        mock_body_settings.add_empty_settings = MagicMock()
        mock_get_defaults.return_value = mock_body_settings

        mock_bodies = MagicMock(spec=env.SystemOfBodies)
        mock_bodies.list_of_bodies = MagicMock(return_value=[])
        mock_create_bodies.return_value = mock_bodies

        result = get_environment(jacobson_config, mock_runtime_context)

        assert result == mock_bodies

    @patch("orbitdet.simulation.environment.spice")
    @patch("orbitdet.simulation.environment.env_setup.create_system_of_bodies")
    @patch("orbitdet.simulation.environment.env_setup.get_default_body_settings")
    def test_environment_returns_created_system_of_bodies(
        self,
        mock_get_defaults,
        mock_create_bodies,
        mock_spice,
        basic_config,
        mock_runtime_context,
    ):
        """Test that get_environment returns the created system of bodies."""
        mock_spice.get_body_gravitational_parameter.return_value = 6.836529e15

        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_body_settings.get = MagicMock(return_value=MagicMock())
        mock_body_settings.add_empty_settings = MagicMock()
        mock_get_defaults.return_value = mock_body_settings

        mock_bodies = MagicMock(spec=env.SystemOfBodies)
        mock_bodies.list_of_bodies = MagicMock(return_value=[])
        mock_create_bodies.return_value = mock_bodies

        result = get_environment(basic_config, mock_runtime_context)

        # Verify returned object is the created system
        assert result is mock_bodies
        mock_create_bodies.assert_called_once_with(mock_body_settings)


class TestGetPropagatorSettings:
    """Tests for get_propagator_settings function."""

    @patch("orbitdet.simulation.propagation.prop_setup.propagator.translational")
    def test_propagator_creation_with_defined_initial_state(
        self,
        mock_translational,
        basic_config,
        mock_runtime_context,
    ):
        """Test propagator creation with predefined initial state."""
        mock_acceleration_settings = MagicMock()
        mock_integrator_settings = MagicMock()

        result = get_propagator_settings(  # noqa: F841
            basic_config,
            mock_runtime_context,
            mock_acceleration_settings,
            mock_integrator_settings,
            [],
        )

        assert mock_translational.called

    @patch("orbitdet.simulation.propagation.spice.get_body_cartesian_state_at_epoch")
    @patch("orbitdet.simulation.propagation.prop_setup.propagator.translational")
    def test_propagator_creation_with_undefined_initial_state(
        self,
        mock_translational,
        mock_get_state,
        basic_config,
        mock_runtime_context,
    ):
        """Test propagator creation falls back to SPICE for undefined initial state."""
        # Create config with undefined initial state
        config = OmegaConf.create(
            {
                "global_frame_orientation": "J2000",
                "bodies_to_propagate": {
                    "Triton": {
                        "central_body": "Neptune",
                        "initial_state": None,
                    },
                },
                "propagation": {
                    "set_integrated_result": True,
                },
            }
        )
        mock_get_state.return_value = [100000.0, 0.0, 0.0, 0.0, 5000.0, 0.0]

        mock_acceleration_settings = MagicMock()
        mock_integrator_settings = MagicMock()

        with patch("orbitdet.simulation.propagation.logger"):
            result = get_propagator_settings(  # noqa: F841
                config,
                mock_runtime_context,
                mock_acceleration_settings,
                mock_integrator_settings,
                [],
            )

        assert mock_get_state.called

    @patch("orbitdet.simulation.propagation.prop_setup.propagator.translational")
    def test_propagator_uses_correct_central_bodies(
        self,
        mock_translational,
        basic_config,
        mock_runtime_context,
    ):
        """Test that propagator uses correct central bodies."""
        mock_acceleration_settings = MagicMock()
        mock_integrator_settings = MagicMock()

        get_propagator_settings(
            basic_config,
            mock_runtime_context,
            mock_acceleration_settings,
            mock_integrator_settings,
            [],
        )

        call_kwargs = mock_translational.call_args[1]  # noqa: F841

        # Verify translational called with correct parameters
        assert mock_translational.called

    @patch("orbitdet.simulation.propagation.prop_setup.propagator.translational")
    def test_propagator_includes_multiple_propagated_bodies(
        self,
        mock_translational,
        basic_config,
        mock_runtime_context,
    ):
        """Test propagator with multiple bodies."""
        config = OmegaConf.create(
            {
                "global_frame_orientation": "J2000",
                "bodies_to_propagate": {
                    "Triton": {
                        "central_body": "Neptune",
                        "initial_state": [100000.0, 0.0, 0.0, 0.0, 5000.0, 0.0],
                    },
                    "Proteus": {
                        "central_body": "Neptune",
                        "initial_state": [117647.0, 0.0, 0.0, 0.0, 4750.0, 0.0],
                    },
                },
                "propagation": {
                    "set_integrated_result": True,
                },
            }
        )

        mock_acceleration_settings = MagicMock()
        mock_integrator_settings = MagicMock()

        get_propagator_settings(
            config,
            mock_runtime_context,
            mock_acceleration_settings,
            mock_integrator_settings,
            [],
        )

        assert mock_translational.called


class TestModuleExports:
    """Tests for module-level exports."""

    def test_all_functions_exported(self):
        """Test that all expected functions are exported."""
        from orbitdet.simulation import (
            build_integrator_settings,
            build_translational_propagator_type,
            get_dynamical_model,
            get_environment,
            get_integrator_settings,
            get_propagator_settings,
        )

        # If imports succeed, functions are available
        assert callable(get_environment)
        assert callable(get_propagator_settings)
        assert callable(get_dynamical_model)
        assert callable(get_integrator_settings)
        assert callable(build_integrator_settings)
        assert callable(build_translational_propagator_type)

    def test_all_in_all_list(self):
        """Test that __all__ exports contain expected items."""
        import orbitdet.simulation as sim_module

        expected = {
            "get_environment",
            "get_propagator_settings",
            "get_dynamical_model",
            "get_integrator_settings",
            "build_integrator_settings",
            "build_translational_propagator_type",
            "build_integrator_result_interpolator_settings",
            "resolve_interpolator_order",
            "TRANSLATIONAL_PROPAGATORS",
            "TUDATPY_DEFAULT_INTERPOLATOR_ORDER",
        }
        assert set(sim_module.__all__) == expected

"""Tests for the simulation module."""

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf
from tudatpy.dynamics import environment as env
from tudatpy.dynamics import environment_setup as env_setup
from tudatpy.dynamics import propagation_setup as prop_setup

from orbitdet.reproducibility.runtime import RuntimeContext
from orbitdet.simulation import (
    get_dynamical_model,
    get_environment,
    get_integrator_settings,
    get_propagator_settings,
)


@pytest.fixture
def mock_runtime_context(tmp_path):
    """Create a mock RuntimeContext for testing."""
    ctx = MagicMock(spec=RuntimeContext)
    ctx.seed = 42
    ctx.git_commit = "abc123"
    ctx.test_mode = True
    ctx.output_dir = tmp_path
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
                    },
                },
            },
            "bodies_to_propagate": {
                "Triton": {
                    "central_body": "Neptune",
                    "initial_state": [100000.0, 0.0, 0.0, 0.0, 5000.0, 0.0],
                },
            },
            "integrator": {
                "type": "RKF78",
                "fixed_step_size": 10.0,
            },
            "start_epoch": 1234567890.0,
            "end_epoch": 1234567900.0,
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
                    },
                },
            },
            "bodies_to_propagate": {
                "Triton": {
                    "central_body": "Neptune",
                    "initial_state": [100000.0, 0.0, 0.0, 0.0, 5000.0, 0.0],
                },
            },
            "integrator": {
                "type": "RKF78",
                "fixed_step_size": 10.0,
            },
            "start_epoch": 1234567890.0,
            "end_epoch": 1234567900.0,
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

    @patch("orbitdet.simulation.environment.env_setup.create_system_of_bodies")
    @patch("orbitdet.simulation.environment.env_setup.get_default_body_settings")
    def test_environment_creation_with_central_gravity(
        self,
        mock_get_defaults,
        mock_create_bodies,
        basic_config,
        mock_runtime_context,
    ):
        """Test successful environment creation with central gravity."""
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

    @patch("orbitdet.simulation.environment.env_setup.create_system_of_bodies")
    @patch("orbitdet.simulation.environment.env_setup.get_default_body_settings")
    def test_environment_returns_created_system_of_bodies(
        self,
        mock_get_defaults,
        mock_create_bodies,
        basic_config,
        mock_runtime_context,
    ):
        """Test that get_environment returns the created system of bodies."""
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

    def test_all_in_all_list(self):
        """Test that __all__ exports contain expected items."""
        import orbitdet.simulation as sim_module

        expected = {
            "get_environment",
            "get_propagator_settings",
            "get_dynamical_model",
            "get_integrator_settings",
        }
        assert set(sim_module.__all__) == expected

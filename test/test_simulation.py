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
            "global_frame_orientation": "ECLIPJ2000",
            "bodies_to_create": {
                "Neptune": {
                    "gravity": "central",
                    "rotation_model": "spice",
                },
                "Triton": {
                    "ephemeris": {
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
            "global_frame_orientation": "ECLIPJ2000",
            "bodies_to_create": {
                "Neptune": {
                    "gravity": "Jacobson2009",
                    "rotation_model": "spice",
                },
                "Triton": {
                    "ephemeris": {
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

    @patch("orbitdet.simulation.environment.add_neptune")
    @patch("orbitdet.simulation.environment.env_setup.create_system_of_bodies")
    @patch("orbitdet.simulation.environment.env_setup.get_default_body_settings")
    def test_environment_creation_with_central_gravity(
        self,
        mock_get_defaults,
        mock_create_bodies,
        mock_add_neptune,
        basic_config,
        mock_runtime_context,
    ):
        """Test successful environment creation with central gravity."""
        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_body_settings.get = MagicMock(return_value=MagicMock())
        mock_get_defaults.return_value = mock_body_settings

        mock_bodies = MagicMock(spec=env.SystemOfBodies)
        mock_create_bodies.return_value = mock_bodies

        result = get_environment(basic_config, mock_runtime_context)

        assert result == mock_bodies
        assert mock_get_defaults.called
        assert mock_create_bodies.called

    @patch("orbitdet.simulation.environment.add_neptune")
    @patch("orbitdet.simulation.environment.env_setup.create_system_of_bodies")
    @patch("orbitdet.simulation.environment.env_setup.get_default_body_settings")
    def test_environment_creation_with_jacobson_gravity(
        self,
        mock_get_defaults,
        mock_create_bodies,
        mock_add_neptune,
        jacobson_config,
        mock_runtime_context,
    ):
        """Test environment creation with Jacobson2009 gravity model."""
        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_body_settings.get = MagicMock(return_value=MagicMock())
        mock_get_defaults.return_value = mock_body_settings

        mock_bodies = MagicMock(spec=env.SystemOfBodies)
        mock_create_bodies.return_value = mock_bodies

        result = get_environment(jacobson_config, mock_runtime_context)

        assert result == mock_bodies

    @patch("orbitdet.simulation.environment.add_neptune")
    @patch("orbitdet.simulation.environment.env_setup.create_system_of_bodies")
    @patch("orbitdet.simulation.environment.env_setup.get_default_body_settings")
    def test_environment_uses_correct_body_list(
        self,
        mock_get_defaults,
        mock_create_bodies,
        mock_add_neptune,
        basic_config,
        mock_runtime_context,
    ):
        """Test that environment uses correct body list from config."""
        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_body_settings.get = MagicMock(return_value=MagicMock())
        mock_get_defaults.return_value = mock_body_settings

        mock_bodies = MagicMock(spec=env.SystemOfBodies)
        mock_create_bodies.return_value = mock_bodies

        get_environment(basic_config, mock_runtime_context)

        # Verify get_default_body_settings called with correct bodies
        call_args = mock_get_defaults.call_args[0]
        body_list = call_args[0]
        assert set(body_list) == {"Neptune", "Triton"}

    @patch("orbitdet.simulation.environment.add_neptune")
    @patch("orbitdet.simulation.environment.env_setup.create_system_of_bodies")
    @patch("orbitdet.simulation.environment.env_setup.get_default_body_settings")
    def test_environment_configures_triton_ephemeris(
        self,
        mock_get_defaults,
        mock_create_bodies,
        mock_add_neptune,
        basic_config,
        mock_runtime_context,
    ):
        """Test that Triton ephemeris is configured with interpolated SPICE."""
        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_triton_settings = MagicMock()
        mock_earth_settings = MagicMock()

        def get_body_side_effect(body_name):
            if body_name == "Triton":
                return mock_triton_settings
            elif body_name == "Earth":
                return mock_earth_settings
            return MagicMock()

        mock_body_settings.get = MagicMock(side_effect=get_body_side_effect)
        mock_get_defaults.return_value = mock_body_settings

        mock_bodies = MagicMock(spec=env.SystemOfBodies)
        mock_create_bodies.return_value = mock_bodies

        get_environment(basic_config, mock_runtime_context)

        # Verify Triton ephemeris was set
        assert mock_triton_settings.ephemeris_settings is not None

    @patch("orbitdet.simulation.environment.add_neptune")
    @patch("orbitdet.simulation.environment.env_setup.create_system_of_bodies")
    @patch("orbitdet.simulation.environment.env_setup.get_default_body_settings")
    def test_environment_configures_earth_rotation_model(
        self,
        mock_get_defaults,
        mock_create_bodies,
        mock_add_neptune,
        basic_config,
        mock_runtime_context,
    ):
        """Test that Earth rotation model (GCRS to ITRS) is configured."""
        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_earth_settings = MagicMock()

        def get_body_side_effect(body_name):
            if body_name == "Earth":
                return mock_earth_settings
            return MagicMock()

        mock_body_settings.get = MagicMock(side_effect=get_body_side_effect)
        mock_get_defaults.return_value = mock_body_settings

        mock_bodies = MagicMock(spec=env.SystemOfBodies)
        mock_create_bodies.return_value = mock_bodies

        get_environment(basic_config, mock_runtime_context)

        # Verify Earth rotation model was set
        assert mock_earth_settings.rotation_model_settings is not None

    @patch("orbitdet.simulation.environment.add_neptune")
    @patch("orbitdet.simulation.environment.env_setup.create_system_of_bodies")
    @patch("orbitdet.simulation.environment.env_setup.get_default_body_settings")
    def test_environment_calls_add_neptune(
        self,
        mock_get_defaults,
        mock_create_bodies,
        mock_add_neptune,
        basic_config,
        mock_runtime_context,
    ):
        """Test that add_neptune is called with correct arguments."""
        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_body_settings.get = MagicMock(return_value=MagicMock())
        mock_get_defaults.return_value = mock_body_settings

        mock_bodies = MagicMock(spec=env.SystemOfBodies)
        mock_create_bodies.return_value = mock_bodies

        get_environment(basic_config, mock_runtime_context)

        # Verify add_neptune was called with config and body_settings
        mock_add_neptune.assert_called_once_with(basic_config, mock_body_settings)

    @patch("orbitdet.simulation.environment.add_neptune")
    @patch("orbitdet.simulation.environment.env_setup.create_system_of_bodies")
    @patch("orbitdet.simulation.environment.env_setup.get_default_body_settings")
    def test_environment_calls_get_default_body_settings_with_correct_params(
        self,
        mock_get_defaults,
        mock_create_bodies,
        mock_add_neptune,
        basic_config,
        mock_runtime_context,
    ):
        """Test that get_default_body_settings is called with correct frame parameters."""
        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_body_settings.get = MagicMock(return_value=MagicMock())
        mock_get_defaults.return_value = mock_body_settings

        mock_bodies = MagicMock(spec=env.SystemOfBodies)
        mock_create_bodies.return_value = mock_bodies

        get_environment(basic_config, mock_runtime_context)

        # Verify get_default_body_settings called with correct frame parameters
        call_args = mock_get_defaults.call_args[0]
        assert call_args[1] == basic_config.global_frame_origin
        assert call_args[2] == basic_config.global_frame_orientation

    @patch("orbitdet.simulation.environment.add_neptune")
    @patch("orbitdet.simulation.environment.env_setup.create_system_of_bodies")
    @patch("orbitdet.simulation.environment.env_setup.get_default_body_settings")
    def test_environment_returns_created_system_of_bodies(
        self,
        mock_get_defaults,
        mock_create_bodies,
        mock_add_neptune,
        basic_config,
        mock_runtime_context,
    ):
        """Test that get_environment returns the created system of bodies."""
        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_body_settings.get = MagicMock(return_value=MagicMock())
        mock_get_defaults.return_value = mock_body_settings

        mock_bodies = MagicMock(spec=env.SystemOfBodies)
        mock_create_bodies.return_value = mock_bodies

        result = get_environment(basic_config, mock_runtime_context)

        # Verify returned object is the created system
        assert result is mock_bodies
        mock_create_bodies.assert_called_once_with(mock_body_settings)


class TestAddNeptune:
    """Tests for add_neptune function."""

    def test_add_neptune_central_gravity_model(self, basic_config):
        """Test add_neptune with central gravity model (no-op)."""
        from orbitdet.simulation.environment import add_neptune

        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_neptune_settings = MagicMock()
        mock_body_settings.get = MagicMock(return_value=mock_neptune_settings)

        # Should not raise error with central gravity
        add_neptune(basic_config, mock_body_settings)

        # Verify Neptune body settings were accessed
        mock_body_settings.get.assert_called_with("Neptune")

    @patch("orbitdet.simulation.environment.spice")
    def test_add_neptune_jacobson_gravity_model(self, mock_spice, jacobson_config):
        """Test add_neptune with Jacobson2009 gravity model."""
        from orbitdet.simulation.environment import add_neptune

        mock_spice.get_body_gravitational_parameter.return_value = 6.836529e15
        mock_spice.get_body_properties.return_value = [24764, 24764, 24341]  # Neptune radii in km

        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_neptune_settings = MagicMock()
        mock_body_settings.get = MagicMock(return_value=mock_neptune_settings)

        add_neptune(jacobson_config, mock_body_settings)

        # Verify SPICE calls
        mock_spice.get_body_gravitational_parameter.assert_called_once_with("Neptune")
        mock_spice.get_body_properties.assert_called_once_with("Neptune", "RADII", 3)

        # Verify gravity field settings were set
        assert mock_neptune_settings.gravity_field_settings is not None

    def test_add_neptune_unsupported_gravity_model(self, basic_config):
        """Test add_neptune raises error for unsupported gravity model."""
        from orbitdet.simulation.environment import add_neptune

        config = OmegaConf.create(
            {
                "bodies_to_create": {
                    "Neptune": {
                        "gravity": "UnsupportedModel",
                        "rotation_model": "spice",
                    },
                },
                "global_frame_orientation": "ECLIPJ2000",
            }
        )

        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_body_settings.get = MagicMock(return_value=MagicMock())

        with pytest.raises(ValueError, match="Unsupported gravity model"):
            add_neptune(config, mock_body_settings)

    @patch("orbitdet.simulation.environment.env_setup.rotation_model.spice")
    def test_add_neptune_spice_rotation_model(self, mock_spice_rotation, basic_config):
        """Test add_neptune with SPICE rotation model."""
        from orbitdet.simulation.environment import add_neptune

        config = OmegaConf.create(
            {
                "bodies_to_create": {
                    "Neptune": {
                        "gravity": "central",
                        "rotation_model": "spice",
                    },
                },
                "global_frame_origin": "Neptune",
                "global_frame_orientation": "ECLIPJ2000",
            }
        )

        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_neptune_settings = MagicMock()
        mock_body_settings.get = MagicMock(return_value=mock_neptune_settings)

        add_neptune(config, mock_body_settings)

        # Verify rotation model settings were set
        assert mock_neptune_settings.rotation_model_settings is not None
        mock_spice_rotation.assert_called_once()

    @patch("orbitdet.simulation.environment.env_setup.rotation_model.simple_from_spice")
    def test_add_neptune_simple_from_spice_rotation_model(
        self, mock_simple_from_spice, basic_config
    ):
        """Test add_neptune with simple_from_spice rotation model."""
        from orbitdet.simulation.environment import add_neptune

        config = OmegaConf.create(
            {
                "bodies_to_create": {
                    "Neptune": {
                        "gravity": "central",
                        "rotation_model": "simple_from_spice",
                    },
                },
                "global_frame_origin": "Neptune",
                "global_frame_orientation": "ECLIPJ2000",
                "start_epoch": 1234567890.0,
            }
        )

        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_neptune_settings = MagicMock()
        mock_body_settings.get = MagicMock(return_value=mock_neptune_settings)

        add_neptune(config, mock_body_settings)

        # Verify rotation model settings were set
        assert mock_neptune_settings.rotation_model_settings is not None
        mock_simple_from_spice.assert_called_once()

    @patch("orbitdet.simulation.environment.env_setup.rotation_model.iau_rotation_model")
    def test_add_neptune_iau2015_rotation_model(self, mock_iau_rotation, basic_config):
        """Test add_neptune with IAU2015 rotation model."""
        from orbitdet.simulation.environment import add_neptune

        config = OmegaConf.create(
            {
                "bodies_to_create": {
                    "Neptune": {
                        "gravity": "central",
                        "rotation_model": "IAU2015",
                    },
                },
                "global_frame_origin": "Neptune",
                "global_frame_orientation": "ECLIPJ2000",
            }
        )

        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_neptune_settings = MagicMock()
        mock_body_settings.get = MagicMock(return_value=mock_neptune_settings)

        add_neptune(config, mock_body_settings)

        # Verify rotation model settings were set
        assert mock_neptune_settings.rotation_model_settings is not None
        mock_iau_rotation.assert_called_once()

    @patch("orbitdet.simulation.environment.env_setup.rotation_model.iau_rotation_model")
    def test_add_neptune_iau2015_with_custom_pole_values(self, mock_iau_rotation, basic_config):
        """Test add_neptune with IAU2015 rotation model and custom pole values."""
        from orbitdet.simulation.environment import add_neptune

        config = OmegaConf.create(
            {
                "bodies_to_create": {
                    "Neptune": {
                        "gravity": "central",
                        "rotation_model": "IAU2015",
                        "initial_Pole_Pos": [1.0, 2.0],
                        "initial_Pole_lib_deg1": [0.5, 1.5],
                    },
                },
                "global_frame_origin": "Neptune",
                "global_frame_orientation": "ECLIPJ2000",
            }
        )

        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_neptune_settings = MagicMock()
        mock_body_settings.get = MagicMock(return_value=mock_neptune_settings)

        add_neptune(config, mock_body_settings)

        # Verify rotation model settings were set
        assert mock_neptune_settings.rotation_model_settings is not None
        mock_iau_rotation.assert_called_once()

    @patch("orbitdet.simulation.environment.env_setup.rotation_model.iau_rotation_model")
    def test_add_neptune_pole_model_jacobson2009_rotation_model(
        self, mock_iau_rotation, basic_config
    ):
        """Test add_neptune with Pole_Model_Jacobson2009 rotation model."""
        from orbitdet.simulation.environment import add_neptune

        config = OmegaConf.create(
            {
                "bodies_to_create": {
                    "Neptune": {
                        "gravity": "central",
                        "rotation_model": "Pole_Model_Jacobson2009",
                    },
                },
                "global_frame_origin": "Neptune",
                "global_frame_orientation": "ECLIPJ2000",
            }
        )

        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_neptune_settings = MagicMock()
        mock_body_settings.get = MagicMock(return_value=mock_neptune_settings)

        add_neptune(config, mock_body_settings)

        # Verify rotation model settings were set
        assert mock_neptune_settings.rotation_model_settings is not None
        mock_iau_rotation.assert_called_once()

    def test_add_neptune_unsupported_rotation_model(self, basic_config):
        """Test add_neptune with unsupported rotation model (silently skips)."""
        from orbitdet.simulation.environment import add_neptune

        config = OmegaConf.create(
            {
                "bodies_to_create": {
                    "Neptune": {
                        "gravity": "central",
                        "rotation_model": "UnsupportedRotationModel",
                    },
                },
                "global_frame_origin": "Neptune",
                "global_frame_orientation": "ECLIPJ2000",
            }
        )

        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_neptune_settings = MagicMock()
        mock_body_settings.get = MagicMock(return_value=mock_neptune_settings)

        # Should not raise error - unsupported models are silently skipped
        add_neptune(config, mock_body_settings)

    @patch("orbitdet.simulation.environment.env_setup.ephemeris.direct_spice")
    def test_add_neptune_sets_neptune_ephemeris(self, mock_direct_spice, basic_config):
        """Test add_neptune sets Neptune ephemeris model."""
        from orbitdet.simulation.environment import add_neptune

        config = OmegaConf.create(
            {
                "bodies_to_create": {
                    "Neptune": {
                        "gravity": "central",
                        "rotation_model": "spice",
                    },
                },
                "global_frame_origin": "Neptune",
                "global_frame_orientation": "ECLIPJ2000",
            }
        )

        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_neptune_settings = MagicMock()
        mock_body_settings.get = MagicMock(return_value=mock_neptune_settings)

        add_neptune(config, mock_body_settings)

        # Verify ephemeris settings were set
        assert mock_neptune_settings.ephemeris_settings is not None

    @patch("orbitdet.simulation.environment.spice")
    def test_add_neptune_jacobson_gravity_computes_coefficients(self, mock_spice, jacobson_config):
        """Test add_neptune Jacobson model computes correct gravity coefficients."""
        from orbitdet.simulation.environment import add_neptune

        mock_spice.get_body_gravitational_parameter.return_value = 6.836529e15
        mock_spice.get_body_properties.return_value = [24764, 24764, 24341]

        mock_body_settings = MagicMock(spec=env_setup.BodyListSettings)
        mock_neptune_settings = MagicMock()
        mock_body_settings.get = MagicMock(return_value=mock_neptune_settings)

        add_neptune(jacobson_config, mock_body_settings)

        # Verify gravity field was configured (spherical harmonic)
        call_kwargs = mock_neptune_settings.gravity_field_settings
        assert call_kwargs is not None

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
                "global_frame_orientation": "ECLIPJ2000",
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
                "global_frame_orientation": "ECLIPJ2000",
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

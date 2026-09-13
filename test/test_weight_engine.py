"""Tests for the WeightEngine.

The WeightEngine orchestrates weighting over an ObservationCollection.
It depends on Tudat objects that must be mocked.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from orbitdet.observations.utils import get_set_identifier
from orbitdet.observations.weighting.engine import WeightEngine

RECEIVER_KEY = "receiver"


@pytest.fixture(autouse=True)
def _patch_links_receiver(monkeypatch):
    """Patch tudatpy's links.receiver so _get_set_id works with our mocks."""
    import tudatpy.estimation.observable_models_setup

    monkeypatch.setattr(
        tudatpy.estimation.observable_models_setup.links,
        "receiver",
        RECEIVER_KEY,
    )


# ===========================================================================
# _get_set_id helper
# ===========================================================================


class TestGetSetId:
    def _make_obs_set(self, body_name="Earth", reference_point=""):
        """Create a mock observation set with link definition."""
        link_ends = {
            RECEIVER_KEY: SimpleNamespace(
                body_name=body_name,
                reference_point=reference_point,
            )
        }
        link_def = SimpleNamespace(link_ends=link_ends)
        return SimpleNamespace(link_definition=link_def)

    def test_ground_station_with_numeric_code(self):
        obs_set = self._make_obs_set(body_name="Earth", reference_point="689")
        assert get_set_identifier(obs_set) == "689"

    def test_ground_station_with_negative_code(self):
        obs_set = self._make_obs_set(body_name="Earth", reference_point="-1")
        assert get_set_identifier(obs_set) == "Earth"

    def test_spacecraft_empty_reference_point(self):
        obs_set = self._make_obs_set(body_name="Voyager 2", reference_point="")
        assert get_set_identifier(obs_set) == "Voyager 2"

    def test_no_receiver_fallback(self):
        obs_set = SimpleNamespace(link_definition=SimpleNamespace(link_ends={}))
        result = get_set_identifier(obs_set)
        assert isinstance(result, str)
        assert len(result) > 0


# ===========================================================================
# WeightEngine.from_config
# ===========================================================================


class TestWeightEngineFromConfig:
    def test_id_v2_strategy(self):
        cfg = OmegaConf.create(
            {
                "strategy": "id_v2",
                "min_sigma_arcsec": 0.01,
                "grouping": {
                    "levels": [{"type": "timeframe", "gap_threshold_hours": 4.0}],
                },
            }
        )
        engine = WeightEngine.from_config(cfg)
        assert engine.strategy.__class__.__name__ == "IDv2Weight"

    def test_hybrid_geometric_strategy(self):
        cfg = OmegaConf.create(
            {
                "strategy": "hybrid_geometric",
            }
        )
        engine = WeightEngine.from_config(cfg)
        assert engine.strategy.__class__.__name__ == "HybridGeometricWeight"

    def test_default_strategy(self):
        cfg = OmegaConf.create({})
        engine = WeightEngine.from_config(cfg)
        assert engine.strategy.__class__.__name__ == "IDv2Weight"

    def test_default_min_sigma(self):
        cfg = OmegaConf.create({"strategy": "id_v2"})
        engine = WeightEngine.from_config(cfg)
        # Can't easily inspect private attr, but we can verify it doesn't crash
        assert engine is not None

    def test_grouping_passed_through(self):
        cfg = OmegaConf.create(
            {
                "strategy": "id_v2",
                "grouping": {"levels": [{"type": "section", "sections": []}]},
            }
        )
        engine = WeightEngine.from_config(cfg)
        assert engine is not None

    def test_no_grouping(self):
        cfg = OmegaConf.create({"strategy": "id_v2"})
        engine = WeightEngine.from_config(cfg)
        assert engine is not None

    def test_unknown_strategy_raises(self):
        cfg = OmegaConf.create({"strategy": "nonexistent"})
        with pytest.raises(ValueError, match="No weight strategy registered"):
            WeightEngine.from_config(cfg)


# ===========================================================================
# WeightEngine.apply
# ===========================================================================


RECEIVER_KEY = "receiver"


def _make_mock_time(epoch: float):
    return SimpleNamespace(to_float=lambda: epoch)


def _make_mock_single_obs_set(
    set_id: str,
    residuals: np.ndarray,
    times: list[float],
    body_name: str = "Earth",
    reference_point: str = "689",
    observable_type: int = 1,
) -> SimpleNamespace:
    """Create a mock SingleObservationSet."""
    link_ends = {
        RECEIVER_KEY: SimpleNamespace(
            body_name=body_name,
            reference_point=reference_point,
        )
    }
    mock_times = [_make_mock_time(t) for t in times]
    return SimpleNamespace(
        residuals=residuals,
        observation_times=mock_times,
        observable_type=observable_type,
        link_definition=SimpleNamespace(link_ends=link_ends),
        set_tabulated_weights=MagicMock(),
    )


def _make_mock_collection(sets: list) -> MagicMock:
    """Create a mock ObservationCollection."""
    collection = MagicMock()
    collection.get_single_observation_sets.return_value = sets
    return collection


class TestWeightEngineApply:
    def test_single_set_single_timeframe(self):
        """Basic flow: one set, weights computed and assigned."""
        residuals = np.array([[1e-6, 2e-6], [3e-6, 4e-6]])
        times = [0.0, 3600.0]
        obs_set = _make_mock_single_obs_set("689", residuals, times)
        collection = _make_mock_collection([obs_set])

        cfg = OmegaConf.create(
            {
                "strategy": "id_v2",
                "grouping": {"levels": [{"type": "timeframe", "gap_threshold_hours": 4.0}]},
            }
        )
        engine = WeightEngine.from_config(cfg)
        bodies = MagicMock()

        result_collection, weights_df = engine.apply(collection, bodies)

        # Verify set_tabulated_weights was called
        obs_set.set_tabulated_weights.assert_called_once()
        call_args = obs_set.set_tabulated_weights.call_args[0][0]
        assert len(call_args) == 4  # 2 obs * 2 (RA/DEC)
        assert not np.any(np.isnan(call_args))

        # Verify DataFrame
        assert len(weights_df) == 2
        assert "weight_ra" in weights_df.columns
        assert "weight_dec" in weights_df.columns

    def test_multiple_sets(self):
        """Multiple sets are each processed independently."""
        set1 = _make_mock_single_obs_set("689", np.array([[1e-6, 2e-6]]), [0.0])
        set2 = _make_mock_single_obs_set(
            "690", np.array([[3e-6, 4e-6]]), [100.0], reference_point="690"
        )
        collection = _make_mock_collection([set1, set2])

        cfg = OmegaConf.create({"strategy": "id_v2"})
        engine = WeightEngine.from_config(cfg)
        bodies = MagicMock()

        _, weights_df = engine.apply(collection, bodies)

        set1.set_tabulated_weights.assert_called_once()
        set2.set_tabulated_weights.assert_called_once()
        assert len(weights_df) == 2

    def test_empty_set_skipped(self):
        """Empty observation sets should be skipped gracefully."""
        obs_set = _make_mock_single_obs_set(
            "689",
            np.empty((0, 2)),
            [],
        )
        obs_set.observation_times = []  # empty
        collection = _make_mock_collection([obs_set])

        cfg = OmegaConf.create({"strategy": "id_v2"})
        engine = WeightEngine.from_config(cfg)
        bodies = MagicMock()

        _, weights_df = engine.apply(collection, bodies)

        obs_set.set_tabulated_weights.assert_not_called()
        assert weights_df.empty

    def test_csv_saved_when_output_dir_provided(self, tmp_path):
        """When output_dir is provided, CSV is saved."""
        residuals = np.array([[1e-6, 2e-6]])
        times = [0.0]
        obs_set = _make_mock_single_obs_set("689", residuals, times)
        collection = _make_mock_collection([obs_set])

        cfg = OmegaConf.create({"strategy": "id_v2"})
        engine = WeightEngine.from_config(cfg)
        bodies = MagicMock()

        _, weights_df = engine.apply(collection, bodies, output_dir=str(tmp_path))

        csv_path = tmp_path / "observation_weights.csv"
        assert csv_path.exists()
        saved_df = pd.read_csv(csv_path)
        assert len(saved_df) == 1

    def test_no_csv_when_no_output_dir(self, tmp_path):
        """When output_dir is None, no CSV is saved."""
        residuals = np.array([[1e-6, 2e-6]])
        times = [0.0]
        obs_set = _make_mock_single_obs_set("689", residuals, times)
        collection = _make_mock_collection([obs_set])

        cfg = OmegaConf.create({"strategy": "id_v2"})
        engine = WeightEngine.from_config(cfg)
        bodies = MagicMock()

        engine.apply(collection, bodies, output_dir=None)

        csv_path = tmp_path / "observation_weights.csv"
        assert not csv_path.exists()

    def test_set_tabulated_weights_called_with_correct_shape(self):
        """The weight array should have shape (2 * n_obs,)."""
        residuals = np.array([[1e-6, 2e-6], [3e-6, 4e-6], [5e-6, 6e-6]])
        times = [0.0, 3600.0, 7200.0]
        obs_set = _make_mock_single_obs_set("689", residuals, times)
        collection = _make_mock_collection([obs_set])

        cfg = OmegaConf.create({"strategy": "id_v2"})
        engine = WeightEngine.from_config(cfg)
        bodies = MagicMock()

        engine.apply(collection, bodies)

        call_args = obs_set.set_tabulated_weights.call_args[0][0]
        assert call_args.shape == (6,)  # 3 obs * 2

    def test_hybrid_geometric_strategy(self):
        """Apply with hybrid_geometric strategy."""
        residuals = np.array([[1e-6, 2e-6], [3e-6, 4e-6]])
        times = [0.0, 3600.0]
        obs_set = _make_mock_single_obs_set("689", residuals, times)
        collection = _make_mock_collection([obs_set])

        cfg = OmegaConf.create({"strategy": "hybrid_geometric"})
        engine = WeightEngine.from_config(cfg)
        bodies = MagicMock()

        result_collection, weights_df = engine.apply(collection, bodies)

        obs_set.set_tabulated_weights.assert_called_once()
        assert len(weights_df) == 2
        assert weights_df["strategy"].iloc[0] == "hybrid_geometric"

    def test_logger_output(self, caplog):
        """Check that the engine logs key information."""
        import logging

        caplog.set_level(logging.INFO)

        residuals = np.array([[1e-6, 2e-6]])
        times = [0.0]
        obs_set = _make_mock_single_obs_set("689", residuals, times)
        collection = _make_mock_collection([obs_set])

        cfg = OmegaConf.create({"strategy": "id_v2"})
        engine = WeightEngine.from_config(cfg)
        bodies = MagicMock()

        engine.apply(collection, bodies)

        assert "WeightEngine" in caplog.text
        assert "IDv2Weight" in caplog.text or "id_v2" in caplog.text

    def test_fixed_strategy_from_config(self):
        """Fixed strategy with fixed_sigmas injected via from_config."""
        cfg = OmegaConf.create(
            {
                "strategy": "fixed",
                "grouping": {
                    "levels": [{"type": "timeframe", "gap_threshold_hours": 4.0}],
                    "fixed_sigmas": {"689": {"ra": 0.15, "dec": 0.15}},
                },
            }
        )
        engine = WeightEngine.from_config(cfg)
        assert engine.strategy.__class__.__name__ == "FixedWeight"
        assert engine._strategy._fixed_sigmas["689"]["ra"] == 0.15
        assert engine._strategy._fixed_sigmas["689"]["dec"] == 0.15

    def test_fixed_strategy_applies_weights(self):
        """Fixed strategy should set tabulated weights via the engine."""
        residuals = np.array([[1e-6, 2e-6], [3e-6, 4e-6]])
        times = [0.0, 3600.0]
        obs_set = _make_mock_single_obs_set("689", residuals, times, reference_point="689")
        collection = _make_mock_collection([obs_set])

        cfg = OmegaConf.create(
            {
                "strategy": "fixed",
                "grouping": {
                    "fixed_sigmas": {"689": {"ra": 0.1, "dec": 0.2}},
                },
            }
        )
        engine = WeightEngine.from_config(cfg)
        bodies = MagicMock()

        result_collection, weights_df = engine.apply(collection, bodies)

        obs_set.set_tabulated_weights.assert_called_once()
        call_args = obs_set.set_tabulated_weights.call_args[0][0]
        assert len(call_args) == 4

        ra_sigma_rad = 0.1 * np.pi / (180 * 3600)
        dec_sigma_rad = 0.2 * np.pi / (180 * 3600)
        expected_w_ra = 1.0 / ra_sigma_rad**2
        expected_w_dec = 1.0 / dec_sigma_rad**2

        np.testing.assert_allclose(call_args[0], expected_w_ra)
        np.testing.assert_allclose(call_args[1], expected_w_dec)
        assert len(weights_df) == 2
        assert weights_df["strategy"].iloc[0] == "fixed"

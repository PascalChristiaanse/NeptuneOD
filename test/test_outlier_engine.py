"""Tests for the OutlierEngine.

The engine orchestrates outlier rejection over an ObservationCollection.
It depends on Tudat objects that must be mocked.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from orbitdet.observations.outlier_rejection.base import OutlierStrategy
from orbitdet.observations.outlier_rejection.engine import (
    OutlierEngine,
    _ScopedStrategy,
    get_set_identifier,
)

# ===========================================================================
# Constants and helpers
# ===========================================================================

RECEIVER_KEY = "receiver"


@pytest.fixture(autouse=True)
def _patch_links_receiver(monkeypatch):
    """Patch tudatpy's links.receiver so get_set_identifier works with mocks."""
    import tudatpy.estimation.observable_models_setup

    monkeypatch.setattr(
        tudatpy.estimation.observable_models_setup.links,
        "receiver",
        RECEIVER_KEY,
    )


@pytest.fixture(autouse=True)
def _patch_tudat_filtering(monkeypatch):
    """Mock Tudat filtering functions used by _build_rejected_set and apply_with_rejected."""
    import orbitdet.observations.outlier_rejection.engine as eng

    class MockObservationCollection:
        def __init__(self, sets):
            self._mock_sets = sets

    # Mock _build_rejected_set to avoid local imports of C extensions
    def mock_build_rejected_set(original_set, accepted_set):
        """Simplified mock that computes set difference of epochs."""
        original_epochs = {t.to_float() for t in original_set.observation_times}
        accepted_epochs = {t.to_float() for t in accepted_set.observation_times}
        rejected_epochs = sorted(original_epochs - accepted_epochs)
        rejected_times = [_make_mock_time(t) for t in rejected_epochs]
        return SimpleNamespace(
            residuals=original_set.residuals,
            observation_times=rejected_times,
            link_definition=original_set.link_definition,
        )

    monkeypatch.setattr(eng, "_build_rejected_set", mock_build_rejected_set)
    monkeypatch.setattr(eng.obs, "ObservationCollection", MockObservationCollection)


def _make_mock_time(epoch: float):
    return SimpleNamespace(to_float=lambda: epoch)


def _make_mock_obs_set(
    set_id: str = "689",
    n_obs: int = 3,
    body_name: str = "Earth",
    reference_point: str = "689",
) -> SimpleNamespace:
    """Create a mock SingleObservationSet."""
    link_ends = {
        RECEIVER_KEY: SimpleNamespace(
            body_name=body_name,
            reference_point=reference_point,
        )
    }
    mock_times = [_make_mock_time(float(i * 100.0)) for i in range(n_obs)]
    return SimpleNamespace(
        residuals=MagicMock(),
        observation_times=mock_times,
        link_definition=SimpleNamespace(link_ends=link_ends),
    )


def _make_mock_collection(sets: list) -> MagicMock:
    """Create a mock ObservationCollection."""
    collection = MagicMock()
    collection.get_single_observation_sets.return_value = sets
    return collection


def _make_mock_strategy(name: str = "MockStrategy") -> MagicMock:
    """Create a mock OutlierStrategy that accepts all observations."""
    strategy = MagicMock(spec=OutlierStrategy)

    def apply_side_effect(obs_set, bodies):
        n = len(obs_set.observation_times)
        return obs_set, {
            "strategy": name,
            "n_accepted": n,
            "n_rejected": 0,
            "n_total": n,
            "rejected_epochs": [],
        }

    strategy.apply.side_effect = apply_side_effect
    return strategy


def _make_mock_rejecting_strategy(
    name: str = "RejectingStrategy",
    reject_indices: list[int] | None = None,
) -> MagicMock:
    """Create a mock strategy that rejects specific observations by index."""
    if reject_indices is None:
        reject_indices = [0]
    strategy = MagicMock(spec=OutlierStrategy)

    def apply_side_effect(obs_set, bodies):
        times = [t.to_float() for t in obs_set.observation_times]
        n = len(times)
        keep_times = [t for i, t in enumerate(times) if i not in reject_indices]
        reject_t = [t for i, t in enumerate(times) if i in reject_indices]
        keep_mock_times = [_make_mock_time(t) for t in keep_times]
        return SimpleNamespace(
            residuals=obs_set.residuals,
            observation_times=keep_mock_times,
            link_definition=obs_set.link_definition,
        ), {
            "strategy": name,
            "n_accepted": len(keep_times),
            "n_rejected": len(reject_t),
            "n_total": n,
            "rejected_epochs": reject_t,
        }

    strategy.apply.side_effect = apply_side_effect
    return strategy


# ===========================================================================
# get_set_identifier
# ===========================================================================


class TestGetSetIdentifier:
    def test_ground_station_with_numeric_code(self):
        link_ends = {
            RECEIVER_KEY: SimpleNamespace(
                body_name="Earth",
                reference_point="689",
            )
        }
        obs_set = SimpleNamespace(
            link_definition=SimpleNamespace(link_ends=link_ends),
        )
        assert get_set_identifier(obs_set) == "689"

    def test_ground_station_with_negative_code(self):
        link_ends = {
            RECEIVER_KEY: SimpleNamespace(
                body_name="Earth",
                reference_point="-1",
            )
        }
        obs_set = SimpleNamespace(
            link_definition=SimpleNamespace(link_ends=link_ends),
        )
        assert get_set_identifier(obs_set) == "Earth"

    def test_spacecraft_empty_reference_point(self):
        link_ends = {
            RECEIVER_KEY: SimpleNamespace(
                body_name="Voyager 2",
                reference_point="",
            )
        }
        obs_set = SimpleNamespace(
            link_definition=SimpleNamespace(link_ends=link_ends),
        )
        assert get_set_identifier(obs_set) == "Voyager 2"

    def test_no_receiver_fallback(self):
        obs_set = SimpleNamespace(
            link_definition=SimpleNamespace(link_ends={}),
        )
        result = get_set_identifier(obs_set)
        assert isinstance(result, str)
        assert len(result) > 0


# ===========================================================================
# _ScopedStrategy
# ===========================================================================


class TestScopedStrategy:
    def test_no_filter_applies_to_all(self):
        strategy = _make_mock_strategy()
        scoped = _ScopedStrategy(strategy, set_filter=None)
        assert scoped.applies_to("any_set") is True
        assert scoped.applies_to("689") is True

    def test_with_filter_matches(self):
        strategy = _make_mock_strategy()
        scoped = _ScopedStrategy(strategy, set_filter={"689", "690"})
        assert scoped.applies_to("689") is True
        assert scoped.applies_to("690") is True

    def test_with_filter_does_not_match(self):
        strategy = _make_mock_strategy()
        scoped = _ScopedStrategy(strategy, set_filter={"689"})
        assert scoped.applies_to("690") is False
        assert scoped.applies_to("unknown") is False


# ===========================================================================
# OutlierEngine construction
# ===========================================================================


class TestOutlierEngineConstruction:
    def test_empty_strategies_raises(self):
        with pytest.raises(ValueError, match="At least one outlier strategy"):
            OutlierEngine([])

    def test_single_strategy(self):
        strategy = _make_mock_strategy()
        engine = OutlierEngine.from_strategies([strategy])
        assert len(engine.strategies) == 1
        assert engine.strategies[0] is strategy

    def test_multiple_strategies(self):
        s1 = _make_mock_strategy("S1")
        s2 = _make_mock_strategy("S2")
        engine = OutlierEngine.from_strategies([s1, s2])
        assert len(engine.strategies) == 2


# ===========================================================================
# OutlierEngine.from_config
# ===========================================================================


class TestOutlierEngineFromConfig:
    def test_single_strategy(self):
        cfg = OmegaConf.create(
            {
                "strategies": [{"type": "residual_threshold", "threshold_arcsec": 1.5}],
            }
        )
        engine = OutlierEngine.from_config(cfg)
        assert len(engine.strategies) == 1

    def test_multiple_strategies(self):
        cfg = OmegaConf.create(
            {
                "strategies": [
                    {"type": "residual_threshold", "threshold_arcsec": 1.5},
                    {"type": "epoch_filter", "epochs": [100.0, 200.0]},
                ],
            }
        )
        engine = OutlierEngine.from_config(cfg)
        assert len(engine.strategies) == 2

    def test_with_set_filter(self):
        cfg = OmegaConf.create(
            {
                "strategies": [
                    {"type": "residual_threshold", "threshold_arcsec": 0.5, "sets": ["Voyager 2"]},
                    {"type": "residual_threshold", "threshold_arcsec": 1.5},
                ],
            }
        )
        engine = OutlierEngine.from_config(cfg)
        assert len(engine.strategies) == 2

    def test_empty_strategies_raises(self):
        cfg = OmegaConf.create({"strategies": []})
        with pytest.raises(ValueError, match="non-empty 'strategies'"):
            OutlierEngine.from_config(cfg)

    def test_missing_strategies_raises(self):
        cfg = OmegaConf.create({})
        with pytest.raises(ValueError, match="non-empty 'strategies'"):
            OutlierEngine.from_config(cfg)

    def test_missing_type_field_raises(self):
        cfg = OmegaConf.create(
            {
                "strategies": [{"threshold_arcsec": 1.5}],
            }
        )
        with pytest.raises(ValueError, match="must have a 'type' field"):
            OutlierEngine.from_config(cfg)

    def test_unknown_strategy_raises(self):
        cfg = OmegaConf.create(
            {
                "strategies": [{"type": "nonexistent"}],
            }
        )
        with pytest.raises(ValueError, match="No outlier strategy registered"):
            OutlierEngine.from_config(cfg)


# ===========================================================================
# OutlierEngine.apply
# ===========================================================================


class TestOutlierEngineApply:
    def test_single_set_single_strategy(self):
        """Basic flow: one set, one strategy, all accepted."""
        obs_set = _make_mock_obs_set("689", n_obs=3)
        collection = _make_mock_collection([obs_set])
        strategy = _make_mock_strategy()
        engine = OutlierEngine.from_strategies([strategy])
        bodies = MagicMock()

        filtered, metadata = engine.apply(collection, bodies)

        assert metadata["n_sets"] == 1
        assert metadata["n_accepted"] == 3
        assert metadata["n_rejected"] == 0
        assert metadata["n_total_observations"] == 3

    def test_single_set_strategy_rejects_some(self):
        """Strategy rejects some observations."""
        obs_set = _make_mock_obs_set("689", n_obs=3)
        collection = _make_mock_collection([obs_set])
        strategy = _make_mock_rejecting_strategy(reject_indices=[0])
        engine = OutlierEngine.from_strategies([strategy])
        bodies = MagicMock()

        filtered, metadata = engine.apply(collection, bodies)

        assert metadata["n_accepted"] == 2
        assert metadata["n_rejected"] == 1
        assert metadata["n_total_observations"] == 3

    def test_multiple_sets(self):
        """Multiple sets are each processed independently."""
        set1 = _make_mock_obs_set("689", n_obs=2)
        set2 = _make_mock_obs_set("690", n_obs=3, reference_point="690")
        collection = _make_mock_collection([set1, set2])
        strategy = _make_mock_strategy()
        engine = OutlierEngine.from_strategies([strategy])
        bodies = MagicMock()

        filtered, metadata = engine.apply(collection, bodies)

        assert metadata["n_sets"] == 2
        assert metadata["n_accepted"] == 5
        assert metadata["n_total_observations"] == 5

    def test_multiple_strategies_in_sequence(self):
        """Multiple strategies are applied in sequence."""
        obs_set = _make_mock_obs_set("689", n_obs=4)
        collection = _make_mock_collection([obs_set])
        s1 = _make_mock_rejecting_strategy("S1", reject_indices=[0])
        s2 = _make_mock_rejecting_strategy("S2", reject_indices=[0])  # rejects first of remaining
        engine = OutlierEngine.from_strategies([s1, s2])
        bodies = MagicMock()

        filtered, metadata = engine.apply(collection, bodies)

        assert metadata["n_accepted"] == 2  # 4 - 1 - 1
        assert metadata["n_rejected"] == 2
        assert metadata["n_total_observations"] == 4

    def test_strategy_with_set_filter_applies_only_to_matching(self):
        """A strategy with a set filter should only apply to matching sets."""
        set1 = _make_mock_obs_set("Voyager 2", n_obs=3, body_name="Voyager 2", reference_point="")
        set2 = _make_mock_obs_set("689", n_obs=3)
        collection = _make_mock_collection([set1, set2])

        # Only apply to Voyager 2
        strategy = _make_mock_rejecting_strategy("VoyagerOnly", reject_indices=[0, 1])
        engine = OutlierEngine.from_strategies([strategy])
        # Manually set the scoped strategy with a filter
        engine._strategies = [_ScopedStrategy(strategy, set_filter={"Voyager 2"})]
        bodies = MagicMock()

        filtered, metadata = engine.apply(collection, bodies)

        # Voyager 2: 3 - 2 = 1 accepted
        # 689: 3 accepted (strategy skipped)
        assert metadata["n_accepted"] == 4
        assert metadata["n_rejected"] == 2
        assert metadata["n_total_observations"] == 6

    def test_per_set_metadata(self):
        """Metadata should contain per-set breakdown."""
        set1 = _make_mock_obs_set("689", n_obs=2)
        set2 = _make_mock_obs_set("690", n_obs=3, reference_point="690")
        collection = _make_mock_collection([set1, set2])
        strategy = _make_mock_strategy()
        engine = OutlierEngine.from_strategies([strategy])
        bodies = MagicMock()

        filtered, metadata = engine.apply(collection, bodies)

        def _find_set(per_set: list, set_id: str) -> dict | None:
            return next((m for m in per_set if m["set_id"] == set_id), None)

        assert _find_set(metadata["per_set"], "689") is not None
        assert _find_set(metadata["per_set"], "690") is not None
        assert _find_set(metadata["per_set"], "689")["n_accepted"] == 2
        assert _find_set(metadata["per_set"], "690")["n_accepted"] == 3


# ===========================================================================
# OutlierEngine.apply_with_rejected
# ===========================================================================


class TestOutlierEngineApplyWithRejected:
    def test_returns_accepted_rejected_and_summary(self):
        obs_set = _make_mock_obs_set("689", n_obs=3)
        collection = _make_mock_collection([obs_set])
        strategy = _make_mock_rejecting_strategy(reject_indices=[0])
        engine = OutlierEngine.from_strategies([strategy])
        bodies = MagicMock()

        accepted, rejected, summary = engine.apply_with_rejected(collection, bodies)

        assert summary["n_accepted"] == 2
        assert summary["n_rejected"] == 1

    def test_rejected_collection_contains_rejected_obs(self):
        obs_set = _make_mock_obs_set("689", n_obs=3)
        collection = _make_mock_collection([obs_set])
        strategy = _make_mock_rejecting_strategy(reject_indices=[0, 1])
        engine = OutlierEngine.from_strategies([strategy])
        bodies = MagicMock()

        accepted, rejected, summary = engine.apply_with_rejected(collection, bodies)

        assert summary["n_accepted"] == 1
        assert summary["n_rejected"] == 2

    def test_no_rejections(self):
        obs_set = _make_mock_obs_set("689", n_obs=3)
        collection = _make_mock_collection([obs_set])
        strategy = _make_mock_strategy()
        engine = OutlierEngine.from_strategies([strategy])
        bodies = MagicMock()

        accepted, rejected, summary = engine.apply_with_rejected(collection, bodies)

        assert summary["n_accepted"] == 3
        assert summary["n_rejected"] == 0

"""Tests for the weighting strategy registry."""

import pytest

from orbitdet.observations.weighting.base import WeightStrategy
from orbitdet.observations.weighting.registry import (
    _STRATEGY_REGISTRY,
    get_strategy_class,
    list_registered_strategies,
    register_weight_strategy,
)


@pytest.fixture(autouse=True)
def reset_weight_registry():
    """Save and restore the registry around each test."""
    original = dict(_STRATEGY_REGISTRY)
    yield
    _STRATEGY_REGISTRY.clear()
    _STRATEGY_REGISTRY.update(original)


class TestWeightRegistry:
    def test_register_and_retrieve(self):
        @register_weight_strategy("test_strat")
        class TestStrat(WeightStrategy):
            def compute_weights(self, observation_set, groups, set_id, min_sigma_arcsec=0.01):
                pass

        retrieved = get_strategy_class("test_strat")
        assert retrieved is TestStrat

    def test_registered_type_appears_in_list(self):
        @register_weight_strategy("list_test")
        class ListTestStrat(WeightStrategy):
            def compute_weights(self, observation_set, groups, set_id, min_sigma_arcsec=0.01):
                pass

        types = list_registered_strategies()
        assert "list_test" in types

    def test_list_is_sorted(self):
        @register_weight_strategy("z_strat")
        class ZStrat(WeightStrategy):
            def compute_weights(self, observation_set, groups, set_id, min_sigma_arcsec=0.01):
                pass

        @register_weight_strategy("a_strat")
        class AStrat(WeightStrategy):
            def compute_weights(self, observation_set, groups, set_id, min_sigma_arcsec=0.01):
                pass

        types = list_registered_strategies()
        assert types == sorted(types)

    def test_duplicate_registration_raises_value_error(self):
        @register_weight_strategy("dup_test")
        class FirstStrat(WeightStrategy):
            def compute_weights(self, observation_set, groups, set_id, min_sigma_arcsec=0.01):
                pass

        with pytest.raises(ValueError, match="already registered"):

            @register_weight_strategy("dup_test")
            class SecondStrat(WeightStrategy):
                def compute_weights(self, observation_set, groups, set_id, min_sigma_arcsec=0.01):
                    pass

        # First registration should still be the one in the registry
        assert get_strategy_class("dup_test") is FirstStrat

    def test_non_subclass_raises_type_error(self):
        with pytest.raises(TypeError, match="must inherit from WeightStrategy"):

            @register_weight_strategy("bad_strat")
            class NotAStrategy:
                pass

    def test_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError, match="No weight strategy registered"):
            get_strategy_class("nonexistent_type")

    def test_error_message_includes_available_types(self):
        @register_weight_strategy("type_a")
        class TypeA(WeightStrategy):
            def compute_weights(self, observation_set, groups, set_id, min_sigma_arcsec=0.01):
                pass

        @register_weight_strategy("type_b")
        class TypeB(WeightStrategy):
            def compute_weights(self, observation_set, groups, set_id, min_sigma_arcsec=0.01):
                pass

        with pytest.raises(ValueError, match="type_a.*type_b|type_b.*type_a"):
            get_strategy_class("nonexistent")

    def test_real_strategies_are_registered(self):
        """Verify that importing the strategies module registers the built-in strategies."""
        # Trigger registration by importing strategies
        import orbitdet.observations.weighting.strategies  # noqa: F401

        assert "id_v2" in list_registered_strategies()
        assert "hybrid_geometric" in list_registered_strategies()

    def test_registered_classes_inherit_weight_strategy(self):
        import orbitdet.observations.weighting.strategies  # noqa: F401

        id_v2_cls = get_strategy_class("id_v2")
        hybrid_cls = get_strategy_class("hybrid_geometric")

        assert issubclass(id_v2_cls, WeightStrategy)
        assert issubclass(hybrid_cls, WeightStrategy)

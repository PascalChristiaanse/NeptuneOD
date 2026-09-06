"""Tests for the outlier rejection strategy registry."""

import pytest

from orbitdet.observations.outlier_rejection.base import OutlierStrategy
from orbitdet.observations.outlier_rejection.registry import (
    _STRATEGY_REGISTRY,
    get_strategy_class,
    list_registered_strategies,
    register_outlier_strategy,
)


@pytest.fixture(autouse=True)
def reset_outlier_registry():
    """Save and restore the registry around each test."""
    original = dict(_STRATEGY_REGISTRY)
    yield
    _STRATEGY_REGISTRY.clear()
    _STRATEGY_REGISTRY.update(original)


class TestOutlierRegistry:
    def test_register_and_retrieve(self):
        @register_outlier_strategy("test_strat")
        class TestStrat(OutlierStrategy):
            def apply(self, observation_set, bodies):
                return observation_set, {}

        retrieved = get_strategy_class("test_strat")
        assert retrieved is TestStrat

    def test_registered_type_appears_in_list(self):
        @register_outlier_strategy("list_test")
        class ListTestStrat(OutlierStrategy):
            def apply(self, observation_set, bodies):
                return observation_set, {}

        types = list_registered_strategies()
        assert "list_test" in types

    def test_list_is_sorted(self):
        @register_outlier_strategy("z_strat")
        class ZStrat(OutlierStrategy):
            def apply(self, observation_set, bodies):
                return observation_set, {}

        @register_outlier_strategy("a_strat")
        class AStrat(OutlierStrategy):
            def apply(self, observation_set, bodies):
                return observation_set, {}

        types = list_registered_strategies()
        assert types == sorted(types)

    def test_duplicate_registration_raises_value_error(self):
        @register_outlier_strategy("dup_test")
        class FirstStrat(OutlierStrategy):
            def apply(self, observation_set, bodies):
                return observation_set, {}

        with pytest.raises(ValueError, match="already registered"):

            @register_outlier_strategy("dup_test")
            class SecondStrat(OutlierStrategy):
                def apply(self, observation_set, bodies):
                    return observation_set, {}

        assert get_strategy_class("dup_test") is FirstStrat

    def test_non_subclass_raises_type_error(self):
        with pytest.raises(TypeError, match="must inherit from OutlierStrategy"):

            @register_outlier_strategy("bad_strat")
            class NotAStrategy:
                pass

    def test_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError, match="No outlier strategy registered"):
            get_strategy_class("nonexistent_type")

    def test_error_message_includes_available_types(self):
        @register_outlier_strategy("type_a")
        class TypeA(OutlierStrategy):
            def apply(self, observation_set, bodies):
                return observation_set, {}

        @register_outlier_strategy("type_b")
        class TypeB(OutlierStrategy):
            def apply(self, observation_set, bodies):
                return observation_set, {}

        with pytest.raises(ValueError, match="type_a.*type_b|type_b.*type_a"):
            get_strategy_class("nonexistent")

    def test_real_strategies_are_registered(self):
        """Verify that importing the strategies module registers the built-in strategies."""
        import orbitdet.observations.outlier_rejection.strategies  # noqa: F401

        assert "residual_threshold" in list_registered_strategies()
        assert "epoch_filter" in list_registered_strategies()

    def test_registered_classes_inherit_outlier_strategy(self):
        import orbitdet.observations.outlier_rejection.strategies  # noqa: F401

        residual_cls = get_strategy_class("residual_threshold")
        epoch_cls = get_strategy_class("epoch_filter")

        assert issubclass(residual_cls, OutlierStrategy)
        assert issubclass(epoch_cls, OutlierStrategy)
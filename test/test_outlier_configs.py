"""Tests for the outlier rejection configuration dataclasses."""

import pytest

from orbitdet.observations.outlier_rejection.configs import (
    OutlierRejectionConfig,
    OutlierStrategyConfig,
)


class TestOutlierStrategyConfig:
    def test_default_construction(self):
        cfg = OutlierStrategyConfig(type="residual_threshold")
        assert cfg.type == "residual_threshold"
        assert cfg.kwargs == {}

    def test_with_kwargs(self):
        cfg = OutlierStrategyConfig(
            type="epoch_filter",
            kwargs={"epochs_to_remove": [100.0, 200.0]},
        )
        assert cfg.type == "epoch_filter"
        assert cfg.kwargs["epochs_to_remove"] == [100.0, 200.0]

    def test_frozen_immutability(self):
        cfg = OutlierStrategyConfig(type="test")
        with pytest.raises(Exception):
            cfg.type = "modified"


class TestOutlierRejectionConfig:
    def test_default_construction(self):
        cfg = OutlierRejectionConfig()
        assert cfg.enabled is False
        assert cfg.strategies == ()

    def test_with_strategies(self):
        strategies = (
            OutlierStrategyConfig(type="residual_threshold", kwargs={"threshold_arcsec": 1.5}),
            OutlierStrategyConfig(type="epoch_filter", kwargs={"epochs_to_remove": [100.0]}),
        )
        cfg = OutlierRejectionConfig(enabled=True, strategies=strategies)
        assert cfg.enabled is True
        assert len(cfg.strategies) == 2
        assert cfg.strategies[0].type == "residual_threshold"
        assert cfg.strategies[1].type == "epoch_filter"

    def test_frozen_immutability(self):
        cfg = OutlierRejectionConfig()
        with pytest.raises(Exception):
            cfg.enabled = True

"""Tests for the weighting configuration dataclass."""

import pytest

from orbitdet.observations.weighting.configs import WeightingConfig


class TestWeightingConfig:
    def test_default_construction(self):
        cfg = WeightingConfig()
        assert cfg.enabled is False
        assert cfg.strategy == "id_v2"
        assert cfg.min_sigma_arcsec == 0.01
        assert cfg.grouping is None

    def test_custom_values(self):
        cfg = WeightingConfig(
            enabled=True,
            strategy="hybrid_geometric",
            min_sigma_arcsec=0.1,
            grouping={"levels": [{"type": "timeframe", "gap_threshold_hours": 2.0}]},
        )
        assert cfg.enabled is True
        assert cfg.strategy == "hybrid_geometric"
        assert cfg.min_sigma_arcsec == 0.1
        assert cfg.grouping["levels"][0]["gap_threshold_hours"] == 2.0

    def test_frozen_immutability(self):
        cfg = WeightingConfig()
        with pytest.raises(Exception):
            cfg.enabled = True

    def test_grouping_none_by_default(self):
        cfg = WeightingConfig()
        assert cfg.grouping is None

    def test_grouping_with_dict(self):
        grouping = {"levels": [{"type": "section", "sections": []}]}
        cfg = WeightingConfig(enabled=True, grouping=grouping)
        assert cfg.grouping == grouping
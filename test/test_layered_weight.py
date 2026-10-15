"""Tests for the layered weighting strategy.

LayeredWeight combines one weight per partitioning layer (e.g. a fixed
section sigma and a residual-driven timeframe weight) into a single weight.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from orbitdet.observations.grouping import Group, GroupList
from orbitdet.observations.weighting.layered import LayeredWeight


def _make_obs_set(times, residuals) -> SimpleNamespace:
    mock_times = [SimpleNamespace(to_float=lambda t=t: float(t)) for t in times]
    return SimpleNamespace(
        residuals=np.asarray(residuals, dtype=float),
        observation_times=mock_times,
        observable_type=1,
    )


def _two_layer_groups(set_id: str) -> GroupList:
    """Section layer (0) + timeframe layer (1) over four observations."""
    pre_section = Group(
        indices=np.array([0, 1, 2], dtype=int),
        group_id=f"{set_id}_pre_encounter",
        level="section",
        layer=0,
        metadata={"section_name": "pre_encounter", "t_start": 0.0, "t_end": 2.0},
    )
    post_section = Group(
        indices=np.array([3], dtype=int),
        group_id=f"{set_id}_post_encounter",
        level="section",
        layer=0,
        metadata={"section_name": "post_encounter", "t_start": 100.0, "t_end": 100.0},
    )
    tf0 = Group(
        indices=np.array([0, 1, 2], dtype=int),
        group_id=f"{set_id}_tf_0000",
        level="timeframe",
        layer=1,
        parent_id=pre_section.group_id,
        metadata={"t_start": 0.0, "t_end": 2.0},
    )
    tf1 = Group(
        indices=np.array([3], dtype=int),
        group_id=f"{set_id}_tf_0001",
        level="timeframe",
        layer=1,
        parent_id=post_section.group_id,
        metadata={"t_start": 100.0, "t_end": 100.0},
    )
    return GroupList([pre_section, post_section, tf0, tf1])


class TestLayeredWeightConstruction:
    def test_requires_layers(self):
        with pytest.raises(ValueError, match="non-empty 'layers'"):
            LayeredWeight(layers=[])

    def test_rejects_unknown_combine(self):
        with pytest.raises(ValueError, match="'combine' must be one of"):
            LayeredWeight(
                layers=[{"level": "timeframe", "mode": "rmse"}],
                combine="mystery",
            )

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="unknown mode"):
            LayeredWeight(layers=[{"level": "timeframe", "mode": "wat"}])

    def test_fixed_requires_sigmas(self):
        with pytest.raises(ValueError, match="requires a 'sigmas'"):
            LayeredWeight(layers=[{"level": "section", "mode": "fixed"}])

    def test_layer_needs_level_or_index(self):
        with pytest.raises(ValueError, match="either 'level' or 'layer'"):
            LayeredWeight(layers=[{"mode": "rmse"}])


class TestLayeredWeightCompute:
    def test_fixed_plus_timeframe_geometric(self):
        """Voyager-style: fixed section sigma combined with timeframe RMSE."""
        strategy = LayeredWeight(
            layers=[
                {
                    "level": "section",
                    "mode": "fixed",
                    "sigmas": {"pre_encounter": {"ra": 1.0, "dec": 1.0},
                               "post_encounter": {"ra": 4.0, "dec": 4.0}},
                },
                {"level": "timeframe", "mode": "rmse_descaled"},
            ],
            combine="geometric",
        )
        # Residuals in radians: ~1e-6 rad ≈ 0.2 arcsec.
        residuals = np.array([[1e-6, 1e-6], [1.2e-6, 1.1e-6], [0.9e-6, 1.0e-6], [5e-6, 5e-6]])
        obs_set = _make_obs_set([0.0, 1.0, 2.0, 100.0], residuals)
        groups = _two_layer_groups("Voyager 2")

        weights, df = strategy.compute_weights(obs_set, groups, "Voyager 2", 0.001)

        assert weights.shape == (8,)
        assert np.all(np.isfinite(weights))
        assert np.all(weights > 0)
        assert len(df) == 4
        # Pre-encounter fixed sigma is tighter than post -> pre weights larger.
        assert weights[0] > weights[6]

    def test_fixed_only_no_timeframes(self):
        """Voyager-style without timeframe clustering."""
        strategy = LayeredWeight(
            layers=[
                {
                    "level": "section",
                    "mode": "fixed",
                    "sigmas": {"pre_encounter": 1.0, "post_encounter": 4.0},
                },
            ],
            combine="geometric",
        )
        residuals = np.array([[1e-6, 1e-6], [1e-6, 1e-6], [1e-6, 1e-6], [1e-6, 1e-6]])
        obs_set = _make_obs_set([0.0, 1.0, 2.0, 100.0], residuals)
        groups = _two_layer_groups("Voyager 2")

        weights, _df = strategy.compute_weights(obs_set, groups, "Voyager 2", 0.001)

        # Pure fixed sigmas: RA and DEC weights equal 1/sigma^2 (in rad).
        arcsec_to_rad = np.pi / (180.0 * 3600.0)
        pre_w = 1.0 / (1.0 * arcsec_to_rad) ** 2
        post_w = 1.0 / (4.0 * arcsec_to_rad) ** 2
        assert weights[0] == pytest.approx(pre_w)
        assert weights[6] == pytest.approx(post_w)

    def test_arithmetic_combine(self):
        strategy = LayeredWeight(
            layers=[
                {"level": "section", "mode": "fixed", "sigmas": {"pre_encounter": 1.0,
                                                                 "post_encounter": 4.0}},
                {"level": "timeframe", "mode": "rmse_descaled"},
            ],
            combine="arithmetic",
        )
        residuals = np.array([[1e-6, 1e-6], [1.2e-6, 1.1e-6], [0.9e-6, 1.0e-6], [5e-6, 5e-6]])
        obs_set = _make_obs_set([0.0, 1.0, 2.0, 100.0], residuals)
        groups = _two_layer_groups("Voyager 2")

        weights, _df = strategy.compute_weights(obs_set, groups, "Voyager 2", 0.001)
        assert weights.shape == (8,)
        assert np.all(np.isfinite(weights))

    def test_missing_layer_raises(self):
        strategy = LayeredWeight(layers=[{"level": "timeframe", "mode": "rmse"}])
        residuals = np.array([[1e-6, 1e-6], [1e-6, 1e-6]])
        obs_set = _make_obs_set([0.0, 1.0], residuals)
        groups = GroupList(
            [Group(indices=np.array([0, 1], dtype=int), group_id="s", level="set", layer=0)]
        )
        with pytest.raises(ValueError, match="no layer matched"):
            strategy.compute_weights(obs_set, groups, "689", 0.001)

    def test_empty_set(self):
        strategy = LayeredWeight(layers=[{"level": "timeframe", "mode": "rmse"}])
        obs_set = _make_obs_set([], np.empty((0, 2)))
        groups = GroupList()
        weights, df = strategy.compute_weights(obs_set, groups, "689", 0.001)
        assert weights.shape == (0,)
        assert df.empty

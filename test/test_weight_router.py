"""Tests for the per-set weighting router.

The router dispatches each set to a sub-strategy by matching rules, enabling
different schemes for Voyager / Gaia / NSDB in a single config.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from orbitdet.observations.grouping import Group, GroupList
from orbitdet.observations.weighting.engine import WeightEngine
from orbitdet.observations.weighting.router import RouterWeight


def _make_obs_set(times, residuals, set_id="689", body_name="Earth", reference_point="689"):
    mock_times = [SimpleNamespace(to_float=lambda t=t: float(t)) for t in times]
    return SimpleNamespace(
        residuals=np.asarray(residuals, dtype=float),
        observation_times=mock_times,
        observable_type=1,
        link_definition=SimpleNamespace(
            link_ends={
                "receiver": SimpleNamespace(body_name=body_name, reference_point=reference_point)
            }
        ),
    )


def _simple_groups(set_id, n):
    return GroupList(
        [
            Group(indices=np.arange(n, dtype=int), group_id=set_id, level="set", layer=-1),
            Group(
                indices=np.arange(n, dtype=int),
                group_id=f"{set_id}_tf_0000",
                level="timeframe",
                layer=0,
                metadata={"t_start": 0.0, "t_end": 1e6},
            ),
        ]
    )


class TestRouterConstruction:
    def test_default_only_is_single_strategy(self):
        """A router with only a default is the degenerate single-strategy case."""
        r = RouterWeight(routes=None, default={"type": "id_v2"})
        assert r.route_source("any").__class__.__name__ == "IDv2Weight"

    def test_no_default_defaults_to_id_v2(self):
        r = RouterWeight(routes=[{"match": {"set_id": "Gaia"}, "source": "id"}])
        assert r.route_source("anything").__class__.__name__ == "IDv2Weight"

    def test_route_requires_match_and_source(self):
        with pytest.raises(ValueError, match="must have 'match' and 'source'"):
            RouterWeight(routes=[{"match": {"set_id": "Gaia"}}])

    def test_unknown_match_key(self):
        with pytest.raises(ValueError, match="Unknown match key"):
            RouterWeight(routes=[{"match": {"colour": "blue"}, "source": "id_v2"}])


class TestRouterMatching:
    def _router(self):
        return RouterWeight(
            routes=[
                {"match": {"set_id_contains": ["Gaia"]}, "source": {"type": "gaia_archive"}},
                {"match": {"set_id_contains": ["Voyager 2"]}, "source": {"type": "id_v2"}},
                {"match": {"set_id": ["689", "675"]}, "source": {"type": "timeframe"}},
            ],
            default={"type": "id"},
        )

    def test_contains_match(self):
        r = self._router()
        assert r.route_source("Gaia").__class__.__name__ == "GaiaArchiveWeight"

    def test_contains_match_case_insensitive(self):
        r = self._router()
        assert r.route_source("gaia").__class__.__name__ == "GaiaArchiveWeight"

    def test_exact_match(self):
        r = self._router()
        assert r.route_source("689").__class__.__name__ == "TFWeight"
        assert r.route_source("675").__class__.__name__ == "TFWeight"

    def test_default_match(self):
        r = self._router()
        assert r.route_source("999").__class__.__name__ == "IDWeight"

    def test_first_route_wins(self):
        r = RouterWeight(
            routes=[
                {"match": {"set_id_contains": ["6"]}, "source": "id"},
                {"match": {"set_id_contains": ["689"]}, "source": "id_v2"},
            ],
            default="timeframe",
        )
        assert r.route_source("689").__class__.__name__ == "IDWeight"

    def test_regex_match(self):
        r = RouterWeight(
            routes=[{"match": {"set_id_matches": [r"^Voyager \d$"]}, "source": "id_v2"}],
            default="id",
        )
        assert r.route_source("Voyager 2").__class__.__name__ == "IDv2Weight"
        assert r.route_source("Voyager 12").__class__.__name__ == "IDWeight"


class TestRouterPreserveFullMatrix:
    def test_true_when_any_route_preserves(self):
        r = RouterWeight(
            routes=[
                {"match": {"set_id_contains": ["Gaia"]}, "source": {"type": "gaia_archive"}},
                {"match": {"set_id_contains": ["Voyager"]}, "source": {"type": "id_v2"}},
            ],
            default="id",
        )
        assert r.preserve_full_weight_matrix is True

    def test_false_when_no_route_preserves(self):
        r = RouterWeight(
            routes=[{"match": {"set_id_contains": ["Voyager"]}, "source": {"type": "id_v2"}}],
            default="id",
        )
        assert r.preserve_full_weight_matrix is False


class TestRouterCompute:
    def test_routes_to_layered(self):
        r = RouterWeight(
            routes=[
                {
                    "match": {"set_id_contains": ["Voyager 2"]},
                    "source": {
                        "type": "layered",
                        "combine": "geometric",
                        "layers": [
                            {"level": "set", "mode": "rmse"},
                            {"level": "timeframe", "mode": "rmse_descaled"},
                        ],
                    },
                }
            ],
            default="id",
        )
        obs_set = _make_obs_set([0.0, 1.0, 2.0], [[1e-6, 1e-6], [1.1e-6, 1e-6], [0.9e-6, 1e-6]])
        groups = _simple_groups("Voyager 2", 3)
        weights, df = r.compute_weights(obs_set, groups, "Voyager 2", 0.001)
        assert weights.shape == (6,)
        assert np.all(np.isfinite(weights))

    def test_routes_to_default_for_unmatched(self):
        r = RouterWeight(
            routes=[{"match": {"set_id_contains": ["Voyager"]}, "source": "id_v2"}],
            default="id",
        )
        obs_set = _make_obs_set([0.0, 1.0], [[1e-6, 1e-6], [1.1e-6, 1e-6]], set_id="689")
        groups = _simple_groups("689", 2)
        weights, _df = r.compute_weights(obs_set, groups, "689", 0.001)
        # ID: single constant weight -> both obs share it.
        assert weights[0] == weights[2]

    def test_per_route_grouping_overrides(self):
        """A route with its own grouping rebuilds groups from that config."""
        r = RouterWeight(
            routes=[
                {
                    "match": {"set_id_contains": ["Voyager 2"]},
                    "source": {
                        "type": "layered",
                        "layers": [
                            {"level": "section", "mode": "fixed",
                             "sigmas": {"pre_encounter": 1.0, "post_encounter": 4.0}},
                        ],
                    },
                    "grouping": {
                        "levels": [
                            {"type": "section", "sections": [ # ignore line too long 
                                {"name": "pre_encounter", "start": -1e30, "end": -326688543.8172541}, # noqa: E501
                                {"name": "post_encounter", "start": -326688543.8172541, "end": 1e30}, # noqa: E501
                            ]},
                        ]
                    },
                }
            ],
            default="id",
        )
        obs_set = _make_obs_set(
            [-326688643.8172541, 618600100.0], [[1e-6, 1e-6], [1e-6, 1e-6]],
            body_name="Voyager 2", reference_point="",
        )
        # Engine-level groups are wrong for this route; the route must rebuild.
        engine_groups = GroupList(
            [Group(indices=np.array([0, 1], dtype=int), group_id="Voyager 2", level="set")]
        )
        weights, _df = r.compute_weights(obs_set, engine_groups, "Voyager 2", 0.001)
        arcsec_to_rad = np.pi / (180.0 * 3600.0)
        assert weights[0] == pytest.approx(1.0 / (1.0 * arcsec_to_rad) ** 2)
        assert weights[2] == pytest.approx(1.0 / (4.0 * arcsec_to_rad) ** 2)


class TestRouterFromConfig:
    def test_engine_builds_router(self):
        cfg = OmegaConf.create(
            {
                "strategy": {
                    "type": "router",
                    "default": {"type": "id_v2"},
                    "routes": [
                        {"match": {"set_id_contains": ["Gaia"]},
                         "source": {"type": "gaia_archive"}},
                        {"match": {"set_id_contains": ["Voyager 2"]},
                         "source": {"type": "layered",
                                    "layers": [{"level": "set", "mode": "rmse"}]}},
                    ],
                },
                "grouping": {"levels": [{"type": "timeframe", "gap_threshold_hours": 4.0}]},
            }
        )
        engine = WeightEngine.from_config(cfg)
        assert engine.strategy.__class__.__name__ == "RouterWeight"
        assert engine.strategy.route_source("Gaia").__class__.__name__ == "GaiaArchiveWeight"
        assert engine.strategy.route_source("Voyager 2").__class__.__name__ == "LayeredWeight"
        assert engine.strategy.route_source("689").__class__.__name__ == "IDv2Weight"

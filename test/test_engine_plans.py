"""Tests for plan-driven dispatch in the weighting and outlier engines.

These cover the wiring rather than the plan model itself (see
``test_plan.py``): that a plan routes each set to the right strategy and
clustering, and that the outlier engine's ``select:`` scoping works.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from omegaconf import OmegaConf

from orbitdet.observations.outlier_rejection.engine import OutlierEngine
from orbitdet.observations.weighting.engine import WeightEngine
from orbitdet.preprocessing.schema import empty_frame

RECEIVER_KEY = "receiver"


@pytest.fixture(autouse=True)
def _patch_links_receiver(monkeypatch):
    import tudatpy.estimation.observable_models_setup

    monkeypatch.setattr(tudatpy.estimation.observable_models_setup.links, "receiver", RECEIVER_KEY)


def _obs_set(set_id, n=2, residuals=None):
    """A mock SingleObservationSet with *n* observations."""
    link_ends = {RECEIVER_KEY: SimpleNamespace(body_name=set_id, reference_point="")}
    times = [SimpleNamespace(to_float=lambda t=float(i): t) for i in range(n)]
    return SimpleNamespace(
        residuals=np.zeros((n, 2)) if residuals is None else residuals,
        observation_times=times,
        observable_type=1,
        link_definition=SimpleNamespace(link_ends=link_ends),
        concatenated_observations=np.zeros(2 * n),
        set_tabulated_weights=MagicMock(),
        set_block_diagonal_weights=MagicMock(),
    )


def _collection(sets):
    collection = MagicMock()
    collection.get_single_observation_sets.return_value = sets
    return collection


def _view(source, code, n=2):
    """A SetView stand-in whose frame carries the routing columns."""
    frame = empty_frame()
    frame["data_source"] = [source] * n
    frame["observatory_code"] = [code] * n
    frame["source_set_id"] = [f"{source}_set"] * n
    frame["reference"] = ["ref"] * n
    frame["coordinate_kind"] = ["ra_dec"] * n
    frame["epoch_tdb"] = [float(i) for i in range(n)]
    return SimpleNamespace(
        frame=frame,
        scan_angles=None,
        column=lambda name: frame[name].to_numpy(),
    )


def _accept_all_strategy(name="mock"):
    """A mock OutlierStrategy that accepts every observation."""
    strategy = MagicMock()

    def apply_side_effect(obs_set, bodies, view=None):
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


@pytest.fixture
def patch_tudat_filtering(monkeypatch):
    """Patch Tudat filtering so the outlier engine runs against mocks."""
    import orbitdet.observations.outlier_rejection.engine as eng

    class MockCollection:
        def __init__(self, sets):
            self._mock_sets = sets

    def fake_create_filtered(obs_set, filt):
        return SimpleNamespace(
            residuals=obs_set.residuals,
            observation_times=obs_set.observation_times,
            link_definition=obs_set.link_definition,
        )

    def fake_build_rejected(original_set, accepted_set):
        return SimpleNamespace(
            residuals=original_set.residuals,
            observation_times=[],
            link_definition=original_set.link_definition,
        )

    monkeypatch.setattr(eng.obs, "create_filtered_observation_set", fake_create_filtered)
    monkeypatch.setattr(eng.obs, "ObservationCollection", MockCollection)
    monkeypatch.setattr(eng, "_build_rejected_set", fake_build_rejected)


# ===========================================================================
# WeightEngine plan dispatch
# ===========================================================================


class TestWeightEnginePlan:
    def _cfg(self):
        return OmegaConf.create(
            {
                "min_sigma_arcsec": 0.01,
                "default": {"strategy": {"type": "id_v2"}},
                "overrides": [
                    {
                        "select": "data_source == 'gaia_archive'",
                        "strategy": {"type": "fixed_scan", "along_scan_sigma_mas": 5.0},
                    }
                ],
                "clustering": {"levels": [{"type": "set"}]},
            }
        )

    def test_from_config_builds_a_plan(self):
        engine = WeightEngine.from_config(self._cfg())
        assert engine.plan is not None
        assert engine.strategy is None
        assert len(engine.plan.overrides) == 1

    def test_plan_routes_to_the_override_strategy(self, monkeypatch):
        # Assert on the routing decision itself, which is what the plan owns;
        # whether a given strategy emits weights is that strategy's business.
        views = {0: _view("gaia_archive", "Gaia"), 1: _view("nsdb", "689")}
        monkeypatch.setattr(
            "orbitdet.analysis.set_view.SetView.build",
            lambda access, set_index: views[set_index],
        )
        engine = WeightEngine.from_config(self._cfg())
        engine.apply(_collection([_obs_set("Gaia"), _obs_set("689")]), MagicMock(), access=object())

        built = engine._plan_strategies
        assert len(built) == 2
        names = {label: s.__class__.__name__ for label, s in built.items()}
        assert any("FixedScan" in n for n in names.values())
        assert any("IDv2" in n for n in names.values())

    def test_plan_strategies_are_built_once(self, monkeypatch):
        views = {0: _view("nsdb", "689"), 1: _view("nsdb", "675")}
        monkeypatch.setattr(
            "orbitdet.analysis.set_view.SetView.build",
            lambda access, set_index: views[set_index],
        )
        engine = WeightEngine.from_config(self._cfg())
        engine.apply(_collection([_obs_set("689"), _obs_set("675")]), MagicMock(), access=object())
        # Both sets matched the default, so exactly one strategy was built.
        assert len(engine._plan_strategies) == 1

    def test_without_access_the_default_entry_is_used(self):
        engine = WeightEngine.from_config(self._cfg())
        sets = [_obs_set("Gaia")]
        engine.apply(_collection(sets), MagicMock())
        # No frame to match on -> default (id_v2) -> tabulated weights.
        sets[0].set_tabulated_weights.assert_called_once()

    def test_config_without_a_default_is_rejected(self):
        # The pre-plan `strategy:` shape is gone.
        cfg = OmegaConf.create(
            {"strategy": {"type": "id_v2"}, "clustering": {"levels": [{"type": "set"}]}}
        )
        with pytest.raises(ValueError, match="no 'default' entry"):
            WeightEngine.from_config(cfg)

    def test_engine_requires_a_strategy_or_a_plan(self):
        with pytest.raises(ValueError, match="either a strategy or a plan"):
            WeightEngine()


# ===========================================================================
# OutlierEngine `select:` scoping
# ===========================================================================


class TestOutlierEngineSelect:
    def _cfg(self, select):
        return OmegaConf.create(
            {
                "strategies": [
                    {
                        "type": "residual_threshold",
                        "threshold_arcsec": 1.5,
                        "select": select,
                    }
                ]
            }
        )

    def test_select_scopes_a_strategy_to_matching_sets(self, monkeypatch, patch_tudat_filtering):
        views = {0: _view("nsdb", "689"), 1: _view("gaia_archive", "Gaia")}
        monkeypatch.setattr(
            "orbitdet.analysis.set_view.SetView.build",
            lambda access, set_index: views[set_index],
        )
        engine = OutlierEngine.from_config(self._cfg("data_source == 'nsdb'"))
        # Swap in a mock so this test exercises routing, not Tudat filtering.
        engine._strategies[0].strategy = _accept_all_strategy()
        _, _, summary = engine.apply_with_rejected(
            _collection([_obs_set("689"), _obs_set("Gaia")]), MagicMock(), access=object()
        )
        # Only the nsdb set had a strategy applied to it.
        applied = [s for s in summary["per_set"] if s["strategies"]]
        assert len(applied) == 1
        assert applied[0]["set_id"].startswith("nsdb_set")

    def test_select_matching_nothing_applies_no_strategies(
        self, monkeypatch, patch_tudat_filtering
    ):
        monkeypatch.setattr(
            "orbitdet.analysis.set_view.SetView.build",
            lambda access, set_index: _view("nsdb", "689"),
        )
        engine = OutlierEngine.from_config(self._cfg("data_source == 'nope'"))
        engine._strategies[0].strategy = _accept_all_strategy()
        _, _, summary = engine.apply_with_rejected(
            _collection([_obs_set("689")]), MagicMock(), access=object()
        )
        assert summary["per_set"][0]["strategies"] == []

    def test_sets_key_is_no_longer_accepted(self):
        # `sets:` was replaced by `select:`; an unknown key must not be silently
        # swallowed into the strategy's constructor kwargs.
        cfg = OmegaConf.create(
            {
                "strategies": [
                    {"type": "residual_threshold", "threshold_arcsec": 1.5, "sets": ["689"]}
                ]
            }
        )
        with pytest.raises(TypeError):
            OutlierEngine.from_config(cfg)

    def test_invalid_select_is_rejected_at_load_time(self):
        with pytest.raises(ValueError, match="unknown column"):
            OutlierEngine.from_config(self._cfg("nope == 1"))

    def test_select_and_clusters_combine(self):
        cfg = OmegaConf.create(
            {
                "strategies": [
                    {
                        "type": "residual_threshold",
                        "threshold_arcsec": 1.5,
                        "select": "data_source == 'nsdb'",
                        "clusters": ["night_1"],
                    }
                ]
            }
        )
        engine = OutlierEngine.from_config(cfg)
        scoped = engine._strategies[0]
        assert scoped.cluster_filter == {"night_1"}
        assert scoped.selection is not None

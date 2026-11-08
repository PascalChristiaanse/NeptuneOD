"""Tests for cluster-scoped outlier rejection.

The outlier subsystem has its **own** clustering, independent of weighting.
Strategies may carry a ``clusters`` filter (matching cluster names such as section
names or timeframe ids) in addition to the existing ``sets`` filter.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from omegaconf import OmegaConf

import orbitdet.observations.outlier_rejection.engine as eng
from orbitdet.observations.outlier_rejection.base import OutlierStrategy
from orbitdet.observations.outlier_rejection.engine import (
    OutlierEngine,
    _ScopedStrategy,
)

RECEIVER_KEY = "receiver"


@pytest.fixture(autouse=True)
def _patch_links_receiver(monkeypatch):
    import tudatpy.estimation.observable_models_setup

    monkeypatch.setattr(tudatpy.estimation.observable_models_setup.links, "receiver", RECEIVER_KEY)


def _mock_time(epoch: float):
    return SimpleNamespace(to_float=lambda: float(epoch))


def _make_obs_set(epochs, set_id="689", body_name="Earth", reference_point="689"):
    link_ends = {
        RECEIVER_KEY: SimpleNamespace(body_name=body_name, reference_point=reference_point)
    }
    return SimpleNamespace(
        residuals=MagicMock(),
        observation_times=[_mock_time(e) for e in epochs],
        link_definition=SimpleNamespace(link_ends=link_ends),
    )


def _make_collection(sets):
    collection = MagicMock()
    collection.get_single_observation_sets.return_value = sets
    return collection


@pytest.fixture
def patch_tudat_filtering(monkeypatch):
    """Patch Tudat filtering so cluster scoping works with mocks."""

    class MockCollection:
        def __init__(self, sets):
            self._mock_sets = sets

    def fake_time_bounds_filter(t_start, t_end, filter_out):
        return ("time", float(t_start), float(t_end), filter_out)

    def fake_epochs_filter(epochs, filter_out):
        return ("epochs", [float(e) for e in epochs], filter_out)

    def fake_create_filtered(obs_set, filt):
        epochs = [t.to_float() for t in obs_set.observation_times]
        kind = filt[0]
        if kind == "time":
            _, t_start, t_end, filter_out = filt
            keep = [e for e in epochs if (t_start <= e <= t_end) != filter_out]
        else:  # epochs
            _, target, filter_out = filt
            target = set(target)
            keep = [e for e in epochs if (e in target) != filter_out]
        return SimpleNamespace(
            residuals=obs_set.residuals,
            observation_times=[_mock_time(e) for e in keep],
            link_definition=obs_set.link_definition,
        )

    def fake_build_rejected(original_set, accepted_set):
        orig = {t.to_float() for t in original_set.observation_times}
        kept = {t.to_float() for t in accepted_set.observation_times}
        rejected = sorted(orig - kept)
        return SimpleNamespace(
            residuals=original_set.residuals,
            observation_times=[_mock_time(e) for e in rejected],
            link_definition=original_set.link_definition,
        )

    monkeypatch.setattr(eng, "_time_bounds_filter", fake_time_bounds_filter)
    monkeypatch.setattr(eng, "_epochs_filter", fake_epochs_filter)
    monkeypatch.setattr(eng.obs, "create_filtered_observation_set", fake_create_filtered)
    monkeypatch.setattr(eng.obs, "ObservationCollection", MockCollection)
    monkeypatch.setattr(eng, "_build_rejected_set", fake_build_rejected)


def _reject_above_strategy(threshold: float):
    """Strategy that rejects observations whose first residual exceeds *threshold*."""
    strategy = MagicMock(spec=OutlierStrategy)

    def apply_side_effect(obs_set, bodies, view=None):
        residuals = np.asarray(obs_set.residuals, dtype=float)
        epochs = [t.to_float() for t in obs_set.observation_times]
        keep_epochs, reject_epochs = [], []
        for i, e in enumerate(epochs):
            if residuals.ndim and residuals.shape[0] > i and residuals[i][0] > threshold:
                reject_epochs.append(e)
            else:
                keep_epochs.append(e)
        return (
            SimpleNamespace(
                residuals=obs_set.residuals,
                observation_times=[_mock_time(e) for e in keep_epochs],
                link_definition=obs_set.link_definition,
            ),
            {
                "strategy": "mock_threshold",
                "n_accepted": len(keep_epochs),
                "n_rejected": len(reject_epochs),
                "n_total": len(epochs),
                "rejected_epochs": reject_epochs,
            },
        )

    strategy.apply.side_effect = apply_side_effect
    return strategy


# ===========================================================================
# _ScopedStrategy cluster filter
# ===========================================================================


class TestScopedStrategyGroupFilter:
    def test_no_group_filter_applies_to_all(self):
        scoped = _ScopedStrategy(MagicMock(spec=OutlierStrategy))
        assert scoped.applies_to_cluster("anything") is True

    def test_group_filter_matches(self):
        scoped = _ScopedStrategy(
            MagicMock(spec=OutlierStrategy), cluster_filter={"night_1", "night_2"}
        )
        assert scoped.applies_to_cluster("night_1") is True
        assert scoped.applies_to_cluster("night_3") is False


# ===========================================================================
# from_config with clustering + clusters filter
# ===========================================================================


class TestOutlierEngineFromConfigGrouping:
    def test_parses_clustering(self):
        cfg = OmegaConf.create(
            {
                "clustering": {"levels": [{"type": "timeframe", "gap_threshold_hours": 4.0}]},
                "strategies": [{"type": "residual_threshold", "threshold_arcsec": 1.5}],
            }
        )
        engine = OutlierEngine.from_config(cfg)
        assert engine._clustering is not None

    def test_parses_group_filter(self):
        cfg = OmegaConf.create(
            {
                "strategies": [
                    {
                        "type": "residual_threshold",
                        "threshold_arcsec": 0.5,
                        "clusters": ["pre_encounter"],
                    }
                ],
            }
        )
        engine = OutlierEngine.from_config(cfg)
        scoped = engine._strategies[0]
        assert scoped.cluster_filter == {"pre_encounter"}
        assert scoped.selection is None

    def test_cluster_and_select_filters_together(self):
        cfg = OmegaConf.create(
            {
                "strategies": [
                    {
                        "type": "residual_threshold",
                        "threshold_arcsec": 0.5,
                        "select": "observatory_code == 'Voyager 2'",
                        "clusters": ["pre_encounter"],
                    }
                ],
            }
        )
        engine = OutlierEngine.from_config(cfg)
        scoped = engine._strategies[0]
        assert scoped.cluster_filter == {"pre_encounter"}
        assert scoped.selection is not None


# ===========================================================================
# Cluster-scoped application
# ===========================================================================


class TestGroupScopedApplication:
    def test_rejects_only_within_named_group(self, patch_tudat_filtering):
        # Two nights; large residual only in the first night.
        epochs = [0.0, 3600.0, 100000.0, 1003600.0]
        residuals = np.array([[9e-6, 9e-6], [9e-6, 9e-6], [1e-7, 1e-7], [1e-7, 1e-7]])
        obs_set = _make_obs_set(epochs)
        obs_set.residuals = residuals
        collection = _make_collection([obs_set])

        clustering = OmegaConf.create(
            {"levels": [{"type": "timeframe", "gap_threshold_hours": 4.0}]}
        )
        # Cluster ids are "<set>_tf_0000"/"<set>_tf_0001"; match the first night.
        scoped = _ScopedStrategy(
            _reject_above_strategy(1e-6),
            cluster_filter={"689_tf_0000"},
        )
        engine = OutlierEngine([scoped], clustering=clustering)

        filtered, summary = engine.apply(collection, MagicMock())

        # Only the two observations in the first night are candidates; both are
        # rejected, leaving the two from the second night.
        assert summary["n_accepted"] == 2
        assert summary["n_rejected"] == 2
        assert summary["n_total_observations"] == 4

    def test_unmatched_group_filter_rejects_nothing(self, patch_tudat_filtering):
        # Cluster ids are "<set>_tf_0000"/"<set>_tf_0001"; name a cluster that does
        # not exist -> no observations are scoped, so nothing is rejected.
        epochs = [0.0, 3600.0, 100000.0, 1003600.0]
        residuals = np.array([[9e-6, 9e-6], [9e-6, 9e-6], [9e-6, 9e-6], [9e-6, 9e-6]])
        obs_set = _make_obs_set(epochs)
        obs_set.residuals = residuals
        collection = _make_collection([obs_set])

        clustering = OmegaConf.create(
            {"levels": [{"type": "timeframe", "gap_threshold_hours": 4.0}]}
        )
        scoped = _ScopedStrategy(
            _reject_above_strategy(1e-6),
            cluster_filter={"689_tf_0099"},  # does not exist
        )
        engine = OutlierEngine([scoped], clustering=clustering)

        _, summary = engine.apply(collection, MagicMock())

        assert summary["n_rejected"] == 0
        assert summary["n_accepted"] == 4

    def test_cluster_filter_without_clustering_is_whole_set(self, patch_tudat_filtering):
        # No clustering config -> cluster_filter is effectively ignored (whole set).
        epochs = [0.0, 3600.0]
        obs_set = _make_obs_set(epochs)
        obs_set.residuals = np.array([[9e-6, 9e-6], [9e-6, 9e-6]])
        collection = _make_collection([obs_set])

        scoped = _ScopedStrategy(_reject_above_strategy(1e-6), cluster_filter={"whatever"})
        engine = OutlierEngine([scoped], clustering=None)

        _, summary = engine.apply(collection, MagicMock())

        # Falls back to whole-set application -> both rejected.
        assert summary["n_rejected"] == 2


# ===========================================================================
# Section-based clustering (Voyager pre/post encounter)
# ===========================================================================


class TestSectionScopedApplication:
    def test_pre_encounter_tighter_than_post(self, patch_tudat_filtering):
        encounter = -326688543.8172541
        epochs = [
            encounter - 100.0,  # pre
            encounter + 100.0,  # post
        ]
        residuals = np.array([[9e-6, 9e-6], [9e-6, 9e-6]])
        obs_set = _make_obs_set(epochs)
        obs_set.residuals = residuals
        collection = _make_collection([obs_set])

        clustering = OmegaConf.create(
            {
                "levels": [
                    {
                        "type": "section",
                        "sections": [
                            {"name": "pre_encounter", "start": -1e30, "end": encounter},
                            {"name": "post_encounter", "start": encounter, "end": 1e30},
                        ],
                    }
                ]
            }
        )
        scoped = _ScopedStrategy(
            _reject_above_strategy(1e-6),
            cluster_filter={"pre_encounter"},
        )
        engine = OutlierEngine([scoped], clustering=clustering)

        _, summary = engine.apply(collection, MagicMock())

        # Only the pre-encounter observation is scoped and rejected.
        assert summary["n_rejected"] == 1
        assert summary["n_accepted"] == 1

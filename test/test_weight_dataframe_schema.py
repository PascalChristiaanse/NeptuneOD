"""Regression tests for the shared weights-DataFrame schema.

Every weighting strategy must emit the same columns, because the engine
concatenates the per-set frames and the visualization layer clusters on
``cluster_id``.  A strategy that omits a column (e.g. ``cluster_id``) produces
``NaN`` for its rows and breaks downstream sorting/clustering.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from orbitdet.observations.clustering import Cluster, ClusterList
from orbitdet.observations.weighting.dataframe import WEIGHTS_DF_COLUMNS
from orbitdet.observations.weighting.layered import LayeredWeight
from orbitdet.observations.weighting.strategies import (
    GaiaArchiveWeight,
    HybridArithmeticWeight,
    HybridGeometricV2Weight,
    HybridGeometricWeight,
    IDv2Weight,
    IDWeight,
    TFFreeWeight,
    TFWeight,
)


def _make_obs_set(times, residuals):
    mock_times = [SimpleNamespace(to_float=lambda t=t: float(t)) for t in times]
    return SimpleNamespace(
        residuals=np.asarray(residuals, dtype=float),
        observation_times=mock_times,
        observable_type=1,
    )


def _groups(set_id: str, n: int) -> ClusterList:
    return ClusterList(
        [
            Cluster(indices=np.arange(n, dtype=int), cluster_id=set_id, level="set", layer=-1),
            Cluster(
                indices=np.arange(n, dtype=int),
                cluster_id=f"{set_id}_tf_0000",
                level="timeframe",
                layer=0,
                metadata={"t_start": 0.0, "t_end": 1e6},
            ),
        ]
    )


#: (strategy, needs_timeframe_groups)
_STRATEGIES = [
    (IDWeight(), False),
    (IDv2Weight(), True),
    (TFWeight(), True),
    (TFFreeWeight(), True),
    (HybridGeometricWeight(), True),
    (HybridGeometricV2Weight(), True),
    (HybridArithmeticWeight(), True),
    (GaiaArchiveWeight(), False),
    (
        LayeredWeight(
            layers=[
                {"level": "set", "mode": "rmse"},
                {"level": "timeframe", "mode": "rmse_descaled"},
            ]
        ),
        True,
    ),
]


@pytest.mark.parametrize("strategy,_needs_tf", _STRATEGIES, ids=lambda s: type(s).__name__)
def test_strategy_emits_full_schema(strategy, _needs_tf):
    """Every strategy must emit all standard columns, with no NaN cluster_id."""
    residuals = np.array([[1e-6, 1e-6], [1.2e-6, 1.1e-6], [0.9e-6, 1.0e-6]])
    obs_set = _make_obs_set([0.0, 1.0, 2.0], residuals)
    clusters = _groups("689", 3)

    weights, df = strategy.compute_weights(obs_set, clusters, "689", 0.001)

    assert weights.shape == (6,)
    missing = [c for c in WEIGHTS_DF_COLUMNS if c not in df.columns]
    assert not missing, f"{type(strategy).__name__} missing columns: {missing}"
    # cluster_id must be a string for every row (never NaN/float).
    assert df["cluster_id"].map(lambda v: isinstance(v, str)).all()
    assert (df["cluster_id"] != "").all()


def test_layered_group_id_is_string_not_nan():
    """Regression: layered previously omitted cluster_id, yielding NaN on concat."""
    strategy = LayeredWeight(
        layers=[
            {
                "level": "section",
                "mode": "fixed",
                "sigmas": {"pre_encounter": 1.0, "post_encounter": 4.0},
            },
            {"level": "timeframe", "mode": "rmse_descaled"},
        ]
    )
    residuals = np.array([[1e-6, 1e-6], [1e-6, 1e-6], [1e-6, 1e-6], [1e-6, 1e-6]])
    obs_set = _make_obs_set([0.0, 1.0, 2.0, 100.0], residuals)
    clusters = ClusterList(
        [
            Cluster(
                indices=np.array([0, 1, 2], dtype=int),
                cluster_id="689_pre_encounter",
                level="section",
                layer=0,
                metadata={"section_name": "pre_encounter"},
            ),
            Cluster(
                indices=np.array([3], dtype=int),
                cluster_id="689_post_encounter",
                level="section",
                layer=0,
                metadata={"section_name": "post_encounter"},
            ),
            Cluster(
                indices=np.array([0, 1, 2], dtype=int),
                cluster_id="689_tf_0000",
                level="timeframe",
                layer=1,
                metadata={"t_start": 0.0, "t_end": 2.0},
            ),
            Cluster(
                indices=np.array([3], dtype=int),
                cluster_id="689_tf_0001",
                level="timeframe",
                layer=1,
                metadata={"t_start": 100.0, "t_end": 100.0},
            ),
        ]
    )

    _weights, df = strategy.compute_weights(obs_set, clusters, "689", 0.001)

    assert "cluster_id" in df.columns
    assert df["cluster_id"].tolist() == ["689_tf_0000"] * 3 + ["689_tf_0001"]
    # Sorting the unique values must not raise (the original crash).
    assert sorted(df["cluster_id"].unique()) == ["689_tf_0000", "689_tf_0001"]

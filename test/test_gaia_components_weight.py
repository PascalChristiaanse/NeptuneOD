"""Tests for the Gaia random/systematic component weighting strategy.

``GaiaComponentsWeight`` rebuilds the archive transit covariance as

    C = random_scale * C_random + systematic_scale * C_systematic

and inverts it, so the intra-transit systematic correlation can be tuned
independently of the random noise level.  The two components are read from the
cache populated by the Gaia observation factory.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import block_diag

from orbitdet.observations.clustering import Cluster, ClusterList
from orbitdet.observations.weighting.gaia_cache import (
    clear_gaia_weights,
    get_gaia_covariance_components,
    store_gaia_covariance_components,
)
from orbitdet.observations.weighting.strategies import GaiaComponentsWeight

SET_ID = "Gaia"


def _gaia_astrometry_available() -> bool:
    """True when the real (non-stubbed) tudatpy GaiaAstrometry interface is importable."""
    try:
        from tudatpy.data.gaia import GaiaAstrometry

        GaiaAstrometry(pd.DataFrame({"number_mp": [1]}))
    except Exception:
        return False
    return True


requires_tudatpy_gaia = pytest.mark.skipif(
    not _gaia_astrometry_available(),
    reason="tudatpy.data.gaia is stubbed/unavailable in this environment",
)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_gaia_weights()
    yield
    clear_gaia_weights()


def _make_obs_set(n_obs, set_id=SET_ID):
    written = {}

    def _set_full_weight_matrix(matrix):
        written["matrix"] = np.asarray(matrix)

    mock_times = [SimpleNamespace(to_float=lambda t=t: float(t)) for t in range(n_obs)]
    obs_set = SimpleNamespace(
        residuals=np.zeros((n_obs, 2)),
        observation_times=mock_times,
        observable_type=1,
        # Declinations are only read when scan angles are available; a small
        # non-zero value keeps the cos(dec) factor in play.
        concatenated_observations=np.tile(np.array([0.0, 0.2]), n_obs),
        has_full_weight_matrix_contribution=True,
        set_full_weight_matrix=_set_full_weight_matrix,
    )
    obs_set._written = written
    return obs_set


def _clusters(set_id, n):
    return ClusterList(
        [Cluster(indices=np.arange(n, dtype=int), cluster_id=set_id, level="set", layer=-1)]
    )


def _synthetic_components(n_obs=2):
    """One transit with a diagonal random block and a fully-correlated systematic block."""
    ra_r = 2e-6
    dec_r = 3e-6
    ra_s = 1e-7
    dec_s = 2e-7
    random_block = block_diag(*[np.diag([ra_r**2, dec_r**2]) for _ in range(n_obs)])
    systematic_block = np.tile(np.diag([ra_s**2, dec_s**2]), (n_obs, n_obs))
    return [random_block], [systematic_block]


# ===========================================================================
# Construction / validation
# ===========================================================================


class TestConstruction:
    def test_defaults(self):
        s = GaiaComponentsWeight()
        assert s.random_scale_al == 1.0
        assert s.random_scale_ac == 1.0
        assert s.systematic_scale_al == 1.0
        assert s.systematic_scale_ac == 1.0

    def test_custom_per_axis_values(self):
        s = GaiaComponentsWeight(
            random_scale_al=2.0,
            random_scale_ac=3.0,
            systematic_scale_al=0.5,
            systematic_scale_ac=0.25,
        )
        assert s.random_scale_al == 2.0
        assert s.random_scale_ac == 3.0
        assert s.systematic_scale_al == 0.5
        assert s.systematic_scale_ac == 0.25

    def test_shorthand_sets_both_axes(self):
        s = GaiaComponentsWeight(random_scale=2.0, systematic_scale=0.5)
        assert s.random_scale_al == 2.0
        assert s.random_scale_ac == 2.0
        assert s.systematic_scale_al == 0.5
        assert s.systematic_scale_ac == 0.5

    def test_explicit_axis_overrides_shorthand(self):
        s = GaiaComponentsWeight(random_scale=2.0, random_scale_ac=7.0)
        assert s.random_scale_al == 2.0
        assert s.random_scale_ac == 7.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"random_scale": 0.0},
            {"random_scale": -1.0},
            {"random_scale": np.nan},
            {"random_scale": np.inf},
            {"systematic_scale": 0.0},
            {"systematic_scale": -1.0},
            {"systematic_scale": np.nan},
            {"random_scale_al": 0.0},
            {"random_scale_ac": -2.0},
            {"systematic_scale_al": np.nan},
            {"systematic_scale_ac": np.inf},
        ],
    )
    def test_rejects_bad_scales(self, kwargs):
        with pytest.raises(ValueError):
            GaiaComponentsWeight(**kwargs)

    def test_preserves_full_weight_matrix(self):
        assert GaiaComponentsWeight().preserve_full_weight_matrix is True


# ===========================================================================
# Cache round-trip
# ===========================================================================


class TestComponentCache:
    def test_store_and_get(self):
        random_blocks, systematic_blocks = _synthetic_components()
        store_gaia_covariance_components(SET_ID, random_blocks, systematic_blocks)
        cached = get_gaia_covariance_components(SET_ID)
        assert cached is not None
        got_random, got_systematic = cached
        np.testing.assert_allclose(got_random[0], random_blocks[0])
        np.testing.assert_allclose(got_systematic[0], systematic_blocks[0])

    def test_stored_blocks_are_copies(self):
        random_blocks, systematic_blocks = _synthetic_components()
        store_gaia_covariance_components(SET_ID, random_blocks, systematic_blocks)
        random_blocks[0][:] = 0.0
        got_random, _ = get_gaia_covariance_components(SET_ID)
        assert np.any(got_random[0] != 0.0)

    def test_missing_returns_none(self):
        assert get_gaia_covariance_components("nope") is None


# ===========================================================================
# Weight computation
# ===========================================================================


class TestComputeWeights:
    def test_identity_scales_match_direct_inverse(self):
        """scales == 1 reproduces inv(C_random + C_systematic) exactly."""
        random_blocks, systematic_blocks = _synthetic_components()
        store_gaia_covariance_components(SET_ID, random_blocks, systematic_blocks)

        n_obs = 2
        obs_set = _make_obs_set(n_obs)
        weights, df = GaiaComponentsWeight().compute_weights(
            obs_set, _clusters(SET_ID, n_obs), SET_ID, 0.001
        )

        total = random_blocks[0] + systematic_blocks[0]
        expected_w = np.linalg.inv(total)
        expected_diag = np.diag(expected_w)
        np.testing.assert_allclose(weights, expected_diag, rtol=1e-9)
        assert weights.shape == (2 * n_obs,)
        assert df["strategy"].unique().tolist() == ["gaia_components"]

    def test_systematic_scale_changes_weights(self):
        random_blocks, systematic_blocks = _synthetic_components()
        store_gaia_covariance_components(SET_ID, random_blocks, systematic_blocks)

        base, _ = GaiaComponentsWeight().compute_weights(
            _make_obs_set(2), _clusters(SET_ID, 2), SET_ID, 0.001
        )
        down, _ = GaiaComponentsWeight(systematic_scale=0.01).compute_weights(
            _make_obs_set(2), _clusters(SET_ID, 2), SET_ID, 0.001
        )
        # Less systematic covariance → smaller total covariance → larger weights.
        assert np.all(down > base)

    def test_random_scale_changes_weights(self):
        random_blocks, systematic_blocks = _synthetic_components()
        store_gaia_covariance_components(SET_ID, random_blocks, systematic_blocks)

        base, _ = GaiaComponentsWeight().compute_weights(
            _make_obs_set(2), _clusters(SET_ID, 2), SET_ID, 0.001
        )
        inflated, _ = GaiaComponentsWeight(random_scale=4.0).compute_weights(
            _make_obs_set(2), _clusters(SET_ID, 2), SET_ID, 0.001
        )
        # More random covariance → larger total covariance → smaller weights.
        assert np.all(inflated < base)

    def test_rewrites_full_matrix_when_present(self):
        random_blocks, systematic_blocks = _synthetic_components()
        store_gaia_covariance_components(SET_ID, random_blocks, systematic_blocks)

        written = {}

        def _capture(matrix):
            written["matrix"] = np.asarray(matrix)

        obs_set = _make_obs_set(2)
        obs_set.set_full_weight_matrix = _capture
        GaiaComponentsWeight().compute_weights(obs_set, _clusters(SET_ID, 2), SET_ID, 0.001)

        assert "matrix" in written
        assert written["matrix"].shape == (4, 4)
        # Symmetric
        np.testing.assert_allclose(written["matrix"], written["matrix"].T)

    def test_empty_set(self):
        weights, df = GaiaComponentsWeight().compute_weights(
            _make_obs_set(0), _clusters(SET_ID, 0), SET_ID, 0.001
        )
        assert weights.shape == (0,)
        assert df.empty

    def test_fallback_without_cache(self):
        weights, df = GaiaComponentsWeight().compute_weights(
            _make_obs_set(3), _clusters(SET_ID, 3), SET_ID, 0.001
        )
        assert weights.shape == (6,)
        assert np.all(np.isfinite(weights))
        assert np.all(weights > 0)
        assert df["strategy"].unique().tolist() == ["gaia_components"]

    def test_fallback_ignores_size_mismatch(self):
        """Cached components for a different number of observations fall back."""
        random_blocks, systematic_blocks = _synthetic_components(n_obs=5)
        store_gaia_covariance_components(SET_ID, random_blocks, systematic_blocks)
        weights, _ = GaiaComponentsWeight().compute_weights(
            _make_obs_set(2), _clusters(SET_ID, 2), SET_ID, 0.001
        )
        assert weights.shape == (4,)
        assert np.all(np.isfinite(weights))


# ===========================================================================
# Per-axis (scan-frame) scaling
# ===========================================================================


def _view(scan_angles):
    """Minimal SetView stand-in exposing only ``scan_angles``."""
    return SimpleNamespace(scan_angles=np.asarray(scan_angles, dtype=float))


class TestPerAxisScaling:
    def test_isotropic_factors_match_shorthand(self):
        """Equal AL/AC factors reproduce the single-factor result exactly."""
        random_blocks, systematic_blocks = _synthetic_components()
        store_gaia_covariance_components(SET_ID, random_blocks, systematic_blocks)

        shorthand, _ = GaiaComponentsWeight(random_scale=2.0, systematic_scale=0.5).compute_weights(
            _make_obs_set(2), _clusters(SET_ID, 2), SET_ID, 0.001
        )
        per_axis, _ = GaiaComponentsWeight(
            random_scale_al=2.0,
            random_scale_ac=2.0,
            systematic_scale_al=0.5,
            systematic_scale_ac=0.5,
        ).compute_weights(_make_obs_set(2), _clusters(SET_ID, 2), SET_ID, 0.001)
        np.testing.assert_allclose(per_axis, shorthand, rtol=1e-12)

    def test_axes_are_independent(self):
        """Scaling only AL differs from scaling only AC."""
        random_blocks, systematic_blocks = _synthetic_components()
        store_gaia_covariance_components(SET_ID, random_blocks, systematic_blocks)

        al_only, _ = GaiaComponentsWeight(random_scale_al=4.0).compute_weights(
            _make_obs_set(2), _clusters(SET_ID, 2), SET_ID, 0.001
        )
        ac_only, _ = GaiaComponentsWeight(random_scale_ac=4.0).compute_weights(
            _make_obs_set(2), _clusters(SET_ID, 2), SET_ID, 0.001
        )
        assert not np.allclose(al_only, ac_only)

    def test_scan_frame_rotation_preserves_isotropic_scaling(self):
        """With equal factors the scan-frame path equals the RA/Dec path."""
        random_blocks, systematic_blocks = _synthetic_components()
        store_gaia_covariance_components(SET_ID, random_blocks, systematic_blocks)

        obs_set = _make_obs_set(2)
        no_view, _ = GaiaComponentsWeight(random_scale_al=3.0, random_scale_ac=3.0).compute_weights(
            obs_set, _clusters(SET_ID, 2), SET_ID, 0.001
        )
        with_view, _ = GaiaComponentsWeight(
            random_scale_al=3.0, random_scale_ac=3.0
        ).compute_weights(
            _make_obs_set(2), _clusters(SET_ID, 2), SET_ID, 0.001, view=_view([0.3, 1.1])
        )
        np.testing.assert_allclose(with_view, no_view, rtol=1e-10)

    def test_scan_frame_rotation_matches_documented_convention(self):
        """T maps a (RA, Dec) offset to (AL, AC) as the docstring states."""
        from orbitdet.observations.weighting.strategies import _scan_frame_rotation

        psi = 0.7
        dec = 0.4
        rotation = _scan_frame_rotation(np.array([psi]), np.array([dec]))[0]

        dra, ddec = 1e-6, 2e-6
        east = dra * np.cos(dec)
        north = ddec
        expected_al = east * np.sin(psi) + north * np.cos(psi)
        expected_ac = east * np.cos(psi) - north * np.sin(psi)

        np.testing.assert_allclose(rotation @ np.array([dra, ddec]), [expected_al, expected_ac])

    def test_scan_frame_rotation_determinant_carries_cos_dec(self):
        """T is a Jacobian, not a pure rotation: |det T| == cos(dec)."""
        from orbitdet.observations.weighting.strategies import _scan_frame_rotation

        psi = np.array([0.0, 0.7, 2.1])
        dec = np.array([0.0, 0.4, -0.9])
        rotation = _scan_frame_rotation(psi, dec)
        for k in range(len(psi)):
            assert np.isclose(abs(np.linalg.det(rotation[k])), np.cos(dec[k]))

    def test_scan_frame_rotation_matches_fixed_scan_convention(self):
        """psi=0 puts AL along Dec and AC along RA, as FixedScanWeight assumes."""
        from orbitdet.observations.weighting.strategies import _scan_frame_rotation

        rotation = _scan_frame_rotation(np.array([0.0]), np.array([0.0]))[0]
        # AL = north = dDec, AC = east = dRA.
        np.testing.assert_allclose(rotation, np.array([[0.0, 1.0], [1.0, 0.0]]))

    def test_scale_in_scan_frame_is_symmetric_positive(self):
        """The scaled covariance stays symmetric and positive definite."""
        from orbitdet.observations.weighting.strategies import (
            _scale_in_scan_frame,
            _scan_frame_rotation,
        )

        random_blocks, systematic_blocks = _synthetic_components()
        total = random_blocks[0] + systematic_blocks[0]
        rotation = _scan_frame_rotation(np.array([0.4, 1.9]), np.array([0.1, -0.3]))
        scaled = _scale_in_scan_frame(total, rotation, 2.0, 0.5)

        np.testing.assert_allclose(scaled, scaled.T, atol=1e-18)
        assert np.all(np.linalg.eigvalsh(scaled) > 0.0)

    def test_missing_scan_angles_falls_back_to_radec(self):
        """No view -> factors applied directly in RA/Dec, still finite."""
        random_blocks, systematic_blocks = _synthetic_components()
        store_gaia_covariance_components(SET_ID, random_blocks, systematic_blocks)

        weights, _ = GaiaComponentsWeight(random_scale_al=2.0, random_scale_ac=5.0).compute_weights(
            _make_obs_set(2), _clusters(SET_ID, 2), SET_ID, 0.001
        )
        assert weights.shape == (4,)
        assert np.all(np.isfinite(weights))
        assert np.all(weights > 0.0)


# ===========================================================================
# tudatpy covariance-component split
# ===========================================================================


@requires_tudatpy_gaia
class TestGaiaAstrometryComponentSplit:
    def _make_astrometry(self):
        from tudatpy.data.gaia import GaiaAstrometry

        n = 3
        table = {
            "number_mp": [1] * n,
            "transit_id": [0] * n,
            "ra_error_random": [2e-6] * n,
            "dec_error_random": [3e-6] * n,
            "ra_dec_correlation_random": [0.1] * n,
            "ra_error_systematic": [1e-7] * n,
            "dec_error_systematic": [2e-7] * n,
            "ra_dec_correlation_systematic": [0.2] * n,
        }
        return GaiaAstrometry(pd.DataFrame(table))

    def test_components_sum_to_total(self):
        astrometry = self._make_astrometry()
        random_blocks, systematic_blocks = astrometry.get_observation_covariance_components(1)
        total_blocks = astrometry._get_observation_covariance(1)

        assert len(random_blocks) == len(systematic_blocks) == len(total_blocks) == 1
        np.testing.assert_allclose(random_blocks[0] + systematic_blocks[0], total_blocks[0])

    def test_scales_applied_per_component(self):
        astrometry = self._make_astrometry()
        random_blocks, systematic_blocks = astrometry.get_observation_covariance_components(
            1, random_scale=2.0, systematic_scale=0.5
        )
        base_random, base_systematic = astrometry.get_observation_covariance_components(1)
        np.testing.assert_allclose(random_blocks[0], 2.0 * base_random[0])
        np.testing.assert_allclose(systematic_blocks[0], 0.5 * base_systematic[0])

    def test_systematic_block_is_fully_correlated(self):
        """The systematic component is constant over a transit → all equal entries."""
        astrometry = self._make_astrometry()
        _, systematic_blocks = astrometry.get_observation_covariance_components(1)
        block = systematic_blocks[0]
        # 3 observations → 6x6, every 2x2 sub-block identical.
        first = block[0:2, 0:2]
        for i in range(3):
            for j in range(3):
                np.testing.assert_allclose(block[2 * i : 2 * i + 2, 2 * j : 2 * j + 2], first)

    def test_random_block_is_diagonal(self):
        astrometry = self._make_astrometry()
        random_blocks, _ = astrometry.get_observation_covariance_components(1)
        block = random_blocks[0]
        off_diag_block = block[0:2, 2:4]
        np.testing.assert_allclose(off_diag_block, 0.0)

"""Tests for the inverse a priori covariance builder.

The builder reads block sizes/ordering from the consolidated parameter set, so
these tests use a lightweight :class:`FakeParameterSet` rather than building a
real system of bodies.
"""

import numpy as np
import pytest
from omegaconf import OmegaConf
from tudatpy.dynamics import parameters_setup as param_setup

from orbitdet.estimation.apriori import get_apriori_covariance_matrix
from orbitdet.estimation.estimators import get_additional_parameter_identifiers

_STATE_ID = (
    param_setup.EstimatableParameterTypes.initial_body_state_type,
    ("Triton", ""),
)


class FakeParameterSet:
    """Fake ``EstimatableParameterSet`` exposing only what the builder needs."""

    def __init__(self, blocks):
        """``blocks`` is a list of ``(identifier, size)`` in parameter order."""
        self._blocks = []
        start = 0
        for identifier, size in blocks:
            self._blocks.append((identifier, start, size))
            start += size
        self.parameter_set_size = start

    def get_parameter_identifiers(self):
        return [identifier for identifier, _, _ in self._blocks]

    def indices_for_parameter_type(self, identifier):
        return [(start, size) for ident, start, size in self._blocks if ident == identifier]


def _cfg(parameters: dict, apriori: dict | None = None):
    estimation = {"parameters_to_estimate": parameters}
    if apriori is not None:
        estimation["apriori"] = apriori
    return OmegaConf.create({"estimation": estimation})


def _state_and_additional(*additional_keys: str, **apriori):
    """Build a config with the state plus ``additional_keys`` and a fake set."""
    parameters = {"initial_state": True}
    for key in additional_keys:
        parameters[key] = True
    cfg = _cfg(parameters, apriori=apriori or None)
    ids = get_additional_parameter_identifiers(cfg)
    blocks = [(_STATE_ID, 6)] + [(ids[key], 1) for key in additional_keys]
    return cfg, FakeParameterSet(blocks)


class TestNoApriori:
    def test_returns_none_when_nothing_configured(self):
        cfg, fake = _state_and_additional("neptune_GM")
        assert get_apriori_covariance_matrix(cfg, fake) is None


class TestGlobalApriori:
    def test_state_only(self):
        cfg, fake = _state_and_additional(position=100.0e3, velocity=1.0)
        m = get_apriori_covariance_matrix(cfg, fake)
        assert m.shape == (6, 6)
        diag = np.diag(m)
        np.testing.assert_allclose(diag[:3], 1.0 / (100.0e3) ** 2)
        np.testing.assert_allclose(diag[3:], 1.0)

    def test_state_plus_inverse_quality_factor(self):
        cfg, fake = _state_and_additional(
            "neptune_inverse_tidal_quality_factor",
            position=100.0e3,
            velocity=1.0,
            neptune_inverse_tidal_quality_factor=1.0,
        )
        m = get_apriori_covariance_matrix(cfg, fake)
        assert m.shape == (7, 7)
        assert np.diag(m)[-1] == pytest.approx(1.0)

    def test_unconstrained_parameter_gets_huge_default(self):
        cfg, fake = _state_and_additional("neptune_GM", position=100.0e3, velocity=1.0)
        m = get_apriori_covariance_matrix(cfg, fake)
        assert m.shape == (7, 7)
        # neptune_GM is unconstrained -> inverse variance ~ 1e-60
        assert np.diag(m)[-1] < 1e-50


class TestPerParameterApriori:
    def test_per_parameter_overrides_global(self):
        cfg = _cfg(
            {"initial_state": True, "neptune_GM": {"apriori": [2.0]}},
            apriori={"position": 100.0e3, "velocity": 1.0, "neptune_GM": 5.0},
        )
        ids = get_additional_parameter_identifiers(cfg)
        fake = FakeParameterSet([(_STATE_ID, 6), (ids["neptune_GM"], 1)])
        m = get_apriori_covariance_matrix(cfg, fake)
        assert np.diag(m)[6] == pytest.approx(1.0 / 2.0**2)

    def test_vector_apriori_sets_each_scalar(self):
        cfg = _cfg(
            {"initial_state": True, "neptune_j2_j4": {"apriori": [1.0e-5, 2.0e-5]}},
            apriori={"position": 100.0e3, "velocity": 1.0},
        )
        ids = get_additional_parameter_identifiers(cfg)
        fake = FakeParameterSet([(_STATE_ID, 6), (ids["neptune_j2_j4"], 2)])
        m = get_apriori_covariance_matrix(cfg, fake)
        diag = np.diag(m)
        assert diag[6] == pytest.approx(1.0 / (1.0e-5) ** 2)
        assert diag[7] == pytest.approx(1.0 / (2.0e-5) ** 2)

    def test_initial_state_per_parameter(self):
        cfg = _cfg({"initial_state": {"apriori": [10.0, 0.5]}})
        fake = FakeParameterSet([(_STATE_ID, 6)])
        m = get_apriori_covariance_matrix(cfg, fake)
        diag = np.diag(m)
        np.testing.assert_allclose(diag[:3], 1.0 / 10.0**2)
        np.testing.assert_allclose(diag[3:], 1.0 / 0.5**2)

    def test_two_state_blocks_interleave_position_and_velocity(self):
        # Two propagated bodies -> a single 12-scalar state block, laid out as
        # [pos0, vel0, pos1, vel1], so pos/vel alternate every 3 entries.
        cfg = _cfg({"initial_state": True}, apriori={"position": 100.0e3, "velocity": 2.0})
        fake = FakeParameterSet([(_STATE_ID, 12)])
        m = get_apriori_covariance_matrix(cfg, fake)
        diag = np.diag(m)
        expected = [1.0 / (100.0e3) ** 2] * 3 + [1.0 / 2.0**2] * 3
        np.testing.assert_allclose(diag, expected + expected)

    def test_wrong_length_apriori_raises(self):
        cfg = _cfg(
            {"initial_state": True, "neptune_j2_j4": {"apriori": [1.0, 2.0, 3.0]}},
            apriori={"position": 100.0e3, "velocity": 1.0},
        )
        ids = get_additional_parameter_identifiers(cfg)
        fake = FakeParameterSet([(_STATE_ID, 6), (ids["neptune_j2_j4"], 2)])
        with pytest.raises(ValueError, match="provides 3 sigma values but its parameter block"):
            get_apriori_covariance_matrix(cfg, fake)


class TestBlockSizesFromParameterSet:
    """The builder must trust the parameter set's block sizes, not the config."""

    def test_libration_block_size_respected(self):
        cfg = _cfg(
            {"initial_state": True, "iau_rotation_model_pole_librations": True},
            apriori={"position": 100.0e3, "velocity": 1.0},
        )
        ids = get_additional_parameter_identifiers(cfg)
        # Real libration block is 4 scalars (2 amplitudes x 2 frequencies).
        fake = FakeParameterSet([(_STATE_ID, 6), (ids["iau_rotation_model_pole_librations"], 4)])
        m = get_apriori_covariance_matrix(cfg, fake)
        assert m.shape == (10, 10)

    def test_matrix_size_matches_parameter_set(self):
        cfg = _cfg(
            {"initial_state": True, "neptune_inverse_tidal_quality_factor": True},
            apriori={"position": 100.0e3, "velocity": 1.0},
        )
        ids = get_additional_parameter_identifiers(cfg)
        fake = FakeParameterSet([(_STATE_ID, 6), (ids["neptune_inverse_tidal_quality_factor"], 1)])
        m = get_apriori_covariance_matrix(cfg, fake)
        assert m.shape == (fake.parameter_set_size, fake.parameter_set_size)

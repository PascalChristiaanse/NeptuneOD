"""Tests for annual aberration of light."""

import erfa
import numpy as np
import pytest

from orbitdet.preprocessing.aberration import (
    SPEED_OF_LIGHT,
    DirectionKind,
    aberration_angle_radians,
    aberration_beta_from_velocity,
    apply_aberration,
    correct_to_apparent,
    correct_to_astrometric,
)

ARCSEC = np.pi / (180.0 * 3600.0)

#: Earth's orbital speed, the dominant aberration source for this pipeline.  The
#: maximum apparent displacement is beta radians, ~20.5 arcsec.
EARTH_ORBITAL_SPEED = 29.7827e3


def random_sky(n: int = 25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``n`` spread-out (ra, dec) pairs in radians."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, n), rng.uniform(-np.pi / 2 * 0.95, np.pi / 2 * 0.95, n)


def random_velocities(n: int = 25, seed: int = 1) -> np.ndarray:
    """Return ``n`` observer velocities in m/s, spread over all directions."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions * rng.uniform(0.0, 2.0 * EARTH_ORBITAL_SPEED, n)[:, None]


def angular_difference(a, b) -> np.ndarray:
    """Absolute angular difference in rad, robust to 2*pi wrapping."""
    return np.abs((np.asarray(a) - np.asarray(b) + np.pi) % (2.0 * np.pi) - np.pi)


def rack(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Return the unit vectors for arrays of (ra, dec) in radians."""
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1)


class TestDirectionKind:
    def test_members(self):
        assert DirectionKind.APPARENT == "apparent"
        assert DirectionKind.ASTROMETRIC == "astrometric"

    @pytest.mark.parametrize("name", ["apparent", "astrometric"])
    def test_construction_from_string(self, name):
        # Values are lowercase, so the bare labels from a configuration resolve
        # directly without normalisation at the call site.
        assert DirectionKind(name) is DirectionKind[name.upper()]


class TestBetaFromVelocity:
    def test_zero_velocity(self):
        beta = aberration_beta_from_velocity(np.zeros((3, 3)))

        assert beta.shape == (3, 3)
        assert not beta.any()

    def test_magnitude_matches_speed_over_c(self):
        velocities = random_velocities(10)
        beta = aberration_beta_from_velocity(velocities)
        speeds = np.linalg.norm(velocities, axis=1)

        assert np.allclose(np.linalg.norm(beta, axis=1), speeds / SPEED_OF_LIGHT)

    def test_accepts_a_single_velocity(self):
        beta = aberration_beta_from_velocity(np.array([1.0e3, 2.0e3, 3.0e3]))

        assert beta.shape == (3,)

    def test_rejects_a_non_three_vector(self):
        with pytest.raises(ValueError, match="last axis"):
            aberration_beta_from_velocity(np.zeros((4, 2)))

    def test_earth_orbital_beta_is_about_twenty_arcsec(self):
        beta = aberration_beta_from_velocity(np.array([EARTH_ORBITAL_SPEED, 0.0, 0.0]))
        displacement = np.rad2deg(float(np.linalg.norm(beta))) * 3600.0

        assert 20.0 < displacement < 21.0


class TestApplyAberration:
    def test_zero_beta_is_the_identity(self):
        ra, dec = random_sky()
        directions = np.stack(
            [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1
        )
        out = apply_aberration(directions, np.zeros(3))

        assert np.allclose(out, directions)

    def test_output_is_normalised(self):
        ra, dec = random_sky()
        directions = np.stack(
            [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1
        )
        out = apply_aberration(directions, aberration_beta_from_velocity(random_velocities(25)))

        assert np.allclose(np.linalg.norm(out, axis=1), 1.0)

    def test_matches_erfa(self):
        """The IAU SOFA formula must agree with ERFA's implementation.

        ``erfa.ab`` takes the Sun-observer distance ``s`` in au, because it also
        applies the Solar gravitational deflection of about 0.4 microarcsec.  To
        isolate pure aberration the radial term is driven to zero with ``s ->
        inf``, and the true reciprocal Lorentz factor is passed as ``bm1``.
        """
        ra, dec = random_sky(25, seed=5)
        directions = rack(ra, dec)
        velocities = random_velocities(25, seed=6)
        beta = aberration_beta_from_velocity(velocities)
        bm1 = np.sqrt(1.0 - np.einsum("ij,ij->i", beta, beta))

        ours = apply_aberration(directions, beta)
        theirs = np.stack(
            [
                np.asarray(erfa.ab(direction, row_beta, 1.0e12, row_bm1))
                for direction, row_beta, row_bm1 in zip(directions, beta, bm1)
            ]
        )

        assert np.all(np.isfinite(theirs))
        assert np.max(np.abs(ours - theirs)) < 1e-15

    def test_beta_is_broadcast_over_directions(self):
        """A single velocity must apply to every row."""
        ra, dec = random_sky(25, seed=7)
        directions = np.stack(
            [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1
        )
        beta = aberration_beta_from_velocity(np.array([EARTH_ORBITAL_SPEED, 0.0, 0.0]))

        broadcast = apply_aberration(directions, beta)
        explicit = apply_aberration(directions, np.tile(beta, (directions.shape[0], 1)))

        assert np.allclose(broadcast, explicit)

    def test_input_vectors_are_normalised_first(self):
        ra, dec = random_sky(10, seed=8)
        directions = np.stack(
            [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1
        )
        beta = aberration_beta_from_velocity(random_velocities(10, seed=9))

        assert np.allclose(
            apply_aberration(directions * 7.5, beta), apply_aberration(directions, beta)
        )

    def test_zero_vectors_stay_nan(self):
        directions = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        out = apply_aberration(directions, np.zeros(3))

        assert np.isfinite(out[0]).all()
        assert np.isnan(out[1]).all()

    @pytest.mark.parametrize("speed", [SPEED_OF_LIGHT, 1.5 * SPEED_OF_LIGHT])
    def test_rejects_superluminal_beta(self, speed):
        with pytest.raises(ValueError, match="below 1"):
            apply_aberration(np.array([1.0, 0.0, 0.0]), np.array([speed, 0.0, 0.0]))

    def test_direction_along_beta_is_unchanged(self):
        """A source in the direction of motion keeps its direction.

        This is the degenerate case of the formula: with ``u`` parallel to
        ``beta`` the numerator and denominator are both ``1 + beta``, so the
        result is ``u`` to within rounding rather than exactly.
        """
        beta = np.array([1.0e-4, 0.0, 0.0])

        along = apply_aberration(np.array([1.0, 0.0, 0.0]), beta)
        anti_along = apply_aberration(np.array([-1.0, 0.0, 0.0]), beta)

        assert np.allclose(along, [1.0, 0.0, 0.0], atol=1e-15)
        assert np.allclose(anti_along, [-1.0, 0.0, 0.0], atol=1e-15)

    def test_displacement_is_along_the_motion_for_a_perpendicular_source(self):
        """A source perpendicular to the motion shifts predominantly along beta.

        It also receives the largest displacement of any source, ``|beta|``:
        aberration is zero along the direction of motion and maximal across it.
        The small cross component is the ``-beta**2 / 2`` contraction that
        renormalising the displaced vector to unit length introduces.
        """
        beta_scalar = 1.0e-4
        beta = np.array([beta_scalar, 0.0, 0.0])
        direction = np.array([0.0, 1.0, 0.0])
        out = apply_aberration(direction, beta)
        displacement = np.asarray(out).ravel() - direction

        assert np.isclose(displacement[0], beta_scalar, rtol=1e-6)
        assert np.isclose(displacement[1], -0.5 * beta_scalar**2, rtol=1e-4)
        assert np.isclose(np.linalg.norm(displacement), beta_scalar, rtol=1e-6)

    def test_displacement_decreases_towards_the_direction_of_motion(self):
        """Displacement falls off as the source aligns with the velocity."""
        velocity = np.array([EARTH_ORBITAL_SPEED, 0.0, 0.0])
        aligned = aberration_angle_radians(0.0, 0.0, velocity)
        perpendicular = aberration_angle_radians(0.0, np.pi / 2, velocity)
        anti_aligned = aberration_angle_radians(np.pi, 0.0, velocity)

        assert aligned < perpendicular
        assert anti_aligned < perpendicular
        assert np.isclose(aligned, 0.0, atol=1e-12)
        assert np.isclose(anti_aligned, 0.0, atol=1e-12)


class TestRoundTrip:
    def test_apparent_then_astrometric_is_the_identity(self):
        ra, dec = random_sky(25, seed=10)
        velocities = random_velocities(25, seed=11)

        apparent_ra, apparent_dec = correct_to_apparent(ra, dec, velocities)
        back_ra, back_dec = correct_to_astrometric(apparent_ra, apparent_dec, velocities)

        assert np.max(angular_difference(back_ra, ra)) < 1e-12
        assert np.max(np.abs(back_dec - dec)) < 1e-12

    def test_astrometric_then_apparent_is_the_identity(self):
        ra, dec = random_sky(25, seed=12)
        velocities = random_velocities(25, seed=13)

        astrometric_ra, astrometric_dec = correct_to_astrometric(ra, dec, velocities)
        back_ra, back_dec = correct_to_apparent(astrometric_ra, astrometric_dec, velocities)

        assert np.max(angular_difference(back_ra, ra)) < 1e-12
        assert np.max(np.abs(back_dec - dec)) < 1e-12

    def test_zero_velocity_is_the_identity(self):
        ra, dec = random_sky(25, seed=14)
        out_ra, out_dec = correct_to_apparent(ra, dec, np.zeros((25, 3)))

        assert np.allclose(out_ra, ra)
        assert np.allclose(out_dec, dec)


class TestCorrectToApparent:
    def test_earth_aberration_is_about_twenty_arcsec(self):
        """The classic annual aberration constant is ~20.5 arcsec."""
        ra, dec = random_sky(200, seed=15)
        velocity = np.tile(np.array([EARTH_ORBITAL_SPEED, 0.0, 0.0]), (200, 1))

        apparent_ra, apparent_dec = correct_to_apparent(ra, dec, velocity)
        separation = np.arccos(
            np.clip(
                np.cos(dec) * np.cos(apparent_dec) * np.cos(ra - apparent_ra)
                + np.sin(dec) * np.sin(apparent_dec),
                -1.0,
                1.0,
            )
        )
        max_separation = float(np.max(separation)) * 3600.0 * 180.0 / np.pi

        assert 20.0 < max_separation < 21.0

    def test_nan_rows_are_preserved(self):
        ra = np.array([0.1, np.nan, 0.3])
        dec = np.array([0.2, 0.4, np.nan])
        velocity = np.tile(np.array([EARTH_ORBITAL_SPEED, 0.0, 0.0]), (3, 1))

        out_ra, out_dec = correct_to_apparent(ra, dec, velocity)

        assert np.isfinite(out_ra[0]) and np.isfinite(out_dec[0])
        assert np.isnan(out_ra[1]) and np.isnan(out_dec[1])
        assert np.isnan(out_ra[2]) and np.isnan(out_dec[2])

    def test_wrap_into_zero_two_pi(self):
        ra, dec = random_sky(100, seed=16)
        velocity = np.tile(np.array([EARTH_ORBITAL_SPEED, 0.0, 0.0]), (100, 1))

        out_ra, _ = correct_to_apparent(ra, dec, velocity, wrap=True)

        assert (out_ra >= 0.0).all() and (out_ra <= 2.0 * np.pi).all()

    def test_default_wrap_is_the_principal_interval(self):
        ra, dec = random_sky(100, seed=17)
        velocity = np.tile(np.array([EARTH_ORBITAL_SPEED, 0.0, 0.0]), (100, 1))

        out_ra, _ = correct_to_apparent(ra, dec, velocity)

        assert (out_ra >= -np.pi).all() and (out_ra <= np.pi).all()

    def test_accepts_lists(self):
        out_ra, out_dec = correct_to_apparent(
            [0.1, 0.2], [0.3, 0.4], np.array([EARTH_ORBITAL_SPEED, 0.0, 0.0])
        )

        assert out_ra.shape == (2,)
        assert out_dec.shape == (2,)


class TestAberrationAngle:
    def test_zero_velocity_gives_zero_angle(self):
        ra, dec = random_sky(10, seed=18)
        angle = aberration_angle_radians(ra, dec, np.zeros((10, 3)))

        # The displacement is computed from an arccos of a dot product of unit
        # vectors, so it is zero to within the arccos' own rounding.
        assert np.allclose(angle, 0.0, atol=1e-7)

    def test_is_never_negative(self):
        ra, dec = random_sky(50, seed=19)
        angle = aberration_angle_radians(ra, dec, random_velocities(50, seed=20))

        assert (angle >= 0.0).all()

    def test_is_symmetric_in_the_sign_of_velocity(self):
        """Aberration displaces along the observer's motion, so reversing the
        velocity mirrors the displacement and leaves its magnitude unchanged.

        The two agree to first order in ``beta``; the residual is the
        ``O(beta**2)`` curvature of the sphere the direction slides over, which
        for Earth's orbital speed is at most a few times ``1e-8`` rad.
        """
        ra, dec = random_sky(50, seed=21)
        velocities = random_velocities(50, seed=22)
        max_beta_squared = np.max(np.einsum("ij,ij->i", velocities, velocities)) / SPEED_OF_LIGHT**2

        forward = aberration_angle_radians(ra, dec, velocities)
        backward = aberration_angle_radians(ra, dec, -velocities)

        assert np.allclose(forward, backward, rtol=1e-3)
        assert np.max(np.abs(forward - backward)) < max_beta_squared

    def test_matches_the_vector_form(self):
        """The scalar RA/Dec helper and the vector helper must agree."""
        ra, dec = random_sky(25, seed=23)
        velocities = random_velocities(25, seed=24)
        directions = np.stack(
            [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1
        )
        beta = aberration_beta_from_velocity(velocities)
        aberrated = apply_aberration(directions, beta)
        expected = np.arccos(np.clip(np.sum(directions * aberrated, axis=1), -1.0, 1.0))

        actual = aberration_angle_radians(ra, dec, velocities)

        assert np.allclose(actual, expected, atol=1e-12)

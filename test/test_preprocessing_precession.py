"""Tests for equinox parsing and precession."""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import FK4, FK5, SkyCoord
from astropy.time import Time

from orbitdet.preprocessing.precession import (
    JD_J2000,
    SPICE_FRAMES,
    EquinoxSystem,
    Fk4Convention,
    equinox_system,
    is_spice_frame,
    mean_of_date_equinox,
    parse_equinox,
    precess_radec,
    radec_to_unit_vectors,
    rotation_matrix,
    unit_vectors_to_radec,
)
from orbitdet.preprocessing.time_scales import SECONDS_PER_DAY

J2000_EPOCH = 0.0
ARCSEC = np.pi / (180.0 * 3600.0)


def angular_difference(a, b) -> np.ndarray:
    """Absolute angular difference in rad, robust to 2*pi wrapping.

    The module returns right ascension in the principal interval ``[-pi, pi]``
    while Astropy returns ``[0, 2*pi)``, so a raw subtraction would report a
    2*pi error for angles near the boundary.
    """
    return np.abs((np.asarray(a) - np.asarray(b) + np.pi) % (2.0 * np.pi) - np.pi)


def random_sky(n: int = 25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``n`` spread-out (ra, dec) pairs in radians."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, n), rng.uniform(-np.pi / 2 * 0.95, np.pi / 2 * 0.95, n)


class TestEquinoxParsing:
    @pytest.mark.parametrize("label", ["J2000", "J2000.0", "2000.0", "2000"])
    def test_j2000_labels(self, label):
        equinox = parse_equinox(label)

        assert equinox.system is EquinoxSystem.FK5
        assert equinox.astropy_time is not None
        assert equinox.astropy_time.iso == "2000-01-01 12:00:00.000"

    @pytest.mark.parametrize("label", ["date", "Date", "of date", "mean of date", "MOD"])
    def test_mean_of_date_labels(self, label):
        equinox = parse_equinox(label)

        assert equinox.system is EquinoxSystem.OF_DATE
        assert equinox.is_of_date

    @pytest.mark.parametrize("label", ["true of date", "TOD", "true equinox of date"])
    def test_true_of_date_labels(self, label):
        equinox = parse_equinox(label)

        assert equinox.system is EquinoxSystem.TRUE_OF_DATE
        assert equinox.is_true_of_date

    def test_b1950_is_besselian(self):
        equinox = parse_equinox("B1950")

        assert equinox.system is EquinoxSystem.FK4
        assert equinox.is_besselian

    @pytest.mark.parametrize("label", ["1905.0", "1899.0", "1919.0"])
    def test_bare_years_are_besselian(self, label):
        # NSDB writes Besselian epochs without the B.  Treating them as Julian
        # would misplace them by ~20 arcsec at 1900.
        equinox = parse_equinox(label)

        assert equinox.system is EquinoxSystem.FK4
        assert equinox.is_besselian

    def test_bare_year_and_b_prefixed_year_agree(self):
        assert parse_equinox("1905.0").astropy_time.iso == parse_equinox("B1905").astropy_time.iso

    def test_j2015_is_julian(self):
        equinox = parse_equinox("J2015.0")

        assert equinox.system is EquinoxSystem.FK5
        assert not equinox.is_besselian

    def test_unknown_label_falls_back_to_j2000(self):
        assert parse_equinox("banana").system is EquinoxSystem.FK5

    @pytest.mark.parametrize("label", ["", "no information", "no inf", "unknown"])
    def test_empty_labels_fall_back_to_j2000(self, label):
        equinox = parse_equinox(label)

        assert equinox.system is EquinoxSystem.FK5
        assert equinox.astropy_time.iso == "2000-01-01 12:00:00.000"

    def test_unknown_label_warns(self, caplog):
        with caplog.at_level("WARNING"):
            parse_equinox("banana")

        assert any("Unrecognised epoch of equinox" in r.message for r in caplog.records)

    def test_bare_year_is_logged(self, caplog):
        with caplog.at_level("INFO"):
            parse_equinox("1905.0")

        assert any("Besselian" in r.message for r in caplog.records)

    def test_equinox_system_helper(self):
        assert equinox_system("date") is EquinoxSystem.OF_DATE
        assert equinox_system("B1950") is EquinoxSystem.FK4
        assert equinox_system("J2000") is EquinoxSystem.FK5

    def test_is_of_date_flags(self):
        assert parse_equinox("date").is_of_date
        assert parse_equinox("true of date").is_of_date
        assert not parse_equinox("J2000").is_of_date
        assert not parse_equinox("B1950").is_of_date

    def test_mean_of_date_equinox_returns_the_epoch(self):
        equinox = mean_of_date_equinox(0.0)

        assert equinox.jd == pytest.approx(JD_J2000, abs=1e-9)


class TestUnitVectorHelpers:
    def test_radec_to_unit_vectors_is_normalised(self):
        ra, dec = random_sky()

        vectors = radec_to_unit_vectors(ra, dec)

        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    def test_round_trip(self):
        ra, dec = random_sky()

        out_ra, out_dec = unit_vectors_to_radec(radec_to_unit_vectors(ra, dec))

        assert np.allclose(out_ra, ra, atol=1e-12)
        assert np.allclose(out_dec, dec, atol=1e-12)

    def test_equinox_direction_is_positive_x(self):
        vectors = radec_to_unit_vectors(np.array([0.0]), np.array([0.0]))

        assert np.allclose(vectors[0], [1.0, 0.0, 0.0])

    def test_pole_is_positive_z(self):
        vectors = radec_to_unit_vectors(np.array([0.0]), np.array([np.pi / 2]))

        assert np.allclose(vectors[0], [0.0, 0.0, 1.0], atol=1e-12)

    def test_radec_wraps_into_principal_interval_by_default(self):
        out_ra, _ = unit_vectors_to_radec(radec_to_unit_vectors(np.array([5.0]), np.array([0.0])))

        assert -np.pi <= out_ra[0] <= np.pi

    def test_radec_can_wrap_into_zero_two_pi(self):
        out_ra, _ = unit_vectors_to_radec(
            radec_to_unit_vectors(np.array([-1.0]), np.array([0.0])), wrap=True
        )

        assert 0.0 <= out_ra[0] < 2 * np.pi

    def test_zero_vector_yields_nan(self):
        out_ra, out_dec = unit_vectors_to_radec(np.array([[0.0, 0.0, 0.0]]))

        assert np.isnan(out_ra[0]) and np.isnan(out_dec[0])


class TestPrecessionIdentity:
    def test_same_equinox_is_the_identity(self):
        ra, dec = random_sky()

        out_ra, out_dec = precess_radec(ra, dec, from_equinox="J2000", to_equinox="J2000")

        assert np.allclose(out_ra, ra, atol=1e-15)
        assert np.allclose(out_dec, dec, atol=1e-15)

    def test_j2000_labels_are_interchangeable(self):
        ra, dec = random_sky()

        out_ra, out_dec = precess_radec(ra, dec, from_equinox="J2000", to_equinox="J2000.0")

        assert np.allclose(out_ra, ra, atol=1e-15)
        assert np.allclose(out_dec, dec, atol=1e-15)

    def test_of_date_at_j2000_is_the_identity(self):
        # The mean equinox of 2000-01-01 12:00 IS J2000.
        ra, dec = random_sky()

        out_ra, out_dec = precess_radec(
            ra, dec, from_equinox="date", to_equinox="J2000", epoch_tdb=J2000_EPOCH
        )

        assert np.all(angular_difference(out_ra, ra) < 1e-9)
        assert np.allclose(out_dec, dec, atol=1e-9)


class TestPrecessionRoundTrip:
    @pytest.mark.parametrize(
        ("source", "target"),
        [
            ("J2000", "B1950"),
            ("J2000", "J2015.0"),
            ("B1950", "1905.0"),
            ("1899.0", "B1950"),
        ],
    )
    def test_fixed_equinox_round_trip(self, source, target):
        ra, dec = random_sky()

        there_ra, there_dec = precess_radec(ra, dec, from_equinox=source, to_equinox=target)
        back_ra, back_dec = precess_radec(
            there_ra, there_dec, from_equinox=target, to_equinox=source
        )

        # The FK4/FK5 transforms are not exactly invertible (the E-terms and the
        # frame bias are applied as one-way corrections), so the round trip is
        # accurate to ~0.001 arcsec rather than to machine precision.
        assert np.all(angular_difference(back_ra, ra) < 1e-8)
        assert np.allclose(back_dec, dec, atol=1e-8)

    def test_of_date_round_trip(self):
        ra, dec = random_sky()
        epochs = np.full(ra.shape, 1e8)

        there_ra, there_dec = precess_radec(
            ra, dec, from_equinox="date", to_equinox="J2000", epoch_tdb=epochs
        )
        back_ra, back_dec = precess_radec(
            there_ra, there_dec, from_equinox="J2000", to_equinox="date", epoch_tdb=epochs
        )

        assert np.all(angular_difference(back_ra, ra) < 1e-11)
        assert np.allclose(back_dec, dec, atol=1e-11)


class TestPrecessionMagnitudes:
    def test_j2000_to_b1950_is_about_50_arcsec_per_year_of_precession(self):
        # The general precession is ~50.29 arcsec/yr, so 50 years is ~0.7 deg.
        ra, dec = np.array([1.0]), np.array([0.5])

        out_ra, out_dec = precess_radec(ra, dec, from_equinox="J2000", to_equinox="B1950")

        displacement = np.hypot(out_ra[0] - ra[0], out_dec[0] - dec[0]) / ARCSEC
        assert 2000 < displacement < 3200

    def test_j2000_to_j2015_is_about_fifteen_years_of_precession(self):
        ra, dec = np.array([1.0]), np.array([0.5])

        out_ra, out_dec = precess_radec(ra, dec, from_equinox="J2000", to_equinox="J2015.0")

        displacement = np.hypot(out_ra[0] - ra[0], out_dec[0] - dec[0]) / ARCSEC
        assert 600 < displacement < 950

    def test_precession_increases_with_epoch_difference(self):
        ra, dec = np.array([1.0]), np.array([0.5])

        near, _ = precess_radec(ra, dec, from_equinox="J2000", to_equinox="J2010.0")
        far, _ = precess_radec(ra, dec, from_equinox="J2000", to_equinox="J2030.0")

        assert abs(far[0] - ra[0]) > abs(near[0] - ra[0])

    def test_b1950_to_j2000_is_the_inverse_displacement(self):
        ra, dec = np.array([1.0]), np.array([0.5])

        fwd_ra, _ = precess_radec(ra, dec, from_equinox="J2000", to_equinox="B1950")
        bwd_ra, _ = precess_radec(ra, dec, from_equinox="B1950", to_equinox="J2000")

        assert (fwd_ra[0] - ra[0]) * (bwd_ra[0] - ra[0]) < 0


class TestPrecessionAgainstAstropy:
    """Agreement with a direct Astropy transform, using the Astropy convention."""

    @pytest.mark.parametrize(
        ("source", "target"),
        [
            ("J2000", "J2015.0"),
            ("J2015.0", "J2000"),
            ("B1950", "J2000"),
            ("1905.0", "J2000"),
            ("1899.0", "B1950"),
        ],
    )
    def test_matches_direct_astropy(self, source, target):
        ra, dec = random_sky()

        out_ra, out_dec = precess_radec(
            ra,
            dec,
            from_equinox=source,
            to_equinox=target,
            fk4_convention=Fk4Convention.ASTROPY,
        )

        source_equinox = parse_equinox(source)
        target_equinox = parse_equinox(target)
        input_frame = (
            FK4(equinox=source_equinox.astropy_time)
            if source_equinox.is_besselian
            else FK5(equinox=source_equinox.astropy_time)
        )
        output_frame = (
            FK4(equinox=target_equinox.astropy_time)
            if target_equinox.is_besselian
            else FK5(equinox=target_equinox.astropy_time)
        )

        reference = SkyCoord(ra * u.rad, dec * u.rad, frame=input_frame).transform_to(output_frame)

        assert np.all(angular_difference(out_ra, reference.ra.to(u.rad).value) < 1e-14)
        assert np.allclose(out_dec, reference.dec.to(u.rad).value, atol=1e-14)

    def test_of_date_matches_direct_astropy(self):
        ra, dec = random_sky()
        epochs = np.linspace(0.0, 5e8, ra.size)

        out_ra, out_dec = precess_radec(
            ra, dec, from_equinox="J2000", to_equinox="date", epoch_tdb=epochs
        )

        obstimes = Time(epochs / SECONDS_PER_DAY + JD_J2000, format="jd", scale="tdb")
        reference = SkyCoord(ra * u.rad, dec * u.rad, frame=FK5(equinox="J2000")).transform_to(
            FK5(equinox=obstimes)
        )

        assert np.all(angular_difference(out_ra, reference.ra.to(u.rad).value) < 1e-13)
        assert np.allclose(out_dec, reference.dec.to(u.rad).value, atol=1e-13)


class TestFk4Convention:
    def test_all_conventions_are_available(self):
        assert {m.value for m in Fk4Convention} == {"spice", "astropy", "astropy_no_eterms"}

    def test_spice_is_the_default(self):
        import inspect

        default = inspect.signature(precess_radec).parameters["fk4_convention"].default

        assert Fk4Convention(default) is Fk4Convention.SPICE

    def test_spice_and_astropy_differ_by_several_arcsec(self):
        # A real convention difference, larger than the astrometric corrections
        # applied elsewhere, so it has to be an explicit choice.  The RA
        # difference grows towards the pole, because the E-term RA component
        # carries a 1/cos(dec) factor: over this random sky sample it reaches a
        # few arcsec, and it would grow without bound at the pole itself.
        ra, dec = random_sky()

        spice_ra, spice_dec = precess_radec(
            ra, dec, from_equinox="J2000", to_equinox="B1950", fk4_convention="spice"
        )
        astropy_ra, astropy_dec = precess_radec(
            ra, dec, from_equinox="J2000", to_equinox="B1950", fk4_convention="astropy"
        )

        ra_diff = angular_difference(spice_ra, astropy_ra).max() / ARCSEC
        dec_diff = np.abs(spice_dec - astropy_dec).max() / ARCSEC
        assert 0.5 < ra_diff < 10.0
        assert 0.0 < dec_diff < 0.5

    def test_astropy_no_eterms_differs_from_astropy(self):
        ra, dec = random_sky()

        with_terms_ra, _ = precess_radec(
            ra, dec, from_equinox="J2000", to_equinox="B1950", fk4_convention="astropy"
        )
        without_ra, _ = precess_radec(
            ra, dec, from_equinox="J2000", to_equinox="B1950", fk4_convention="astropy_no_eterms"
        )

        assert not np.all(angular_difference(with_terms_ra, without_ra) < 1e-9)

    def test_convention_does_not_affect_non_besselian_equinoxes(self):
        ra, dec = random_sky()

        spice_ra, _ = precess_radec(
            ra, dec, from_equinox="J2000", to_equinox="J2015.0", fk4_convention="spice"
        )
        astropy_ra, _ = precess_radec(
            ra, dec, from_equinox="J2000", to_equinox="J2015.0", fk4_convention="astropy"
        )

        assert np.allclose(spice_ra, astropy_ra, atol=1e-15)

    def test_spice_convention_matches_spice_matrix(self):
        ra, dec = random_sky()

        out_ra, out_dec = precess_radec(
            ra, dec, from_equinox="J2000", to_equinox="B1950", fk4_convention="spice"
        )

        matrix = rotation_matrix("J2000", "B1950", 0.0)
        rotated = radec_to_unit_vectors(ra, dec) @ matrix.T
        expected_ra, expected_dec = unit_vectors_to_radec(rotated)

        assert np.all(angular_difference(out_ra, expected_ra) < 1e-14)
        assert np.allclose(out_dec, expected_dec, atol=1e-14)


class TestOfDateEpochHandling:
    def test_of_date_without_epoch_raises(self):
        ra, dec = random_sky()

        with pytest.raises(ValueError, match="observation epoch is required"):
            precess_radec(ra, dec, from_equinox="date", to_equinox="J2000")

    def test_of_date_to_of_date_needs_no_conversion(self):
        ra, dec = random_sky()

        out_ra, out_dec = precess_radec(
            ra, dec, from_equinox="date", to_equinox="date", epoch_tdb=1e8
        )

        assert np.all(angular_difference(out_ra, ra) < 1e-15)
        assert np.allclose(out_dec, dec, atol=1e-15)

    def test_scalar_epoch_is_broadcast(self):
        ra, dec = random_sky()

        out_ra, _ = precess_radec(ra, dec, from_equinox="date", to_equinox="J2000", epoch_tdb=1e8)

        assert out_ra.shape == ra.shape

    def test_mismatched_epoch_length_raises(self):
        ra, dec = random_sky(10)

        with pytest.raises(ValueError, match="must match"):
            precess_radec(ra, dec, from_equinox="date", to_equinox="J2000", epoch_tdb=np.zeros(5))

    def test_true_of_date_differs_from_mean_of_date(self):
        # Nutation is up to ~17 arcsec in the equator, and grows towards the pole
        # because the RA component is divided by cos(dec).
        ra, dec = random_sky()

        mean_ra, mean_dec = precess_radec(
            ra, dec, from_equinox="date", to_equinox="J2000", epoch_tdb=1e8
        )
        true_ra, true_dec = precess_radec(
            ra, dec, from_equinox="true of date", to_equinox="J2000", epoch_tdb=1e8
        )

        equatorial = np.abs(dec) < 0.5
        displacement = (
            np.hypot(angular_difference(true_ra, mean_ra), true_dec - mean_dec)[equatorial].max()
            / ARCSEC
        )
        assert 0.01 < displacement < 60.0


class TestInputHandling:
    def test_accepts_lists(self):
        out_ra, out_dec = precess_radec([1.0], [0.5], from_equinox="J2000", to_equinox="B1950")

        assert out_ra.shape == (1,)

    def test_mismatched_ra_dec_shapes_raise(self):
        with pytest.raises(ValueError, match="matching shapes"):
            precess_radec(np.zeros(3), np.zeros(4), from_equinox="J2000", to_equinox="B1950")

    def test_nan_rows_stay_nan_without_aborting(self):
        ra = np.array([1.0, np.nan, 2.0])
        dec = np.array([0.5, 0.5, -0.3])

        out_ra, out_dec = precess_radec(ra, dec, from_equinox="J2000", to_equinox="B1950")

        assert np.isnan(out_ra[1]) and np.isnan(out_dec[1])
        assert np.isfinite(out_ra[0]) and np.isfinite(out_ra[2])

    def test_wrap_option(self):
        ra, dec = np.array([-3.0]), np.array([0.0])

        unwrapped, _ = precess_radec(ra, dec, from_equinox="J2000", to_equinox="J2015.0")
        wrapped, _ = precess_radec(ra, dec, from_equinox="J2000", to_equinox="J2015.0", wrap=True)

        assert -np.pi <= unwrapped[0] <= np.pi
        assert 0.0 <= wrapped[0] < 2 * np.pi

    def test_accepts_preparsed_equinox_objects(self):
        ra, dec = np.array([1.0]), np.array([0.5])

        out_ra, _ = precess_radec(
            ra, dec, from_equinox=parse_equinox("J2000"), to_equinox=parse_equinox("B1950")
        )

        assert np.isfinite(out_ra[0])

    def test_empty_input(self):
        out_ra, out_dec = precess_radec([], [], from_equinox="J2000", to_equinox="B1950")

        assert out_ra.shape == (0,)


class TestSpiceFrames:
    def test_known_spice_frames(self):
        for frame in ("J2000", "B1950", "ECLIPJ2000"):
            assert is_spice_frame(frame), frame

    @pytest.mark.parametrize("frame", ["ICRF", "DATE", "J2015.0", "astrometric", "Gaia"])
    def test_non_spice_frames(self, frame):
        # SPICE cannot express these; they must route through Astropy.
        assert not is_spice_frame(frame), frame

    def test_spice_frame_set(self):
        assert SPICE_FRAMES == frozenset({"J2000", "B1950", "ECLIPJ2000", "ECLIPB1950"})

    def test_rotation_matrix_rejects_non_spice_frame(self):
        with pytest.raises(ValueError, match="not a SPICE frame"):
            rotation_matrix("DATE", "J2000", 0.0)

    def test_rotation_matrix_is_orthonormal(self):
        matrix = rotation_matrix("J2000", "B1950", 0.0)

        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-14)

    def test_rotation_matrix_identity_for_same_frame(self):
        matrix = rotation_matrix("J2000", "J2000", 0.0)

        assert np.allclose(matrix, np.eye(3), atol=1e-15)

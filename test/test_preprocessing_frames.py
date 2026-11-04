"""Tests for frame-name routing and dataframe-level coordinate conversion."""

import numpy as np
import pandas as pd
import pytest
from tudatpy.interface import spice

from orbitdet.preprocessing.frames import (
    ICRS_FRAME_NAMES,
    DirectionKind,
    FrameKind,
    canonicalize_frame,
    classify_frame,
    convert_cartesian_frame,
    convert_radec_frame,
    radec_to_direction_vectors,
)

spice.load_standard_kernels()

ARCSEC = np.pi / (180.0 * 3600.0)
J2000_EPOCH = 0.0

#: Epochs spanning a century, in TDB seconds since J2000.
EPOCHS = np.linspace(-1.6e9, 1.6e9, 8)


def random_sky(n: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``n`` spread-out (ra, dec) pairs in degrees."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 360.0, n), rng.uniform(-89.0, 89.0, n)


def sky_frame(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Return a dataframe of RA/Dec in degrees with epochs."""
    ra, dec = random_sky(n, seed)
    return pd.DataFrame({"ra": ra, "dec": dec, "epoch": np.resize(EPOCHS, n)})


def angular_separation_deg(ra1, dec1, ra2, dec2) -> np.ndarray:
    """Great-circle separation in degrees, robust to RA wrapping.

    Uses the haversine form rather than ``arccos`` of a dot product.  The two
    agree analytically, but ``arccos`` loses catastrophic precision for small
    separations - its derivative diverges as the cosine approaches 1 - and
    bottoms out around 3e-3 arcsec.  Since the round trips below are correct to
    better than 1e-9 arcsec, the ``arccos`` form would report the method's error
    rather than the conversion's.
    """
    ra1, dec1, ra2, dec2 = (np.deg2rad(np.asarray(x)) for x in (ra1, dec1, ra2, dec2))
    haversine = (
        np.sin((dec2 - dec1) / 2.0) ** 2
        + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2.0) ** 2
    )
    return np.rad2deg(2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0))))


class TestClassifyFrame:
    @pytest.mark.parametrize(
        "frame",
        ["J2000", "B1950", "ECLIPJ2000", "ECLIPB1950", "j2000", "EME2000", "J2000.0"],
    )
    def test_spice_frames(self, frame):
        assert classify_frame(frame) is FrameKind.SPICE

    @pytest.mark.parametrize("frame", ["ICRS", "ICRF", "icrs", " ICRS "])
    def test_icrs_names_are_spice_frames(self, frame):
        """ICRS/ICRF are aliases for J2000, not a separate frame.

        The SPICE kernels this pipeline loads are ICRF-referenced and labelled
        J2000, so no rotation is required between them.
        """
        assert classify_frame(frame) is FrameKind.SPICE

    @pytest.mark.parametrize("frame", ["date", "Date", "of date", "mean of date", "MOD"])
    def test_of_date_frames(self, frame):
        assert classify_frame(frame) is FrameKind.OF_DATE

    @pytest.mark.parametrize("frame", ["true of date", "TOD", "true equinox of date"])
    def test_true_of_date_frames(self, frame):
        assert classify_frame(frame) is FrameKind.TRUE_OF_DATE

    @pytest.mark.parametrize("frame", ["astrometric", "apparent"])
    def test_kind_frames(self, frame):
        assert classify_frame(frame) is FrameKind.KIND

    @pytest.mark.parametrize("frame", ["J2015.0", "1905.0", "B1905", "J1900.0"])
    def test_equinox_frames(self, frame):
        assert classify_frame(frame) is FrameKind.EQUINOX

    @pytest.mark.parametrize("frame", ["nonsense", "", "xyzzy"])
    def test_unknown_frames(self, frame):
        assert classify_frame(frame) is FrameKind.UNKNOWN

    def test_apparent_is_a_kind_not_an_equinox(self):
        """'apparent' must resolve to the aberration vocabulary.

        It is a real NSDB ``reference_frame`` value (48 occurrences), and
        treating it as a true-of-date equinox would silently apply the wrong
        correction.
        """
        assert classify_frame("apparent") is FrameKind.KIND
        assert DirectionKind("apparent") is DirectionKind.APPARENT


class TestCanonicalizeFrame:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("J2000.0", "J2000"),
            ("j2000", "J2000"),
            ("EME2000", "J2000"),
            ("J2000", "J2000"),
            ("B1950.0", "B1950"),
            ("FK4", "B1950"),
            ("B1950", "B1950"),
        ],
    )
    def test_aliases(self, given, expected):
        assert canonicalize_frame(given) == expected

    @pytest.mark.parametrize("frame", ["ICRS", "ICRF"])
    def test_icrs_names_resolve_to_j2000(self, frame):
        assert canonicalize_frame(frame) == "J2000"
        assert canonicalize_frame(frame.lower()) == "J2000"

    @pytest.mark.parametrize("frame", ["ECLIPJ2000", "J2015.0", "date", "true of date"])
    def test_unmappable_names_pass_through_unchanged(self, frame):
        """A frame SPICE cannot express must not be silently approximated.

        Rewriting e.g. ``J2015.0`` to ``J2000`` would produce a plausible but
        wrong answer, so the name survives canonicalisation and the routing
        layer is forced to handle it.
        """
        assert canonicalize_frame(frame) == frame

    def test_returns_a_string_for_non_string_input(self):
        assert canonicalize_frame(None) == "None"


class TestIdentityConversion:
    def test_same_frame_returns_a_copy(self):
        data = sky_frame()
        out = convert_radec_frame(data, "ra", "dec", "J2000", "J2000")

        assert out is not data
        pd.testing.assert_frame_equal(out, data)

    @pytest.mark.parametrize(("a", "b"), [("J2000", "J2000.0"), ("J2000", "EME2000")])
    def test_alias_frames_are_recognised_and_equivalent(self, a, b):
        """Aliases differ textually, so they route through the conversion.

        Whatever internal route is taken, the result must be the identity,
        which is what makes the alias map safe.  ``EME2000`` in particular is
        only reachable through the alias map, so this also guards against it
        being classified as an unknown equinox.
        """
        data = sky_frame()
        out = convert_radec_frame(data, "ra", "dec", a, b)

        assert np.allclose(out["ra"], data["ra"])
        assert np.allclose(out["dec"], data["dec"])


class TestSpiceConversions:
    @pytest.mark.parametrize(
        ("source", "target"),
        [
            ("J2000", "ECLIPJ2000"),
            ("ECLIPJ2000", "J2000"),
            ("J2000", "B1950"),
            ("B1950", "J2000"),
            ("ECLIPJ2000", "B1950"),
        ],
    )
    def test_fixed_frame_conversions_need_no_epoch(self, source, target):
        data = sky_frame()[["ra", "dec"]]
        out = convert_radec_frame(data, "ra", "dec", source, target)

        assert not np.allclose(out["ra"], data["ra"]) or not np.allclose(out["dec"], data["dec"])

    def test_j2000_to_ecliptic_is_the_obliquity(self):
        """The J2000 -> ECLIPJ2000 rotation is the 23.44 degree obliquity."""
        data = pd.DataFrame({"ra": [0.0], "dec": [0.0]})
        out = convert_radec_frame(data, "ra", "dec", "J2000", "ECLIPJ2000")

        # The vernal equinox is shared by both frames, so it is a fixed point.
        assert np.isclose(out["ra"].iloc[0], 0.0, atol=1e-12)
        assert np.isclose(out["dec"].iloc[0], 0.0, atol=1e-12)

        # The north celestial pole tilts to ecliptic latitude 90 - obliquity.
        pole = pd.DataFrame({"ra": [0.0], "dec": [90.0]})
        out = convert_radec_frame(pole, "ra", "dec", "J2000", "ECLIPJ2000")
        assert np.isclose(out["dec"].iloc[0], 90.0 - 23.4392911, atol=1e-6)

    def test_fixed_frames_ignore_a_supplied_epoch(self):
        """J2000 <-> ECLIPJ2000 is a constant rotation, so epochs must not matter."""
        data = sky_frame(8, seed=3)
        out = convert_radec_frame(data, "ra", "dec", "J2000", "ECLIPJ2000")
        no_epoch = convert_radec_frame(data[["ra", "dec"]], "ra", "dec", "J2000", "ECLIPJ2000")

        assert np.allclose(out["ra"], no_epoch["ra"])
        assert np.allclose(out["dec"], no_epoch["dec"])

    def test_conversion_is_reversible(self):
        data = sky_frame()
        forward = convert_radec_frame(data, "ra", "dec", "J2000", "B1950")
        back = convert_radec_frame(forward, "ra", "dec", "B1950", "J2000")

        assert np.max(np.abs(back["dec"] - data["dec"])) < 1e-9
        assert (
            np.max(angular_separation_deg(data["ra"], data["dec"], back["ra"], back["dec"])) < 1e-9
        )


class TestOfDateConversions:
    @pytest.mark.parametrize(
        ("source", "target"),
        [("date", "J2000"), ("J2000", "date"), ("date", "ECLIPJ2000"), ("date", "B1950")],
    )
    def test_of_date_conversions_round_trip(self, source, target):
        data = sky_frame()
        forward = convert_radec_frame(data, "ra", "dec", source, target, time_column="epoch")
        back = convert_radec_frame(forward, "ra", "dec", target, source, time_column="epoch")

        assert np.max(np.abs(back["dec"] - data["dec"])) < 1e-8
        assert (
            np.max(angular_separation_deg(data["ra"], data["dec"], back["ra"], back["dec"])) < 1e-8
        )

    def test_of_date_requires_an_epoch(self):
        data = sky_frame()[["ra", "dec"]]
        with pytest.raises(ValueError, match="time_column is required"):
            convert_radec_frame(data, "ra", "dec", "date", "J2000")

    def test_of_date_at_j2000_is_nearly_the_identity(self):
        data = pd.DataFrame({"ra": [10.0, 200.0], "dec": [20.0, -40.0]})
        data["epoch"] = J2000_EPOCH
        out = convert_radec_frame(data, "ra", "dec", "date", "J2000", time_column="epoch")

        assert np.max(np.abs(out["dec"] - data["dec"])) < 0.01
        assert np.max(angular_separation_deg(data["ra"], data["dec"], out["ra"], out["dec"])) < 0.01

    def test_precession_grows_with_the_epoch_difference(self):
        """A larger gap from J2000 means a larger precession displacement."""
        ra, dec = 10.0, 20.0
        displacements = []
        for years in (5, 25, 50):
            data = pd.DataFrame({"ra": [ra], "dec": [dec], "epoch": [years * 365.25 * 86400.0]})
            out = convert_radec_frame(data, "ra", "dec", "date", "J2000", time_column="epoch")
            displacements.append(
                float(angular_separation_deg(ra, dec, out["ra"].iloc[0], out["dec"].iloc[0]))
            )

        assert displacements[0] < displacements[1] < displacements[2]

    def test_missing_epoch_column_raises(self):
        data = sky_frame()
        with pytest.raises(ValueError, match="not present in the dataframe"):
            convert_radec_frame(data, "ra", "dec", "date", "J2000", time_column="nope")


class TestIcrSAliasing:
    """ICRS/ICRF are aliases for J2000, not a distinct frame.

    The SPICE kernels this pipeline loads are ICRF-referenced and labelled
    J2000, so NAIF prescribes no transformation between them.  Applying the
    FK5 frame bias here would introduce a ~25 mas error rather than remove one.
    """

    @pytest.mark.parametrize("target", ["J2000", "ECLIPJ2000", "B1950"])
    def test_icrs_to_other_and_back(self, target):
        data = sky_frame()
        forward = convert_radec_frame(data, "ra", "dec", "ICRS", target, time_column="epoch")
        back = convert_radec_frame(forward, "ra", "dec", target, "ICRS", time_column="epoch")

        assert np.max(np.abs(back["dec"] - data["dec"])) < 1e-8
        assert (
            np.max(angular_separation_deg(data["ra"], data["dec"], back["ra"], back["dec"])) < 1e-8
        )

    def test_icrs_to_j2000_is_the_identity(self):
        """No frame bias is applied: the loaded kernels are ICRF-referenced."""
        data = sky_frame(50, seed=11)
        out = convert_radec_frame(data, "ra", "dec", "ICRS", "J2000")

        assert_frame_converted_equally(out, data)

    @pytest.mark.parametrize("target", ["J2000", "ECLIPJ2000", "B1950"])
    def test_icrs_matches_j2000_for_every_target(self, target):
        """An ICRS conversion must equal the same conversion from J2000."""
        data = sky_frame(50, seed=13)
        assert_frame_converted_equally(
            convert_radec_frame(data, "ra", "dec", "ICRS", target, time_column="epoch"),
            convert_radec_frame(data, "ra", "dec", "J2000", target, time_column="epoch"),
        )

    def test_icrs_to_j2000_needs_no_epoch(self):
        """The alias is a fixed frame, so no time is required."""
        data = sky_frame()[["ra", "dec"]]
        with_epoch = convert_radec_frame(sky_frame(), "ra", "dec", "ICRS", "J2000")
        without = convert_radec_frame(data, "ra", "dec", "ICRS", "J2000")

        assert np.allclose(with_epoch["ra"], without["ra"])
        assert np.allclose(with_epoch["dec"], without["dec"])

    @pytest.mark.parametrize("name", sorted(ICRS_FRAME_NAMES))
    def test_both_icrs_spellings_agree(self, name):
        data = sky_frame()
        assert_frame_converted_equally(
            convert_radec_frame(data, "ra", "dec", name, "J2000"),
            convert_radec_frame(data, "ra", "dec", "ICRS", "J2000"),
        )


def assert_frame_converted_equally(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Assert two conversions produced the same angles."""
    assert np.allclose(left["ra"], right["ra"], atol=1e-12)
    assert np.allclose(left["dec"], right["dec"], atol=1e-12)


class TestAngleUnits:
    def test_degrees_round_trip_through_radians(self):
        data = sky_frame()
        as_deg = convert_radec_frame(data, "ra", "dec", "J2000", "ECLIPJ2000")

        radians = data.copy()
        radians["ra"] = np.deg2rad(radians["ra"])
        radians["dec"] = np.deg2rad(radians["dec"])
        as_rad = convert_radec_frame(radians, "ra", "dec", "J2000", "ECLIPJ2000", angle_unit="rad")

        assert np.allclose(np.rad2deg(as_rad["ra"]), as_deg["ra"])
        assert np.allclose(np.rad2deg(as_rad["dec"]), as_deg["dec"])

    def test_radians_are_used_raw(self):
        """With ``angle_unit='rad'`` the columns must already be radians."""
        data = pd.DataFrame({"ra": [np.pi], "dec": [0.0]})
        out = convert_radec_frame(data, "ra", "dec", "J2000", "ECLIPJ2000", angle_unit="rad")

        assert np.isclose(out["ra"].iloc[0], np.pi, atol=1e-12)

    def test_output_keeps_the_input_unit(self):
        data = pd.DataFrame({"ra": [1.0], "dec": [0.5]})
        out = convert_radec_frame(data, "ra", "dec", "J2000", "ECLIPJ2000", angle_unit="rad")

        assert abs(out["ra"].iloc[0]) <= np.pi
        assert abs(out["dec"].iloc[0]) <= np.pi / 2

    def test_unit_argument_is_case_insensitive(self):
        data = pd.DataFrame({"ra": [0.5], "dec": [0.25]})
        lower = convert_radec_frame(data, "ra", "dec", "J2000", "ECLIPJ2000", angle_unit="rad")
        upper = convert_radec_frame(data, "ra", "dec", "J2000", "ECLIPJ2000", angle_unit="RAD")

        assert np.allclose(lower["ra"], upper["ra"])


class TestRaWrap:
    def test_wrapped_output_stays_in_zero_two_pi(self):
        data = sky_frame(200, seed=20)
        out = convert_radec_frame(
            data, "ra", "dec", "date", "J2000", time_column="epoch", ra_wrap=True
        )

        assert (out["ra"] >= 0.0).all()
        assert (out["ra"] < 360.0).all()

    def test_unwrapped_output_uses_the_principal_interval(self):
        data = sky_frame(200, seed=21)
        out = convert_radec_frame(
            data, "ra", "dec", "date", "J2000", time_column="epoch", ra_wrap=False
        )

        assert (out["ra"] >= -180.0).all()
        assert (out["ra"] <= 180.0).all()

    def test_the_two_conventions_differ_only_by_full_turns(self):
        data = sky_frame(200, seed=22)
        wrapped = convert_radec_frame(
            data, "ra", "dec", "date", "J2000", time_column="epoch", ra_wrap=True
        )
        unwrapped = convert_radec_frame(
            data, "ra", "dec", "date", "J2000", time_column="epoch", ra_wrap=False
        )

        difference = (wrapped["ra"].to_numpy() - unwrapped["ra"].to_numpy()) / 360.0
        assert np.allclose(difference, np.round(difference))

    def test_ra_wrap_is_honoured_for_icrs_input(self):
        """The ICRS alias must respect ``ra_wrap`` like every other route."""
        rng = np.random.default_rng(23)
        ra = np.concatenate(
            [rng.uniform(0.0, 0.5, 50), rng.uniform(2 * np.pi - 0.5, 2 * np.pi, 50)]
        )
        data = pd.DataFrame({"ra": np.deg2rad(ra), "dec": np.deg2rad(rng.uniform(-0.5, 0.5, 100))})

        unwrapped = convert_radec_frame(
            data, "ra", "dec", "ICRS", "J2000", ra_wrap=False, angle_unit="rad"
        )

        assert (unwrapped["ra"] >= -np.pi).all()
        assert (unwrapped["ra"] <= np.pi).all()


class TestNaNPreservation:
    @pytest.mark.parametrize(
        ("source", "target"),
        [("J2000", "ECLIPJ2000"), ("date", "J2000"), ("ICRS", "J2000"), ("J2000", "B1950")],
    )
    def test_nan_rows_survive_every_route(self, source, target):
        data = sky_frame(10, seed=30)
        data.loc[3, "ra"] = np.nan
        data.loc[5, "dec"] = np.nan

        out = convert_radec_frame(data, "ra", "dec", source, target, time_column="epoch")

        assert np.isnan(out["ra"].iloc[3])
        assert np.isnan(out["ra"].iloc[5])
        assert np.isnan(out["dec"].iloc[5])
        assert np.isfinite(out["ra"].iloc[0])

    def test_nan_epochs_do_not_poison_neighbouring_rows(self):
        data = sky_frame(10, seed=31)
        data.loc[4, "epoch"] = np.nan

        out = convert_radec_frame(data, "ra", "dec", "date", "J2000", time_column="epoch")

        finite = out.loc[out.index != 4, "ra"]
        assert np.isfinite(finite).all()

    def test_all_nan_input_does_not_raise(self):
        data = pd.DataFrame({"ra": [np.nan] * 5, "dec": [np.nan] * 5, "epoch": [0.0] * 5})
        out = convert_radec_frame(data, "ra", "dec", "date", "J2000", time_column="epoch")

        assert out["ra"].isna().all()


class TestInputValidation:
    def test_rejects_non_dataframe_input(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            convert_radec_frame({"ra": [1.0]}, "ra", "dec", "J2000", "ECLIPJ2000")

    def test_rejects_unknown_input_frame(self):
        data = sky_frame()
        with pytest.raises(ValueError, match="Unrecognised frame name"):
            convert_radec_frame(data, "ra", "dec", "nonsense", "J2000")

    def test_rejects_unknown_output_frame(self):
        data = sky_frame()
        with pytest.raises(ValueError, match="Unrecognised frame name"):
            convert_radec_frame(data, "ra", "dec", "J2000", "nonsense")

    def test_rejects_aberration_only_conversion(self):
        data = sky_frame()
        with pytest.raises(ValueError, match="observer velocity"):
            convert_radec_frame(data, "ra", "dec", "astrometric", "apparent")

    def test_the_error_message_lists_the_accepted_names(self):
        data = sky_frame()
        with pytest.raises(ValueError) as excinfo:
            convert_radec_frame(data, "ra", "dec", "nonsense", "J2000")

        message = str(excinfo.value)
        assert "J2000" in message
        assert "ICRS" in message
        assert "astrometric" in message

    def test_does_not_mutate_the_input(self):
        data = sky_frame()
        original = data.copy()
        convert_radec_frame(data, "ra", "dec", "J2000", "ECLIPJ2000")

        pd.testing.assert_frame_equal(data, original)


class TestBatchBehaviour:
    def test_row_count_and_columns_are_preserved(self):
        data = sky_frame(37, seed=40)
        out = convert_radec_frame(data, "ra", "dec", "date", "J2000", time_column="epoch")

        assert len(out) == len(data)
        assert list(out.columns) == list(data.columns)

    def test_each_row_uses_its_own_epoch(self):
        """Two rows at the same angles but different epochs must differ."""
        data = pd.DataFrame({"ra": [10.0, 10.0], "dec": [20.0, 20.0], "epoch": [0.0, 1.0e9]})
        out = convert_radec_frame(data, "ra", "dec", "date", "J2000", time_column="epoch")

        # 1e9 s is 31.69 years.  General precession is about 46 arcsec per year
        # in RA at the equator, scaled by 1/cos(dec), so at dec = +20 degrees the
        # expected displacement is ~1550 arcsec = 0.43 degrees.  Astropy's FK5
        # gives 1543.68 arcsec for this exact input.
        separation_arcsec = (
            angular_separation_deg(
                out["ra"].iloc[0], out["dec"].iloc[0], out["ra"].iloc[1], out["dec"].iloc[1]
            )
            * 3600.0
        )
        assert 1500.0 < separation_arcsec < 1600.0

    def test_declination_change_is_bounded_by_precession(self):
        """Declination must not swing wildly at a moderate epoch offset."""
        data = pd.DataFrame({"ra": [10.0], "dec": [20.0], "epoch": [-1.6e9]})
        out = convert_radec_frame(data, "ra", "dec", "date", "J2000", time_column="epoch")

        assert abs(out["dec"].iloc[0] - 20.0) < 1.0

    def test_batch_matches_row_by_row(self):
        """Vectorised conversion must equal converting rows one at a time."""
        data = sky_frame(15, seed=41)
        batch = convert_radec_frame(data, "ra", "dec", "date", "J2000", time_column="epoch")

        singles = [
            convert_radec_frame(
                data.iloc[[index]], "ra", "dec", "date", "J2000", time_column="epoch"
            )
            for index in range(len(data))
        ]
        for index, single in enumerate(singles):
            assert np.isclose(batch["ra"].iloc[index], single["ra"].iloc[0], atol=1e-9)
            assert np.isclose(batch["dec"].iloc[index], single["dec"].iloc[0], atol=1e-9)


class TestCartesianConversions:
    def test_same_frame_returns_a_copy(self):
        data = pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]})
        out = convert_cartesian_frame(data, "x", "y", "z", "J2000", "J2000")

        assert out is not data
        pd.testing.assert_frame_equal(out, data)

    def test_rotation_preserves_the_norm(self):
        rng = np.random.default_rng(50)
        data = pd.DataFrame(
            {
                "x": rng.normal(size=20),
                "y": rng.normal(size=20),
                "z": rng.normal(size=20),
                "epoch": np.resize(EPOCHS, 20),
            }
        )
        out = convert_cartesian_frame(
            data, "x", "y", "z", "J2000", "ECLIPJ2000", time_column="epoch"
        )

        before = np.linalg.norm(data[["x", "y", "z"]].to_numpy(), axis=1)
        after = np.linalg.norm(out[["x", "y", "z"]].to_numpy(), axis=1)
        assert np.allclose(before, after)

    @pytest.mark.parametrize(
        ("source", "target"), [("J2000", "ECLIPJ2000"), ("J2000", "B1950"), ("B1950", "J2000")]
    )
    def test_cartesian_round_trip(self, source, target):
        rng = np.random.default_rng(51)
        data = pd.DataFrame(
            {"x": rng.normal(size=20), "y": rng.normal(size=20), "z": rng.normal(size=20)}
        )
        forward = convert_cartesian_frame(data, "x", "y", "z", source, target)
        back = convert_cartesian_frame(forward, "x", "y", "z", target, source)

        assert np.allclose(back[["x", "y", "z"]].to_numpy(), data[["x", "y", "z"]].to_numpy())

    def test_missing_time_column_defaults_to_epoch_zero(self):
        """The fixed frames need no epoch, so omitting the column must work."""
        data = pd.DataFrame({"x": [1.0], "y": [0.0], "z": [0.0]})
        explicit = convert_cartesian_frame(
            data.assign(epoch=0.0), "x", "y", "z", "J2000", "ECLIPJ2000", time_column="epoch"
        )
        implicit = convert_cartesian_frame(data, "x", "y", "z", "J2000", "ECLIPJ2000")

        assert np.allclose(
            explicit[["x", "y", "z"]].to_numpy(), implicit[["x", "y", "z"]].to_numpy()
        )

    def test_matches_the_radec_route(self):
        """Rotating vectors and rotating their angles must agree.

        The same sky positions are converted both ways, so the RA/Dec recovered
        from the rotated vectors has to equal the RA/Dec the angle route
        returned.
        """
        data = sky_frame(20, seed=52)
        vectors = radec_to_direction_vectors(data, "ra", "dec")
        rotated = convert_cartesian_frame(vectors, "obs_x", "obs_y", "obs_z", "J2000", "ECLIPJ2000")

        recovered_ra = (
            np.rad2deg(np.arctan2(rotated["obs_y"].to_numpy(), rotated["obs_x"].to_numpy())) % 360.0
        )
        recovered_dec = np.rad2deg(np.arcsin(np.clip(rotated["obs_z"].to_numpy(), -1.0, 1.0)))

        from_angles = convert_radec_frame(data, "ra", "dec", "J2000", "ECLIPJ2000")

        np.testing.assert_allclose(recovered_ra, from_angles["ra"].to_numpy(), atol=1e-9)
        np.testing.assert_allclose(recovered_dec, from_angles["dec"].to_numpy(), atol=1e-9)


class TestDirectionVectors:
    def test_creates_the_expected_columns(self):
        data = sky_frame(5, seed=60)
        out = radec_to_direction_vectors(data, "ra", "dec")

        assert {"obs_x", "obs_y", "obs_z"}.issubset(out.columns)

    def test_vectors_are_unit_length(self):
        data = sky_frame(20, seed=61)
        out = radec_to_direction_vectors(data, "ra", "dec")

        norms = np.linalg.norm(out[["obs_x", "obs_y", "obs_z"]].to_numpy(), axis=1)
        assert np.allclose(norms, 1.0)

    def test_origin_maps_to_positive_x(self):
        data = pd.DataFrame({"ra": [0.0], "dec": [0.0]})
        out = radec_to_direction_vectors(data, "ra", "dec")

        assert np.isclose(out["obs_x"].iloc[0], 1.0)
        assert np.isclose(out["obs_y"].iloc[0], 0.0)
        assert np.isclose(out["obs_z"].iloc[0], 0.0)

    def test_north_pole_maps_to_positive_z(self):
        data = pd.DataFrame({"ra": [0.0], "dec": [90.0]})
        out = radec_to_direction_vectors(data, "ra", "dec")

        assert np.isclose(out["obs_z"].iloc[0], 1.0)

    def test_accepts_radian_input(self):
        data = pd.DataFrame({"ra": [np.pi / 2], "dec": [0.0]})
        out = radec_to_direction_vectors(data, "ra", "dec", angle_unit="rad")

        assert np.isclose(out["obs_y"].iloc[0], 1.0, atol=1e-12)

    def test_input_columns_are_kept(self):
        data = sky_frame(5, seed=62)
        out = radec_to_direction_vectors(data, "ra", "dec")

        assert "ra" in out.columns
        assert "dec" in out.columns

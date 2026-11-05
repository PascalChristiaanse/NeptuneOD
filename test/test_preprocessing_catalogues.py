"""Tests for ``orbitdet.preprocessing.catalogues``."""

import numpy as np
import pandas as pd
import pytest

from orbitdet.preprocessing.catalogues import (
    _SECONDS_PER_JULIAN_YEAR,
    CATALOGUE_EPOCHS,
    CATALOGUE_TIES,
    apply_proper_motion,
    catalogue_epoch_seconds_since_j2000,
    describe_tie,
    normalize_catalogue,
    normalize_catalogue_series,
    tie_to_icrf,
)
from orbitdet.preprocessing.schema import ReductionCatalogue

#: One milliarcsecond in radians.
MAS = np.pi / 648000.0 / 1000.0

#: One arcsecond in radians.
ARCSEC = np.pi / 648000.0


class TestNormalizeCatalogue:
    @pytest.mark.parametrize(
        "label, expected",
        [
            ("Gaia DR2", ReductionCatalogue.GAIA_DR2),
            ("Gaia DR1", ReductionCatalogue.GAIA_DR1),
            ("gaia dr2", ReductionCatalogue.GAIA_DR2),
            ("Catalogue Tycho-2", ReductionCatalogue.TYCHO2),
            ("Tycho-2", ReductionCatalogue.TYCHO2),
            ("Catalogue - Tyc2", ReductionCatalogue.TYCHO2),
            ("Catalog SAO", ReductionCatalogue.SAO),
            ("Catalogue UCAC2", ReductionCatalogue.UCAC2),
            ("UCAC2", ReductionCatalogue.UCAC2),
            ("Catalogue UCAC4", ReductionCatalogue.UCAC4),
        ],
    )
    def test_recognises_real_labels(self, label, expected):
        assert normalize_catalogue(label) is expected

    def test_given_in_data_typo(self):
        # nm0077/nm0078/nm0079 spell it "diven".
        assert normalize_catalogue("Catalogue diven in the data") is (
            ReductionCatalogue.GIVEN_IN_DATA
        )

    def test_given_in_data_correct_spelling(self):
        assert normalize_catalogue("Catalogue given in the data") is (
            ReductionCatalogue.GIVEN_IN_DATA
        )

    def test_act_is_accepted_as_ast_typo(self):
        # nm0077 carries per-row values of "ACT", a typo for "AST".
        assert normalize_catalogue("ACT") is ReductionCatalogue.AST

    def test_multi_catalogue_label_resolves_to_the_first(self):
        label = "1 - Catalogue AST, 2 -Catalogue Tyc-2 (ICRF)"

        assert normalize_catalogue(label) is ReductionCatalogue.AST

    @pytest.mark.parametrize(
        "label",
        [
            "no information",
            "no inf",
            "no info",
            "none",
            "",
            None,
            "nan",
            "unknown",
        ],
    )
    def test_absent_catalogue_is_none(self, label):
        assert normalize_catalogue(label) is ReductionCatalogue.NONE

    def test_correction_prose_is_not_a_catalogue(self):
        label = (
            "positions are corrected for full relativistic aberration but not "
            "for relativistic light deflection in the gravitational field of the "
            "Solar System"
        )

        assert normalize_catalogue(label) is ReductionCatalogue.NONE

    def test_unrecognised_catalogue_is_unknown_and_warns(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            result = normalize_catalogue("Hipparcos")

        assert result is ReductionCatalogue.UNKNOWN
        assert any("names no known catalogue" in r.getMessage() for r in caplog.records)

    def test_punctuation_and_spacing_are_tolerated(self):
        assert normalize_catalogue("  Gaia   DR2  ") is ReductionCatalogue.GAIA_DR2

    def test_tycho_matched_before_tyc(self):
        # "tycho" and "tyc" are both patterns; both must give TYCHO2.
        assert normalize_catalogue("tycho") is ReductionCatalogue.TYCHO2
        assert normalize_catalogue("tyc") is ReductionCatalogue.TYCHO2

    def test_dr1_matched_before_dr2(self):
        assert normalize_catalogue("Gaia DR1") is ReductionCatalogue.GAIA_DR1
        assert normalize_catalogue("Gaia DR2") is ReductionCatalogue.GAIA_DR2


class TestNormalizeCatalogueSeries:
    def test_maps_elementwise(self):
        values = pd.Series(["Gaia DR2", "no information", "UCAC2"])

        out = normalize_catalogue_series(values)

        assert out.tolist() == ["gaia_dr2", "none", "ucac2"]

    def test_preserves_length_and_index(self):
        values = pd.Series(["Gaia DR2", "SAO"], index=[10, 20])

        out = normalize_catalogue_series(values)

        assert out.index.tolist() == [10, 20]


class TestCatalogueEpochs:
    def test_tycho2_epoch_is_j1991_25(self):
        epoch = catalogue_epoch_seconds_since_j2000(ReductionCatalogue.TYCHO2)

        assert epoch / _SECONDS_PER_JULIAN_YEAR + 2000.0 == pytest.approx(1991.25)

    def test_gaia_dr2_epoch_is_j2015_5(self):
        epoch = catalogue_epoch_seconds_since_j2000("gaia_dr2")

        assert epoch / _SECONDS_PER_JULIAN_YEAR + 2000.0 == pytest.approx(2015.5)

    def test_accepts_string_value(self):
        assert catalogue_epoch_seconds_since_j2000("tycho2") == pytest.approx(
            CATALOGUE_EPOCHS[ReductionCatalogue.TYCHO2]
        )

    def test_undated_catalogue_returns_none(self):
        assert catalogue_epoch_seconds_since_j2000(ReductionCatalogue.NONE) is None

    def test_given_in_data_is_undated(self):
        assert catalogue_epoch_seconds_since_j2000(ReductionCatalogue.GIVEN_IN_DATA) is None


class TestApplyProperMotion:
    def test_zero_elapsed_time_is_identity(self):
        ra, dec = apply_proper_motion(
            np.array([1.0]),
            np.array([0.5]),
            np.array([1e-3]),
            np.array([1e-3]),
            np.array([0.0]),
            0.0,
        )

        assert ra[0] == pytest.approx(1.0)
        assert dec[0] == pytest.approx(0.5)

    def test_motion_accumulates_over_elapsed_years(self):
        # 1e-3 rad/yr for 100 yr -> +0.1 rad.
        ra, _ = apply_proper_motion(
            np.array([0.0]),
            np.array([0.0]),
            np.array([1e-3]),
            np.array([0.0]),
            np.array([100 * _SECONDS_PER_JULIAN_YEAR]),
            0.0,
        )

        assert ra[0] == pytest.approx(0.1)

    def test_negative_elapsed_time_moves_backwards(self):
        ra, _ = apply_proper_motion(
            np.array([0.0]),
            np.array([0.0]),
            np.array([1e-3]),
            np.array([0.0]),
            np.array([0.0]),
            100 * _SECONDS_PER_JULIAN_YEAR,
        )

        assert ra[0] == pytest.approx(-0.1)

    def test_dec_component_is_independent(self):
        _, dec = apply_proper_motion(
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([2e-3]),
            np.array([50 * _SECONDS_PER_JULIAN_YEAR]),
            0.0,
        )

        assert dec[0] == pytest.approx(0.1)

    def test_scalar_catalogue_epoch_broadcasts(self):
        ra, _ = apply_proper_motion(
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([1e-3, 1e-3]),
            np.array([0.0, 0.0]),
            np.array([0.0, 100 * _SECONDS_PER_JULIAN_YEAR]),
            0.0,
        )

        assert ra[0] == pytest.approx(0.0)
        assert ra[1] == pytest.approx(0.1)

    def test_nan_position_stays_nan(self):
        ra, dec = apply_proper_motion(
            np.array([np.nan]),
            np.array([0.0]),
            np.array([1e-3]),
            np.array([1e-3]),
            np.array([1e4]),
            0.0,
        )

        assert np.isnan(ra[0])
        assert np.isnan(dec[0])

    def test_nan_proper_motion_stays_nan(self):
        # An uncorrected coordinate must not be passed off as corrected.
        ra, _ = apply_proper_motion(
            np.array([1.0]),
            np.array([0.0]),
            np.array([np.nan]),
            np.array([np.nan]),
            np.array([1e4]),
            0.0,
        )

        assert np.isnan(ra[0])

    def test_nan_epoch_stays_nan(self):
        ra, _ = apply_proper_motion(
            np.array([1.0]),
            np.array([0.0]),
            np.array([1e-3]),
            np.array([0.0]),
            np.array([np.nan]),
            0.0,
        )

        assert np.isnan(ra[0])

    def test_nan_propagates_row_wise(self):
        ra, _ = apply_proper_motion(
            np.array([0.0, np.nan, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1e-3, 1e-3, 1e-3]),
            np.array([0.0, 0.0, 0.0]),
            np.array([10 * _SECONDS_PER_JULIAN_YEAR] * 3),
            0.0,
        )

        assert np.isfinite(ra[0])
        assert np.isnan(ra[1])
        assert np.isfinite(ra[2])

    def test_tycho2_century_scale_displacement_is_arcseconds(self):
        # The point of the module: 50 mas/yr over a century is 5 arcsec.
        elapsed = 100 * _SECONDS_PER_JULIAN_YEAR
        ra, _ = apply_proper_motion(
            np.array([0.0]),
            np.array([0.0]),
            np.array([50 * MAS]),
            np.array([0.0]),
            np.array([elapsed]),
            0.0,
        )

        assert ra[0] / ARCSEC == pytest.approx(5.0)


class TestTieToIcrf:
    def test_gaia_dr2_identity_leaves_coordinates_alone(self):
        ra, dec, report = tie_to_icrf(
            np.array([1.0]), np.array([0.5]), catalogue="gaia_dr2", epoch=np.array([0.0])
        )

        assert ra[0] == pytest.approx(1.0)
        assert dec[0] == pytest.approx(0.5)
        assert report.has_published_tie

    def test_given_in_data_reports_no_published_tie(self):
        _, _, report = tie_to_icrf(
            np.array([1.0]),
            np.array([0.5]),
            catalogue="given_in_data",
            epoch=np.array([0.0]),
        )

        assert not report.has_published_tie

    def test_tycho2_without_proper_motion_warns(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            _, _, report = tie_to_icrf(
                np.array([1.0]), np.array([0.5]), catalogue="tycho2", epoch=np.array([1e8])
            )

        assert not report.proper_motion_applied
        assert any("proper motion was supplied" in r.getMessage() for r in caplog.records)

    def test_tycho2_with_proper_motion_applies_it(self):
        # Tycho-2 positions are at J1991.25, so an observation 100 yr after J2000
        # is 108.75 yr after the catalogue epoch, not 100.
        ra, _, report = tie_to_icrf(
            np.array([0.0]),
            np.array([0.0]),
            catalogue="tycho2",
            epoch=np.array([100 * _SECONDS_PER_JULIAN_YEAR]),
            pm_first=np.array([1e-3]),
            pm_second=np.array([0.0]),
        )

        assert report.proper_motion_applied
        expected = 1e-3 * (100.0 - (-8.75))
        assert ra[0] == pytest.approx(expected)

    def test_undated_catalogue_warns_it_cannot_propagate(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            _, _, report = tie_to_icrf(
                np.array([1.0]),
                np.array([0.5]),
                catalogue="given_in_data",
                epoch=np.array([0.0]),
                pm_first=np.array([1e-3]),
                pm_second=np.array([1e-3]),
            )

        assert not report.proper_motion_applied
        assert any("no reference epoch" in r.getMessage() for r in caplog.records)

    def test_unrecognised_catalogue_name_raises(self):
        # Every catalogue in the vocabulary has a tie, so the only way to reach
        # a lookup failure is a name outside the vocabulary entirely.
        with pytest.raises(ValueError):
            tie_to_icrf(
                np.array([1.0]),
                np.array([0.5]),
                catalogue="hipparcos",
                epoch=np.array([0.0]),
            )

    def test_accepts_a_scalar_override_epoch(self):
        ra, _, report = tie_to_icrf(
            np.array([0.0]),
            np.array([0.0]),
            catalogue="tycho2",
            epoch=np.array([10 * _SECONDS_PER_JULIAN_YEAR]),
            pm_first=np.array([1e-3]),
            pm_second=np.array([0.0]),
            catalogue_epoch=0.0,
        )

        assert report.proper_motion_applied
        assert ra[0] == pytest.approx(0.01)

    def test_report_records_the_catalogue(self):
        _, _, report = tie_to_icrf(
            np.array([1.0]), np.array([0.5]), catalogue="ucac2", epoch=np.array([0.0])
        )

        assert report.catalogue is ReductionCatalogue.UCAC2

    def test_tie_results_match_or_exceed_input_shape(self):
        ra, dec, _ = tie_to_icrf(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.1, 0.2, 0.3]),
            catalogue="gaia_dr2",
            epoch=np.zeros(3),
        )

        assert ra.shape == (3,)
        assert dec.shape == (3,)


class TestVocabularyCompleteness:
    def test_every_catalogue_enum_member_has_a_tie(self):
        missing = [c for c in ReductionCatalogue if c not in CATALOGUE_TIES]

        assert missing == []

    def test_only_gaia_entries_claim_a_published_tie(self):
        # Documents the module's honest limitation: no non-Gaia catalogue has a
        # real orientation field available from local data.
        published = {c.value for c, t in CATALOGUE_TIES.items() if t.has_published_tie}

        assert published == {"gaia_dr1", "gaia_dr2"}

    def test_every_tie_has_notes_explaining_itself(self):
        missing = [c.value for c, t in CATALOGUE_TIES.items() if not t.notes]

        assert missing == []

    def test_undated_catalogues_have_no_epoch_entry(self):
        for catalogue in (ReductionCatalogue.NONE, ReductionCatalogue.UNKNOWN):
            assert CATALOGUE_EPOCHS.get(catalogue) is None


class TestDescribeTie:
    def test_names_the_frame(self):
        assert "ICRS" in describe_tie("gaia_dr2")

    def test_marks_unaligned_catalogues(self):
        assert "NOT aligned" in describe_tie("tycho2")

    def test_marks_aligned_catalogues(self):
        description = describe_tie("gaia_dr2")

        assert "aligned" in description
        assert "NOT aligned" not in description

    def test_reports_undated(self):
        assert "undated" in describe_tie("given_in_data")

"""Tests for the Gaia standard-frame reader.

The reader's job is narrow but load-bearing: it turns ``GaiaAstrometry``'s table
into the standard frame, and it must not disturb the two things that make Gaia
different from every other source - the per-transit covariance (which the frame
cannot carry) and the scan angle (which it can, and which used to live in a
separate epoch-keyed registry).

Retrieval is stubbed.  The archive is cached on disk for a week but a test must
not depend on the network, and the mapping is what is under test, not astroquery.
"""

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from orbitdet.data.frame_sources import registered_frame_sources, standard_frame_for
from orbitdet.data.gaia_source import GaiaSource
from orbitdet.preprocessing.schema import DataSource, FrameCenter, verify

TARGET_SOURCE_ID = -4293959295


def gaia_config(**overrides):
    """A Gaia dataset config, shaped like the committed ``gaia.yaml``."""
    base = {
        "identifier": "gaia_triton",
        "type": "gaia",
        "target": "Triton",
        "source_ids": [TARGET_SOURCE_ID],
        "filter_outcomes": True,
        "epoch_of_equinox": "ICRS",
        # Corrections off: they need a system of bodies and are covered
        # separately.  The mapping is the same either way.
        "correct_photocenter": False,
        "light_deflection": [],
    }
    base.update(overrides)
    return OmegaConf.create(base)


@pytest.fixture
def gaia_table():
    """A small table shaped like GaiaAstrometry's, in radians and TDB seconds."""
    return pd.DataFrame(
        {
            "source_id": [TARGET_SOURCE_ID] * 3,
            "number_mp": [0, 0, 0],
            "denomination": ["triton"] * 3,
            "transit_id": [1, 1, 2],
            "epoch": [4.677331e08, 4.6773315e08, 4.677332e08],
            "ra": [-0.404770, -0.404760, np.nan],
            "dec": [0.523599, 0.523600, 0.523601],
            "position_angle_scan": [5.289061, 5.289064, 5.289070],
            "ra_error_random": [1.0e-6, 1.1e-6, 1.2e-6],
            "dec_error_random": [1.0e-6, 1.1e-6, 1.2e-6],
            "ra_dec_correlation_random": [0.1, 0.1, 0.1],
            "ra_error_systematic": [2.0e-7, 2.0e-7, 2.0e-7],
            "dec_error_systematic": [2.0e-7, 2.0e-7, 2.0e-7],
            "ra_dec_correlation_systematic": [0.0, 0.0, 0.0],
        }
    )


def make_source(gaia_table):
    """A :class:`GaiaSource` whose table is pre-supplied, so nothing is fetched."""
    source = GaiaSource(OmegaConf.create({"global_frame_orientation": "J2000"}))
    source._supplied_table = gaia_table
    return source


class TestColumnMapping:
    def test_radians_pass_through_unmodified(self, gaia_table):
        # Gaia's table is already in radians, so unlike NSDB and Voyager there is
        # no unit conversion - a degree conversion here would corrupt everything.
        frame = make_source(gaia_table).frame_from_table(gaia_table, gaia_config())

        assert frame["first_value"].iloc[0] == pytest.approx(-0.404770)
        assert frame["second_value"].iloc[0] == pytest.approx(0.523599)

    def test_the_epoch_passes_through_unmodified(self, gaia_table):
        frame = make_source(gaia_table).frame_from_table(gaia_table, gaia_config())

        # Row 2 has no RA and is dropped.
        assert frame["epoch_tdb"].tolist() == [4.677331e08, 4.6773315e08]

    def test_rows_without_a_coordinate_are_dropped(self, gaia_table):
        frame = make_source(gaia_table).frame_from_table(gaia_table, gaia_config())

        assert len(frame) == 2


class TestGaiaSpecificColumns:
    def test_the_scan_angle_is_carried_in_the_frame(self, gaia_table):
        # This is what retires the epoch-keyed ``_SCAN_ANGLE_REGISTRY``: the
        # angle is per-observation, so it belongs in a column.
        frame = make_source(gaia_table).frame_from_table(gaia_table, gaia_config())

        assert "scan_angle" in frame.columns
        assert frame["scan_angle"].tolist() == [5.289061, 5.289064]

    def test_a_missing_scan_angle_column_is_tolerated(self, gaia_table):
        # A local-archive read may not carry the scan angle, and that must not
        # make the whole frame invalid.
        table = gaia_table.drop(columns=["position_angle_scan"])

        frame = make_source(table).frame_from_table(table, gaia_config())

        assert frame["scan_angle"].isna().all()
        assert verify(frame) == []

    def test_the_marginal_uncertainty_combines_random_and_systematic(self, gaia_table):
        # Gaia reports the two components separately; the frame carries their
        # quadrature sum, which is the marginal sigma a diagnostic wants.
        frame = make_source(gaia_table).frame_from_table(gaia_table, gaia_config())

        expected = np.sqrt(1.0e-6**2 + 2.0e-7**2)
        assert frame["first_uncertainty"].iloc[0] == pytest.approx(expected)
        assert frame["second_uncertainty"].iloc[0] == pytest.approx(expected)

    def test_uncertainty_falls_back_to_whichever_component_exists(self, gaia_table):
        table = gaia_table.drop(columns=["ra_error_systematic", "dec_error_systematic"])

        frame = make_source(table).frame_from_table(table, gaia_config())

        assert frame["first_uncertainty"].iloc[0] == pytest.approx(1.0e-6)


class TestIdentifierConventions:
    def test_the_receiver_is_named_gaia(self, gaia_table):
        frame = make_source(gaia_table).frame_from_table(gaia_table, gaia_config())

        assert set(frame["observatory_code"]) == {"Gaia"}

    def test_the_frame_centre_is_spacecraft(self, gaia_table):
        # Gaia observes from orbit, so the receiver is a body rather than a
        # ground station - the same distinction Voyager makes.
        frame = make_source(gaia_table).frame_from_table(gaia_table, gaia_config())

        assert set(frame["frame_center"]) == {FrameCenter.SPACECRAFT.value}

    def test_the_data_source_is_the_gaia_archive(self, gaia_table):
        frame = make_source(gaia_table).frame_from_table(gaia_table, gaia_config())

        assert set(frame["data_source"]) == {DataSource.GAIA_ARCHIVE.value}

    def test_the_equinox_is_carried_so_the_shared_conversion_can_rotate(self, gaia_table):
        # ICRS is not just a label: it differs from J2000 by the frame bias, and
        # the shared conversion knows how to apply it.
        frame = make_source(gaia_table).frame_from_table(gaia_table, gaia_config())

        assert set(frame["original_equinox"]) == {"ICRS"}


class TestFrameValidity:
    def test_the_frame_satisfies_the_schema(self, gaia_table):
        frame = make_source(gaia_table).frame_from_table(gaia_table, gaia_config())

        assert verify(frame) == []


class TestCorrectionsRequireEphemerides:
    def test_asking_for_corrections_without_bodies_raises(self):
        # The corrections are computed from ephemerides, so silently skipping
        # them would produce a frame that looks fine and is systematically wrong.
        source = GaiaSource(OmegaConf.create({}), system_of_bodies=None)
        cfg = gaia_config(correct_photocenter=True, diameter=2706800.0)

        with pytest.raises(ValueError, match="need a system of bodies"):
            source._apply_corrections(cfg)

    def test_photocenter_needs_a_positive_diameter(self):
        source = GaiaSource(OmegaConf.create({}), system_of_bodies=object())
        cfg = gaia_config(correct_photocenter=True, diameter=0.0)

        with pytest.raises(ValueError, match="positive 'diameter'"):
            source._apply_corrections(cfg)

    def test_no_corrections_requested_needs_nothing(self):
        source = GaiaSource(OmegaConf.create({}), system_of_bodies=None)

        # Must not raise: this is the common case for a quick prefit look.
        source._apply_corrections(gaia_config())


class TestRegistration:
    def test_gaia_is_registered_as_a_frame_source(self):
        assert "gaia" in registered_frame_sources()

    def test_dispatch_recognises_a_gaia_config(self, monkeypatch):
        calls = []

        import orbitdet.data.gaia_source as gaia_source

        def fake_frame_from_config(self, dataset_cfg):
            calls.append(dataset_cfg.get("identifier"))
            return "FRAME"

        monkeypatch.setattr(gaia_source.GaiaSource, "frame_from_config", fake_frame_from_config)

        result = standard_frame_for(gaia_config(), cfg=OmegaConf.create({}))

        assert result == "FRAME"
        assert calls == ["gaia_triton"]

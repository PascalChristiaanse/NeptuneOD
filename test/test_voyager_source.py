"""Tests for the Voyager standard-frame reader.

The reader turns Jacobson (1991)'s two hand-transcribed tables into a standard
frame.  Two things are worth testing beyond "it parses":

* the mapping is *right*, not merely schema-valid - the measurements are degrees
  in the table and radians in the frame, the epoch is already TDB, and the
  receiver is a spacecraft rather than a ground station; and
* the identifier conventions survive, because the receiver name is what the
  weighting router and the link ends key on.
"""

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from orbitdet.data.frame_sources import (
    register_frame_source,
    registered_frame_sources,
    standard_frame_for,
)
from orbitdet.data.voyager_source import VoyagerSource
from orbitdet.preprocessing.schema import (
    DataSource,
    FrameCenter,
    ObservationType,
    ReceptorType,
    ReductionCatalogue,
    ReferenceFrame,
    verify,
)

ARCSEC_TO_RAD = np.pi / 180.0 / 3600.0


def voyager_config(**overrides):
    """A Voyager dataset config, shaped like the committed ``voyager.yaml``."""
    base = {
        "identifier": "voyager_triton",
        "type": "voyager",
        "data": {"file": "data.csv"},
        "ancillary_files": {"file": "ancillary.csv"},
        "epoch_of_equinox": "B1950",
        "centre_of_frame": "spacecraft",
        "reference_frame": "astrometric",
        "reduction": "no information",
        "receptor": "Vidicon 800x800 CCD",
        "observatory": {"code": -31, "name": "Voyager 2"},
        "satellites": {"Triton": {"designation": "N1-Triton"}},
        "merge_key": "PICNO",
    }
    base.update(overrides)
    return OmegaConf.create(base)


@pytest.fixture
def merged_table(monkeypatch):
    """Stub the shared merge so the reader is tested without the real CSVs.

    The reader deliberately goes through
    :func:`orbitdet.data.voyager_data.load_and_merge_voyager_tables` rather than
    re-reading the tables, so the stub is where the epoch and the RA/Dec columns
    come from - exactly the inputs the real merge supplies.
    """
    table = pd.DataFrame(
        {
            "PICNO": ["51920B+08", "51920B+15", "51920B+22"],
            "jd": [2447290.99259041, 2447290.99646818, 2447290.99646818],
            "alpha (Deg)": [297.25654, 297.25668, np.nan],
            "delta (Deg)": [-18.30199, -18.30199, -18.30199],
            "epoch_TDB": [-3.67546240e08, -3.67545905e08, -3.67545905e08],
        }
    )

    def fake_merge(cfg, dataset_cfg):
        return table.copy()

    import orbitdet.data.voyager_data as voyager_data

    monkeypatch.setattr(voyager_data, "load_and_merge_voyager_tables", fake_merge)
    return table


class TestColumnMapping:
    def test_degrees_are_converted_to_radians(self, merged_table):
        frame = VoyagerSource(OmegaConf.create({})).frame_from_config(voyager_config())

        assert frame["first_value"].iloc[0] == pytest.approx(np.deg2rad(297.25654))
        assert frame["second_value"].iloc[0] == pytest.approx(np.deg2rad(-18.30199))

    def test_the_source_unit_is_recorded_for_provenance(self, merged_table):
        frame = VoyagerSource(OmegaConf.create({})).frame_from_config(voyager_config())

        assert set(frame["first_unit"]) == {"deg"}
        assert set(frame["second_unit"]) == {"deg"}

    def test_the_epoch_passes_through_unmodified(self, merged_table):
        # The table's Julian date is already TDB, so there is no clock to apply.
        # Converting it would shift every observation by its timescale offset.
        frame = VoyagerSource(OmegaConf.create({})).frame_from_config(voyager_config())

        # The fixture's third row has no right ascension and is dropped, so the
        # expected epochs are the usable ones.
        usable = merged_table["alpha (Deg)"].notna()
        assert frame["epoch_tdb"].tolist() == merged_table.loc[usable, "epoch_TDB"].tolist()

    def test_the_satellite_is_the_observed_object(self, merged_table):
        # Voyager measured Triton; the planet is the origin its own state vector
        # is expressed relative to, not the body observed.
        frame = VoyagerSource(OmegaConf.create({})).frame_from_config(voyager_config())

        assert set(frame["observed_object"]) == {"Triton"}

    def test_rows_without_a_usable_measurement_are_dropped(self, merged_table):
        # The fixture's third row has no right ascension.  It cannot become an
        # observation, so it must not reach the factory.
        frame = VoyagerSource(OmegaConf.create({})).frame_from_config(voyager_config())

        assert len(frame) == 2


class TestIdentifierConventions:
    def test_the_receiver_is_the_spacecraft_name(self, merged_table):
        # The name, not the numeric code: the weights router matches on
        # "Voyager 2" and the link end is a body origin, so a code would send the
        # receiver down the ground-station path where nothing exists.
        frame = VoyagerSource(OmegaConf.create({})).frame_from_config(voyager_config())

        assert set(frame["observatory_code"]) == {"Voyager 2"}

    def test_a_config_with_only_a_code_falls_back_to_it(self, merged_table):
        cfg = voyager_config(observatory={"code": -31})

        frame = VoyagerSource(OmegaConf.create({})).frame_from_config(cfg)

        assert set(frame["observatory_code"]) == {"-31"}

    def test_a_config_naming_no_observatory_raises(self, merged_table):
        cfg = voyager_config()
        cfg.pop("observatory")

        with pytest.raises(ValueError, match="names no observatory"):
            VoyagerSource(OmegaConf.create({})).frame_from_config(cfg)

    def test_the_dataset_identifier_is_the_source_set_id(self, merged_table):
        frame = VoyagerSource(OmegaConf.create({})).frame_from_config(voyager_config())

        assert set(frame["source_set_id"]) == {"voyager_triton"}
        assert set(frame["data_source"]) == {DataSource.VOYAGER.value}


class TestDescriptorNormalisation:
    def test_the_descriptor_fields_are_normalised(self, merged_table):
        # These are prose in the config, exactly as in an NSDB descriptor, so
        # they go through the same vocabulary mappers rather than being cast.
        frame = VoyagerSource(OmegaConf.create({})).frame_from_config(voyager_config())

        assert set(frame["frame_center"]) == {FrameCenter.SPACECRAFT.value}
        assert set(frame["original_reference_frame"]) == {ReferenceFrame.ASTROMETRIC.value}
        assert set(frame["original_reduction"]) == {ReductionCatalogue.NONE.value}
        assert set(frame["receptor_type"]) == {ReceptorType.VIDICON.value}
        assert set(frame["observation_type"]) == {ObservationType.ABSOLUTE.value}

    def test_a_spacecraft_label_normalises_to_spacecraft(self, merged_table):
        for label in ("spacecraft", "Space craft", "satellite"):
            frame = VoyagerSource(OmegaConf.create({})).frame_from_config(
                voyager_config(centre_of_frame=label)
            )
            assert set(frame["frame_center"]) == {FrameCenter.SPACECRAFT.value}, label


class TestFrameValidity:
    def test_the_frame_satisfies_the_schema(self, merged_table):
        frame = VoyagerSource(OmegaConf.create({})).frame_from_config(voyager_config())

        assert verify(frame) == []

    def test_the_equinox_is_carried_through_for_the_conversion(self, merged_table):
        # The frame records B1950 so the shared conversion knows what to rotate
        # from; deciding it here would duplicate that logic.
        frame = VoyagerSource(OmegaConf.create({})).frame_from_config(voyager_config())

        assert set(frame["original_equinox"]) == {"B1950"}


class TestMultipleSatellites:
    def test_multiple_satellites_are_rejected(self, merged_table):
        cfg = voyager_config(
            satellites={"Triton": {"designation": "N1-Triton"}, "Nereid": {"designation": "N2"}}
        )

        with pytest.raises(NotImplementedError, match="multiple satellites"):
            VoyagerSource(OmegaConf.create({})).frame_from_config(cfg)

    def test_a_config_with_no_satellite_raises(self, merged_table):
        cfg = voyager_config(satellites={})

        with pytest.raises(ValueError, match="names no satellite"):
            VoyagerSource(OmegaConf.create({})).frame_from_config(cfg)


class TestFrameSourceRegistry:
    def test_voyager_is_registered(self):
        assert "voyager" in registered_frame_sources()

    def test_dispatch_uses_the_dataset_type(self, merged_table):
        cfg = voyager_config()

        frame = standard_frame_for(cfg, cfg=OmegaConf.create({}))

        assert frame is not None and len(frame) == 2

    def test_an_unregistered_type_has_no_frame(self):
        # `simulated` generates its own observations from nothing at all, so
        # "no reader" is a normal answer rather than an error.
        cfg = OmegaConf.create({"type": "simulated", "identifier": "simulated_example"})

        assert standard_frame_for(cfg) is None

    def test_data_source_field_is_honoured(self, merged_table):
        # A config that records its provider rather than a type still dispatches,
        # so a preprocessed config needs no 'type' field.
        cfg = OmegaConf.create(
            {
                "data_source": "voyager",
                "identifier": "voyager_triton",
                "data": {"file": "data.csv"},
                "ancillary_files": {"file": "ancillary.csv"},
                "observatory": {"code": -31, "name": "Voyager 2"},
                "satellites": {"Triton": {}},
            }
        )

        assert standard_frame_for(cfg, cfg=OmegaConf.create({})) is not None

    def test_registering_a_taken_name_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            register_frame_source("voyager")

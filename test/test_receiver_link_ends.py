"""Tests for how a standard frame's receiver link end is chosen.

The receiver is the one link end whose *kind* varies between datasets: a ground
station is a reference point on Earth, a spacecraft is a body origin, and a
geocentric observer is Earth's centre.  Getting this wrong is not a cosmetic
error - it either sends a body name into the MPC catalogue lookup (raising) or
silently omits the ground station from the environment.

This module guards a regression where the frame-centre label was parsed with the
bare enum constructor instead of the vocabulary normaliser, so NSDB's British
spelling ``topocentre`` - which appears in 77 of the 82 committed configs and is
*not* an enum member - raised ``ValueError`` for every relative dataset.
"""

import pytest
from omegaconf import OmegaConf

from orbitdet.data.nsdb_formatter import normalize_frame_center
from orbitdet.observations.standard_frames import (
    build_link_ends,
    frame_center_of,
    receiver_needs_observatory,
)
from orbitdet.preprocessing.schema import FrameCenter, build_frame

TUDAT_REQUIRED = pytest.importorskip("tudatpy.estimation.observable_models_setup")


def relative_config(**overrides):
    """A relative NSDB dataset config, as the generated configs look."""
    base = {
        "identifier": "nm0021",
        "type": "relative_position_angle_and_separation_micrometric_nsdb",
        "centre_of_frame": "topocentre",
        "satellites": {"Triton": {"designation": "N1-Triton"}},
        "relative_to": {"designation": "planet"},
        "planet": {"name": "Neptune"},
    }
    base.update(overrides)
    return OmegaConf.create(base)


class TestBritishSpelling:
    """The regression: NSDB writes ``topocentre``, which is not an enum member."""

    def test_topocentre_builds_a_station_link_end(self):
        link_ends = build_link_ends(relative_config(), "787", relative_to="Neptune")

        receiver = link_ends[TUDAT_REQUIRED.links.receiver]
        assert receiver.reference_point == "787"

    def test_topocentre_is_not_an_enum_member(self):
        # The premise of the regression: normalising through the vocabulary is
        # the *only* thing that can resolve this label, so a bare
        # FrameCenter("topocentre") cannot work.
        with pytest.raises(ValueError):
            FrameCenter("topocentre")

    def test_every_corpus_spelling_resolves(self):
        # These are the frame-centre labels actually present in the committed
        # configs, typos included.
        assert normalize_frame_center("topocentre") is FrameCenter.TOPOCENTRIC
        assert normalize_frame_center("geocentre") is FrameCenter.GEOCENTRIC
        assert normalize_frame_center("geocenter") is FrameCenter.GEOCENTRIC
        assert normalize_frame_center("geocentric") is FrameCenter.GEOCENTRIC

    def test_an_absolute_config_with_topocentre_builds(self):
        link_ends = build_link_ends(relative_config(centre_of_frame="topocentre"), "689")

        assert link_ends[TUDAT_REQUIRED.links.receiver].reference_point == "689"

    @pytest.mark.parametrize("spelling", ["topocentre", "topocentric", "Topocentre"])
    def test_the_british_spelling_is_case_insensitive(self, spelling):
        link_ends = build_link_ends(relative_config(centre_of_frame=spelling), "787")

        assert link_ends[TUDAT_REQUIRED.links.receiver].reference_point == "787"


class TestReceiverKind:
    def test_a_spacecraft_receiver_is_a_body_origin(self):
        cfg = relative_config(centre_of_frame="spacecraft")

        link_ends = build_link_ends(cfg, "Voyager 2", relative_to="Neptune")

        receiver = link_ends[TUDAT_REQUIRED.links.receiver]
        assert receiver.body_name == "Voyager 2"
        # An empty reference point is what makes it a body origin rather than a
        # station; getting this wrong is what sent "Voyager 2" to the MPC
        # catalogue lookup.
        assert receiver.reference_point == ""

    def test_a_geocentric_receiver_is_earths_centre(self):
        cfg = relative_config(centre_of_frame="geocentric")

        link_ends = build_link_ends(cfg, "119", relative_to="Neptune")

        receiver = link_ends[TUDAT_REQUIRED.links.receiver]
        assert receiver.body_name == "Earth"
        assert receiver.reference_point == ""

    def test_an_explicit_frame_center_argument_wins(self):
        # The factories pass the frame's own column, which is authoritative:
        # parsing the config again would let the two disagree.
        cfg = relative_config(centre_of_frame="spacecraft")

        link_ends = build_link_ends(
            cfg, "787", relative_to="Neptune", frame_center=FrameCenter.TOPOCENTRIC
        )

        assert link_ends[TUDAT_REQUIRED.links.receiver].reference_point == "787"


class TestObservatoryPredicate:
    """``receiver_needs_observatory`` must agree with the link-end branch."""

    def test_a_ground_station_needs_the_catalogue(self):
        assert receiver_needs_observatory(FrameCenter.TOPOCENTRIC) is True

    def test_a_spacecraft_does_not(self):
        assert receiver_needs_observatory(FrameCenter.SPACECRAFT) is False

    def test_a_geocentric_observer_does_not(self):
        assert receiver_needs_observatory(FrameCenter.GEOCENTRIC) is False

    def test_an_unrecognised_centre_is_treated_as_a_station(self):
        # This must match build_link_ends, which falls through to the
        # ground-station branch for anything it cannot place.  If the predicate
        # disagreed, the station would be skipped while the link end still
        # referenced it, and the lookup would fail.
        assert receiver_needs_observatory(FrameCenter.UNKNOWN) is True

        link_ends = build_link_ends(relative_config(centre_of_frame="geoocentre"), "689")

        assert link_ends[TUDAT_REQUIRED.links.receiver].reference_point == "689"

    def test_the_predicate_matches_the_link_end_for_every_member(self):
        # The invariant, stated once: the predicate and the link end are two
        # views of the same decision, so they must not drift apart.
        for member in FrameCenter:
            link_ends = build_link_ends(
                relative_config(), "689", relative_to="Neptune", frame_center=member
            )
            receiver = link_ends[TUDAT_REQUIRED.links.receiver]
            is_station = receiver.reference_point == "689"
            assert is_station is receiver_needs_observatory(member), member


class TestFrameCenterOf:
    def _frame(self, center: str):
        return build_frame(
            [
                {
                    "observed_object": "Triton",
                    "observatory_code": "689",
                    "data_source": "nsdb",
                    "source_set_id": "nm0001",
                    "reference": "ref",
                    "observation_type": "absolute",
                    "coordinate_kind": "ra_dec",
                    "frame_center": center,
                    "original_equinox": "J2000",
                    "original_reference_frame": "astrometric",
                    "original_reduction": "none",
                    "receptor_type": "ccd",
                    "epoch_tdb": 1.0,
                    "first_value": 1.0,
                    "second_value": 0.1,
                }
            ]
        )

    def test_reads_the_frames_single_centre(self):
        assert frame_center_of(self._frame("topocentric")) is FrameCenter.TOPOCENTRIC

    def test_a_mixed_frame_is_rejected(self):
        # Two centres in one frame means two kinds of receiver, which needs
        # splitting before either can be built.
        mixed = build_frame(
            [
                {
                    "observed_object": "Triton",
                    "observatory_code": code,
                    "data_source": "nsdb",
                    "source_set_id": "nm0001",
                    "reference": "ref",
                    "observation_type": "absolute",
                    "coordinate_kind": "ra_dec",
                    "frame_center": center,
                    "original_equinox": "J2000",
                    "original_reference_frame": "astrometric",
                    "original_reduction": "none",
                    "receptor_type": "ccd",
                    "epoch_tdb": 1.0,
                    "first_value": 1.0,
                    "second_value": 0.1,
                }
                for code, center in (("689", "topocentric"), ("Gaia", "spacecraft"))
            ]
        )

        with pytest.raises(ValueError, match="one frame centre"):
            frame_center_of(mixed)

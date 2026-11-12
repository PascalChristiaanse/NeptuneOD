"""Tests for the refactored ``ResidualScatter`` plot (access + standard frame)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tudatpy.estimation.observable_models_setup as settings  # noqa: E402
import tudatpy.estimation.observations as obs  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from tudatpy.estimation.observable_models_setup import links  # noqa: E402

from orbitdet.analysis import ObservationAccess  # noqa: E402
from orbitdet.preprocessing.schema import OBSERVATION_SET_INDEX_COLUMN, build_frame  # noqa: E402
from orbitdet.visualization import ResidualScatter  # noqa: E402

ARCSEC = np.rad2deg(1.0) * 3600.0


def _record(epoch, station="673"):
    return {
        "observed_object": "Triton",
        "observatory_code": station,
        "data_source": "nsdb",
        "source_set_id": "nm0001",
        "reference": "ref",
        "observation_type": "absolute",
        "coordinate_kind": "ra_dec",
        "frame_center": "topocentric",
        "original_equinox": "J2000",
        "original_reference_frame": "astrometric",
        "original_reduction": "none",
        "receptor_type": "ccd",
        "epoch_tdb": epoch,
        "first_value": 2.0,
        "second_value": -0.3,
        "scan_angle": np.nan,
    }


def _set(epochs, residuals, reference_point):
    link_ends = {
        links.transmitter: links.body_origin_link_end_id("Triton"),
        links.receiver: links.body_reference_point_link_end_id("Earth", reference_point),
    }
    single_set = obs.create_single_observation_set(
        settings.model_settings.ObservableType.angular_position_type,
        link_ends,
        [np.array([[0.5], [0.25]])] * len(epochs),
        list(epochs),
        links.LinkEndType.receiver,
    )
    single_set.set_residuals(np.asarray(residuals, dtype=float).ravel())
    single_set.set_tabulated_weights(np.ones(2 * len(epochs)))
    return single_set


def _access():
    """Two observatories, three observations, distinct residuals."""
    set_a = _set([1.0, 2.0], np.array([[1e-6, 2e-6], [3e-6, 4e-6]]), "673")
    set_b = _set([3.0], np.array([[5e-6, 6e-6]]), "689")
    frame = build_frame([_record(1.0), _record(2.0), _record(3.0, station="689")])
    frame[OBSERVATION_SET_INDEX_COLUMN] = [0.0, 0.0, 1.0]
    return ObservationAccess.build(obs.ObservationCollection([set_a, set_b]), frame)


def _cfg(**scatter_cfg):
    return OmegaConf.create(scatter_cfg)


class TestResidualScatter:
    def test_plots_ra_against_dec(self):
        access = _access()
        plot = ResidualScatter(_cfg(), access)
        fig, ax = plot._make_figure()
        offsets = ax.collections[0].get_offsets()
        # First cluster (673): RA = 1e-6, 3e-6 rad -> arcsec.
        assert offsets[0, 0] == pytest.approx(1e-6 * ARCSEC, rel=1e-6)
        assert offsets[0, 1] == pytest.approx(2e-6 * ARCSEC, rel=1e-6)
        plt.close(fig)

    def test_groups_by_observatory_default(self):
        access = _access()
        plot = ResidualScatter(_cfg(), access)
        fig, ax = plot._make_figure()
        # Two observatories -> two scatter series.
        assert len(ax.collections) == 2
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "673" in labels and "689" in labels
        plt.close(fig)

    def test_group_by_argument_overrides_default(self):
        access = _access()
        plot = ResidualScatter(_cfg(group_by="source_set_id"), access)
        fig, ax = plot._make_figure()
        # Both sets share source_set_id "nm0001" -> a single series.
        assert len(ax.collections) == 1
        plt.close(fig)

    def test_group_by_from_config(self):
        access = _access()
        plot = ResidualScatter(_cfg(group_by="observatory_code"), access)
        fig, ax = plot._make_figure()
        assert len(ax.collections) == 2
        plt.close(fig)

    def test_mas_unit(self):
        access = _access()
        plot = ResidualScatter(_cfg(axes={"unit": "mas"}), access)
        fig, ax = plot._make_figure()
        offsets = ax.collections[0].get_offsets()
        # 1e-6 rad in mas = 1e-6 * ARCSEC * 1000.
        assert offsets[0, 0] == pytest.approx(1e-6 * ARCSEC * 1000.0, rel=1e-6)
        plt.close(fig)

    def test_colourbar_present(self):
        access = _access()
        plot = ResidualScatter(_cfg(), access)
        fig, ax = plot._make_figure()
        assert len(fig.axes) == 2  # main axes + colourbar
        plt.close(fig)

    def test_rejected_rows_excluded_by_default(self):
        access = _access()
        access.frame.loc[access.frame.index[0], "rejected"] = True
        plot = ResidualScatter(_cfg(), access)
        fig, ax = plot._make_figure()
        offsets = ax.collections[0].get_offsets()
        # The rejected first row is gone; the remaining 673 row is 3e-6.
        assert offsets[0, 0] == pytest.approx(3e-6 * ARCSEC, rel=1e-6)
        plt.close(fig)

    def test_include_rejected_keeps_rows(self):
        access = _access()
        access.frame.loc[access.frame.index[0], "rejected"] = True
        plot = ResidualScatter(_cfg(include_rejected=True), access)
        fig, ax = plot._make_figure()
        offsets = ax.collections[0].get_offsets()
        assert offsets[0, 0] == pytest.approx(1e-6 * ARCSEC, rel=1e-6)
        plt.close(fig)

    def test_unknown_grouping_column_raises(self):
        access = _access()
        plot = ResidualScatter(_cfg(group_by="not_a_column"), access)
        with pytest.raises(ValueError, match="not in the standard frame"):
            plot._make_figure()

    def test_builds_under_journal_style(self):
        """The journal style installs a constrained layout engine.

        That engine builds the colourbar through its gridspec, so a plotter that
        then forces tight layout raises ``RuntimeError``.  Building the figure
        with the style applied guards against that regression.
        """
        from orbitdet.visualization.style import apply_style

        saved = dict(matplotlib.rcParams)
        try:
            apply_style("A&A")
            access = _access()
            plot = ResidualScatter(_cfg(), access)
            fig, ax = plot._make_figure()
            assert len(fig.axes) == 2  # main axes + colourbar
            assert type(fig.get_layout_engine()).__name__ == "ConstrainedLayoutEngine"
            plt.close(fig)
        finally:
            matplotlib.rcParams.update(saved)

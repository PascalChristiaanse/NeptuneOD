"""Tests for the Gaia transit random/systematic error decomposition."""

import numpy as np

from orbitdet.visualization.gaia_transit_errors import (
    _transit_labels,
    compute_transit_error_components,
)


def test_transit_labels_splits_on_gap():
    # Three clusters: [0, 10], [1000, 1010, 1020], [100000].
    epochs = np.array([0.0, 10.0, 1000.0, 1010.0, 1020.0, 100000.0])
    labels = _transit_labels(epochs, gap_seconds=100.0)
    assert labels.tolist() == [0, 0, 1, 1, 1, 2]


def test_transit_labels_rejects_unsorted():
    epochs = np.array([10.0, 0.0])
    try:
        _transit_labels(epochs, gap_seconds=1.0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unsorted epochs")


def test_random_and_systematic_are_separable():
    # Two transits, each with two observations.  Within-transit offsets are the
    # only scatter inside a transit; the transit means differ by 10 (systematic).
    epochs = np.array([0.0, 10.0, 3600.0, 3610.0])
    al = np.array([-1.0, 1.0, 9.0, 11.0])
    ac = np.array([-2.0, 2.0, -2.0, 2.0])
    components = compute_transit_error_components(epochs, al, ac)

    assert components.transit_id.tolist() == [0, 0, 1, 1]
    assert components.transit_count.tolist() == [2, 2]

    # AL transit means are 0 and 10 -> systematic (sample std, ddof=1) = sqrt(50).
    np.testing.assert_allclose(components.transit_mean_al, [0.0, 10.0])
    np.testing.assert_allclose(components.systematic_sigma_al, np.sqrt(50.0))

    # Within-transit AL deviations are +-1 in each (sum of squares 4 over
    # N - M = 4 - 2 = 2 dof) -> pooled sigma = sqrt(2).
    np.testing.assert_allclose(components.random_sigma_al, np.sqrt(2.0))

    # AC within-transit scatter is +-2 in each -> pooled sigma = 2*sqrt(2).
    np.testing.assert_allclose(components.transit_mean_ac, [0.0, 0.0])
    np.testing.assert_allclose(components.systematic_sigma_ac, 0.0)
    np.testing.assert_allclose(components.random_sigma_ac, 2.0 * np.sqrt(2.0))


def test_pooled_random_matches_formula():
    # Three transits with different sizes; pooled variance uses N - M dof.
    epochs = np.array([0.0, 10.0, 20.0, 3600.0, 3610.0, 100000.0])
    al = np.array([1.0, -1.0, 2.0, -2.0, 0.0, 5.0])
    ac = np.zeros_like(al)
    components = compute_transit_error_components(epochs, al, ac)

    transit_id = components.transit_id
    within = al - components.transit_mean_al[transit_id]
    expected = np.sqrt(np.sum(within**2) / (al.size - components.transit_epoch.size))
    np.testing.assert_allclose(components.random_sigma_al, expected)


def test_single_transit_has_nan_systematic():
    epochs = np.array([0.0, 10.0, 20.0])
    al = np.array([1.0, -1.0, 0.0])
    ac = np.array([2.0, -2.0, 0.0])
    components = compute_transit_error_components(epochs, al, ac)
    assert np.isnan(components.systematic_sigma_al)
    assert np.isnan(components.systematic_sigma_ac)
    # Random is still defined (dof = 3 - 1 = 2).
    assert np.isfinite(components.random_sigma_al)


def test_singleton_transits_have_nan_random():
    # Four transits of one observation each: no within-transit scatter.
    epochs = np.array([0.0, 3600.0, 7200.0, 10800.0])
    al = np.array([1.0, 2.0, 3.0, 4.0])
    ac = np.array([1.0, 2.0, 3.0, 4.0])
    components = compute_transit_error_components(epochs, al, ac)
    assert np.isnan(components.random_sigma_al)
    assert components.transit_count.tolist() == [1, 1, 1, 1]
    # Systematic is the spread of those means.
    np.testing.assert_allclose(components.systematic_sigma_al, np.std(al, ddof=1))


def test_nan_observations_dropped_and_sorted():
    epochs = np.array([3600.0, np.nan, 0.0, 10.0])
    al = np.array([9.0, 1.0, -1.0, 1.0])
    ac = np.array([9.0, 1.0, -1.0, 1.0])
    components = compute_transit_error_components(epochs, al, ac)
    # NaN epoch dropped; the rest sorted ascending.
    np.testing.assert_allclose(components.epochs, [0.0, 10.0, 3600.0])
    assert components.transit_id.tolist() == [0, 0, 1]


def test_empty_input():
    components = compute_transit_error_components(
        np.array([]), np.array([]), np.array([])
    )
    assert components.epochs.size == 0
    assert np.isnan(components.random_sigma_al)
    assert np.isnan(components.systematic_sigma_al)


def test_length_mismatch_raises():
    try:
        compute_transit_error_components(np.array([0.0]), np.array([1.0]), np.array([1.0, 2.0]))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for length mismatch")

"""Distributions and spectral estimates for the analysis layer.

The residual figures need a little more than reductions: a Q-Q plot against a
normal, a density histogram with Poisson error bars, a normality test, and a
power spectral density (Welch for evenly sampled series, Lomb-Scargle for the
bursty Gaia cadence).  These all live here so the plot modules stop each
reimplementing them and so they are unit-testable without a figure.

Everything is pure - arrays in, arrays/values out - and imports SciPy only where
a SciPy routine is genuinely needed (the normal quantile function, the Shapiro
test, ``welch``/``lombscargle``, ``find_peaks``).  Nothing reads config or the
file system.

Conventions
-----------
* Residuals are in whatever unit the caller passes; the functions do not convert
  units (a Q-Q plot is scale-free anyway once standardized).
* A PSD takes **times in days** and returns a frequency in cycles per day, the
  unit the Triton figures use.
"""

from __future__ import annotations

import numpy as np
from scipy import signal, stats

__all__ = [
    "standardize",
    "qq_points",
    "density_histogram",
    "normality_test",
    "welch_psd",
    "lombscargle_psd",
    "find_psd_peaks",
    "DEFAULT_LOMB_FREQS_PER_DECADE",
]

#: Frequency samples per decade for the log-spaced Lomb-Scargle grid.
DEFAULT_LOMB_FREQS_PER_DECADE: int = 50


def standardize(values) -> np.ndarray:
    """Return the finite values standardized to zero mean and unit sample std.

    ``(x − mean) / std(ddof=1)``, which is the transformation a Q-Q plot against
    a standard normal applies to the sample.

    Parameters
    ----------
    values : array-like
        The sample.

    Returns
    -------
    np.ndarray
        The finite values standardized, in their original order.
    """
    values = np.asarray(values, dtype="float64").ravel()
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return np.array([], dtype="float64")
    return (finite - np.mean(finite)) / np.std(finite, ddof=1)


def qq_points(values) -> tuple[np.ndarray, np.ndarray]:
    """Return theoretical and sample quantiles for a normal Q-Q plot.

    The sample is standardized (:func:`standardize`) and sorted; the theoretical
    quantiles are ``norm.ppf((i − 0.5) / n)`` (the Blom/Weibull plotting
    positions the figure uses).  A sample that is normal lies on the ``y = x``
    line.

    Parameters
    ----------
    values : array-like
        The sample (any unit - it is standardized).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(theoretical, sample)`` of equal length (both empty when there are
        fewer than two finite values).
    """
    standardized = np.sort(standardize(values))
    n = standardized.size
    if n == 0:
        return np.array([], dtype="float64"), np.array([], dtype="float64")
    theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    return theoretical, standardized


def density_histogram(
    values,
    bins: int | np.ndarray = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Density histogram with Poisson error bars.

    The bin heights are probability densities (``Σ density·width = 1``); the
    error on a density bin is the Poisson error on its count, propagated through
    ``count = density · N · width``::

        err = sqrt(count) / (N · width)

    Parameters
    ----------
    values : array-like
        The sample.
    bins : int | np.ndarray
        Number of bins, or explicit bin edges.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(density, bin_edges, density_error, bin_centers)``.  All empty when
        there is no finite value.
    """
    values = np.asarray(values, dtype="float64").ravel()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        empty = np.array([], dtype="float64")
        return empty, empty, empty, empty

    density, edges = np.histogram(finite, bins=bins, density=True)
    widths = np.diff(edges)
    counts = density * finite.size * widths
    with np.errstate(divide="ignore", invalid="ignore"):
        density_error = np.sqrt(counts) / (finite.size * widths)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return density, edges, density_error, centers


def normality_test(values) -> tuple[float, float]:
    """Shapiro-Wilk normality test; ``(statistic, p_value)``.

    Parameters
    ----------
    values : array-like
        The sample.

    Returns
    -------
    tuple[float, float]
        The Shapiro-Wilk ``W`` statistic and its p-value, or ``(NaN, NaN)`` when
        fewer than three finite values are available (the test's minimum).
    """
    finite = np.asarray(values, dtype="float64").ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size < 3:
        return float("nan"), float("nan")
    statistic, p_value = stats.shapiro(finite)
    return float(statistic), float(p_value)


def welch_psd(
    times,
    values,
    *,
    window_length: float,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density of an evenly resampled series.

    The (irregular) samples are first resampled onto a uniform grid at the median
    spacing (:func:`_resample_uniform`), mean-centred, and passed to
    :func:`scipy.signal.welch` with constant detrending and density scaling.

    Parameters
    ----------
    times : array-like
        Sample times in **days**.
    values : array-like
        Sample values (any unit; the PSD is in that unit squared per cycles/day).
    window_length : float
        Welch segment length in days.
    window : str
        A SciPy window name (``"hann"``, ``"hamming"``, ``"boxcar"``, ...).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(frequencies, power_density)``.

    Raises
    ------
    ValueError
        If there are fewer than two usable samples, or the sampling spacing is
        not positive, or the window name is not a SciPy window.
    """
    _, uniform_values, spacing = _resample_uniform(times, values)
    centered = uniform_values - np.mean(uniform_values)
    n_samples = centered.size

    segment = int(round(window_length / spacing))
    segment = max(8, segment)
    segment = min(segment, n_samples)
    noverlap = segment // 2

    try:
        frequencies, power_density = signal.welch(
            centered,
            fs=1.0 / spacing,
            window=window,
            nperseg=segment,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
            return_onesided=True,
        )
    except ValueError as exc:
        raise ValueError(
            f"Unsupported window type '{window}' for Welch PSD. Use a SciPy-compatible "
            "window name (e.g. hamming, hann, boxcar, blackman)."
        ) from exc
    return frequencies, power_density


def lombscargle_psd(
    times,
    values,
    *,
    frequencies_per_decade: int = DEFAULT_LOMB_FREQS_PER_DECADE,
) -> tuple[np.ndarray, np.ndarray]:
    """Lomb-Scargle periodogram of an irregularly sampled series.

    Unlike the Welch estimate this does not require uniform resampling, so it is
    the right choice for a bursty cadence such as Gaia's.  The frequency grid
    runs from one cycle over the full span up to ``min(1, Nyquist)`` cycles per
    day (so periods stay at or above one day), log-spaced to resolve peaks.

    Parameters
    ----------
    times : array-like
        Sample times in **days**.
    values : array-like
        Sample values.
    frequencies_per_decade : int
        Log-spaced frequency samples per decade.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(frequencies, normalized_power)``.

    Raises
    ------
    ValueError
        If there are fewer than two usable samples or the time span is not
        positive.
    """
    times = np.asarray(times, dtype="float64").ravel()
    values = np.asarray(values, dtype="float64").ravel()
    finite = np.isfinite(times) & np.isfinite(values)
    times, values = times[finite], values[finite]
    if times.size < 2:
        raise ValueError("Need at least two finite residual samples to compute a PSD.")

    order = np.argsort(times)
    times, values = times[order], values[order]
    centered = values - np.mean(values)

    span = float(times[-1] - times[0])
    if span <= 0.0:
        raise ValueError("Could not infer a valid sampling interval for the residual PSD.")

    f_min = 1.0 / span
    f_max = min(1.0, 0.5 / float(np.median(np.diff(times))))
    if not np.isfinite(f_max) or f_max <= f_min:
        f_max = f_min * 10.0

    n_frequencies = frequencies_per_decade * int(np.log10(f_max / f_min)) + 1
    frequencies = np.geomspace(f_min, f_max, num=n_frequencies)
    # SciPy's ``lombscargle`` wants *angular* frequency (radians per unit time),
    # while the grid and the returned frequencies are in cycles/day.
    power = signal.lombscargle(times, centered, 2.0 * np.pi * frequencies, normalize=True)
    return frequencies, power


def find_psd_peaks(
    frequencies,
    power,
    *,
    prominence: float = 0.1,
) -> list[tuple[float, float]]:
    """Return ``(frequency, power)`` of the most prominent positive-frequency peaks.

    Parameters
    ----------
    frequencies : array-like
        Frequencies (cycles/day).
    power : array-like
        Power density.
    prominence : float
        Minimum peak prominence passed to :func:`scipy.signal.find_peaks`.

    Returns
    -------
    list[tuple[float, float]]
        Peaks sorted by descending power (empty when none are found).
    """
    frequencies = np.asarray(frequencies, dtype="float64").ravel()
    power = np.asarray(power, dtype="float64").ravel()
    positive = frequencies > 0.0
    frequencies, power = frequencies[positive], power[positive]
    if frequencies.size == 0:
        return []

    peaks, _ = signal.find_peaks(power, prominence=prominence)
    if peaks.size == 0:
        return []

    peak_power = power[peaks]
    order = np.argsort(peak_power)[::-1]
    return [(float(frequencies[peaks[i]]), float(peak_power[i])) for i in order]


def _resample_uniform(times, values) -> tuple[np.ndarray, np.ndarray, float]:
    """Resample an irregular series onto a uniform grid at the median spacing.

    Parameters
    ----------
    times, values : array-like
        Sample times (days) and values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        ``(uniform_times, uniform_values, spacing)`` with ``spacing`` in days.

    Raises
    ------
    ValueError
        If there are fewer than two usable or unique times, or the spacing is not
        positive.
    """
    times = np.asarray(times, dtype="float64").ravel()
    values = np.asarray(values, dtype="float64").ravel()
    finite = np.isfinite(times) & np.isfinite(values)
    times, values = times[finite], values[finite]
    if times.size < 2:
        raise ValueError("Need at least two finite residual samples to compute a PSD.")

    order = np.argsort(times)
    times, values = times[order], values[order]
    unique_times, unique_indices = np.unique(times, return_index=True)
    values = values[unique_indices]
    if unique_times.size < 2:
        raise ValueError("Need at least two unique observation times to compute a PSD.")

    spacing = float(np.median(np.diff(unique_times)))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Could not infer a valid sampling interval for the residual PSD.")

    uniform_times = np.arange(
        unique_times[0], unique_times[-1] + spacing / 2.0, spacing
    )
    uniform_values = np.interp(uniform_times, unique_times, values)
    return uniform_times, uniform_values, spacing

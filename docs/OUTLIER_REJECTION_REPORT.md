# Outlier Rejection in the Neptune–Triton Orbit Determination Repository

**Scope:** All mechanisms used to detect and remove outlier observations across the codebase.
**Purpose:** Reference document for the thesis describing how bad observations are identified and excluded from the Triton orbit-determination (OD) pipeline.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Core Concepts](#2-core-concepts)
3. [Primary Mechanism: Residual Filtering](#3-primary-mechanism-residual-filtering)
4. [Epoch-Based Filtering](#4-epoch-based-filtering)
5. [Rejected-Epoch Tracking](#5-rejected-epoch-tracking)
6. [Legacy Mechanism: SPICE Residual Outlier Removal](#6-legacy-mechanism-spice-residual-outlier-removal)
7. [File-Level Filtering (Whole-File Exclusion)](#7-file-level-filtering-whole-file-exclusion)
8. [Diagnostics & Visualization](#8-diagnostics--visualization)
9. [Settings & Configuration](#9-settings--configuration)
10. [Summary of Distinct Rejection Strategies](#10-summary-of-distinct-rejection-strategies)
11. [File Map](#11-file-map)
12. [Notes & Caveats](#12-notes--caveats)

---

## 1. Overview

Outlier rejection in this repository operates at **three distinct levels**:

| Level | Granularity | Mechanism | Location |
|---|---|---|---|
| **File-level** | Whole observation file | Mean/std thresholds per file | `ObservationsTestFileIgnore.py` |
| **Observation-level (primary)** | Individual observations | Residual threshold (1.5 arcsec) | `ObsFunc.LoadObservations()` |
| **Observation-level (legacy)** | Individual observations | SPICE-residual threshold (25) | `nsdc.remove_nsdc_outliers()` |

The **primary** and most important mechanism is the **residual-based filtering** applied inside `ObsFunc.LoadObservations()`, which uses Tudat's built-in `observation_filter` machinery. This is the mechanism active in the real-observation OD pipeline.

---

## 2. Core Concepts

### 2.1 Residual

The residual is the difference between the **observed** angular position (RA/DEC) and the **computed** position from the dynamical model:

$$r = \text{observed} - \text{computed}$$

Residuals are computed by Tudat via `compute_residuals_and_dependent_variables()` against the current reference trajectory.

### 2.2 Filtering vs. Weighting

- **Filtering (outlier rejection)** — *removes* observations entirely from the estimation.
- **Weighting** — *keeps* observations but down-weights the noisy ones.

These are complementary: filtering removes grossly bad points, while weighting handles moderate noise. In this pipeline, filtering happens **before** weighting.

### 2.3 Observation set / file

Observations are grouped into **sets** identified by `set_id` (e.g. `689_nm0077`). Filtering is applied **per set** (per file).

---

## 3. Primary Mechanism: Residual Filtering

**Location:** `HelperFunctions/ObsFunc.py` → `LoadObservations()`

This is the canonical, active outlier-rejection mechanism in the OD pipeline.

### 3.1 Threshold

```python
arcsec_to_rad = np.pi / (180.0 * 3600.0)
upper_bound = 1.5            # arcseconds
upper_bound_rad = upper_bound * arcsec_to_rad
```

The threshold is **1.5 arcseconds**, converted to radians. Any observation whose residual magnitude exceeds this is considered an outlier.

### 3.2 Filter construction

```python
outlier_filter = observations_processing.observation_filter(
    observations_processing.ObservationFilterType.residual_filtering,
    upper_bound_rad)

opposite_outlier_filter = observations_processing.observation_filter(
    observations_processing.ObservationFilterType.residual_filtering,
    upper_bound_rad, use_opposite_condition=True)
```

- `outlier_filter` — keeps observations with residual **below** the threshold (the accepted set).
- `opposite_outlier_filter` — keeps observations with residual **above** the threshold (the rejected set), used purely for diagnostics/counting.

### 3.3 Application

```python
observation_single_set_current_filtered = estimation.observations.create_filtered_observation_set(
    observation_single_set_current, outlier_filter)
```

`create_filtered_observation_set()` returns a new observation set containing only the accepted observations.

### 3.4 Flow

For each file (set):

1. Compute residuals against the reference trajectory.
2. Build the residual filter (1.5 arcsec).
3. Apply the filter → accepted set.
4. Apply the opposite filter → rejected set (for counting).
5. Track rejected epochs (see §5).
6. Append the **filtered** set to the observation collection.

### 3.5 Gating flag

The filtering block is gated by the `FilterObservations` global flag. When `FilterObservations == True`, filtering is applied; otherwise the unfiltered set is used.

---

## 4. Epoch-Based Filtering

A second, **independent** filter can be applied based on **exact epochs** rather than residuals.

**Location:** `ObsFunc.LoadObservations()`

```python
if epoch_filter_dict is not None and set_id in epoch_filter_dict:
    if np.shape(epoch_filter_dict[set_id])[0] != 0:
        epoch_filter = observations_processing.observation_filter(
            observations_processing.ObservationFilterType.epochs_filtering,
            epoch_filter_dict[set_id])
        observation_single_set_current_filtered = estimation.observations.create_filtered_observation_set(
            observation_single_set_current_filtered, epoch_filter)
```

- `epoch_filter_dict` maps `set_id` → list of exact epochs to remove.
- Uses `ObservationFilterType.epochs_filtering`.
- Applied **independently** of (and after) the residual filter.

This allows manual removal of specific bad epochs identified by inspection, complementing the automatic residual filter.

---

## 5. Rejected-Epoch Tracking

The rejected epochs are **always tracked**, regardless of which filters are active, for diagnostics and downstream analysis.

**Location:** `ObsFunc.LoadObservations()`

```python
epochs_all = observation_single_set_current.observation_times
residual_filtered_set = estimation.observations.create_filtered_observation_set(
    observation_single_set_current, outlier_filter)
epochs_filtered = residual_filtered_set.observation_times
epochs_rejected_current = [t for t in epochs_all if t not in epochs_filtered]
epochs_rejected[set_id] = [t.to_float() for t in epochs_rejected_current]
```

The `epochs_rejected` dict (keyed by `set_id`) is returned from `LoadObservations()` and can be:

- **Saved to JSON** — `ObservationImplementation.py` writes it to `residuals_rejected_epochs.json`.
- **Used for plotting** — `Test_Observations.py` marks rejected points with `x` markers.

---

## 6. Legacy Mechanism: SPICE Residual Outlier Removal

**Location:** `HelperFunctions/nsdc.py` → `remove_nsdc_outliers()`

This is the **older** mechanism used during the raw NSDC data-processing stage (before observations are converted to CSV).

### 6.1 Approach

1. Compute residuals between the **observed** positions and the **SPICE reference ephemeris** (`analyse_nsdc_data`).
2. Flag an observation as an outlier if **either** RA or DEC residual exceeds a threshold.

### 6.2 Threshold

```python
outlier_indices = [i for i, nested_list in enumerate(diflist)
                   if any(abs(x) > 25 for x, mean, std_dev in zip(nested_list, means, stddevs))]
```

- **Hardcoded threshold: 25** (in the residual units, effectively arcseconds).
- The `outlier_limit` parameter (default 3) is **not actually used** in the active code.
- A commented-out alternative uses a **2.5σ** criterion:
  ```python
  # if any(abs(x - mean) > 2.5 * std for x, mean, std in zip(nested_list, means, stds))
  ```

### 6.3 Output

Returns filtered lists (`times`, `observations`, `moons`, `observatories`, `diflist`) plus recomputed `rms`, `means`, `stddevs`.

### 6.4 Caller

Called from `process_nsdc_file()` (with `outlier_limit=3`), which is invoked by `NSDC_processing.py` during raw-data processing. This stage produces the cleaned CSV files that feed the main OD pipeline.

---

## 7. File-Level Filtering (Whole-File Exclusion)

**Location:** `ObservationsTestFileIgnore.py` → `filter_by_mean_std()`

This is a **coarse, file-level** filter applied during data preparation. It excludes **entire observation files** whose aggregate statistics are poor.

### 7.1 Criteria

```python
too_mean = (df2["mean_RA"].abs() > max_mean_ra) | (df2["mean_DEC"].abs() > max_mean_dec)
too_std  = (df2["std_RA"] > max_std_ra) | (df2["std_DEC"] > max_std_dec)
exclude  = too_mean | too_std
```

A file is excluded if **either** its mean RA/DEC residual exceeds `max_mean` **or** its std RA/DEC exceeds `max_std`.

### 7.2 Defaults vs. usage

| Parameter | Default | Used in main |
|---|---|---|
| `max_mean_ra` | 0.1 | 2 |
| `max_mean_dec` | 0.1 | 2 |
| `max_std_ra` | 0.5 | 2 |
| `max_std_dec` | 0.5 | 2 |

(Units are arcseconds.)

### 7.3 Output

Returns `(filtered_df, removed_df)` for auditing, with a `remove_reason` column explaining why each file was excluded.

### 7.4 Downstream effect

The filtered file list is written to `file_names.json` / `observation_set_ids.json`, which **controls which files are loaded** into the OD pipeline. Thus this is a **pre-processing** rejection step.

---

## 8. Diagnostics & Visualization

### 8.1 `Test_Observations.py`

A dedicated investigation script that:

- Loads all files **unfiltered** and **filtered**.
- Computes residuals from both **SPICE (nep097)** and **tudatpy**.
- Plots, per file, RA and DEC residuals with **accepted** points as circles and **rejected** points as `x` markers.
- Saves to `observation_residuals_all_files.pdf`.

This is used to visually validate that the residual filter is rejecting the right observations.

### 8.2 `ObservationImplementation.py`

Saves the rejected epochs to `residuals_rejected_epochs.json` for every run:

```python
output_file = out_dir / "residuals_rejected_epochs.json"
with open(output_file, 'w') as f:
    json.dump(epochs_rejected, f, indent=2)
```

### 8.3 `MainPostprocessing.py`

Contains commented-out code comparing estimation results **with vs. without** outliers (RSW differences), indicating an earlier analysis of outlier impact.

### 8.4 `ObservationsTestFileIgnore.py`

Plots per-file mean/std bar charts (`plot_mean_std_analysis`) and observation histograms (`plot_observation_analysis`) before and after file-level filtering.

---

## 9. Settings & Configuration

The outlier-rejection behavior is controlled via the `obs` settings dictionary:

| Setting | Default | Meaning |
|---|---|---|
| `residual_filtering` | `True` | Enable the 1.5 arcsec residual filter |
| `epoch_filter_dict` | `None` | Optional dict of exact epochs to remove per file |

**Locations:**
- `ConcurentEstimationRealObservations.py` — `settings_obs["residual_filtering"] = True`, `epoch_filter_dict = None`.
- `Experiments/settings.py` — `"residual_filtering": True`, `"epoch_filter_dict": None`.
- `Test_Observations.py` — same defaults.

---

## 10. Summary of Distinct Rejection Strategies

| # | Strategy | Granularity | Threshold | Stage |
|---|---|---|---|---|
| 1 | **Residual filtering** | Per observation | 1.5 arcsec | OD pipeline (primary) |
| 2 | **Epoch filtering** | Per observation | Exact epochs (manual) | OD pipeline (optional) |
| 3 | **SPICE-residual removal** | Per observation | 25 (hardcoded) | Raw NSDC processing (legacy) |
| 4 | **File-level mean/std filter** | Whole file | mean ≤ 2, std ≤ 2 | Data preparation |
| 5 | **2.5σ criterion** | Per observation | 2.5 × std | Commented-out alternative |

---

## 11. File Map

| File | Role |
|---|---|
| `HelperFunctions/ObsFunc.py` | Primary residual + epoch filtering in `LoadObservations()`. |
| `HelperFunctions/nsdc.py` | Legacy `remove_nsdc_outliers()` during raw processing. |
| `NSDC_processing.py` | Drives raw NSDC processing (calls `process_nsdc_file`). |
| `ObservationsTestFileIgnore.py` | File-level `filter_by_mean_std()`; writes `file_names.json`. |
| `ObservationImplementation.py` | Saves `residuals_rejected_epochs.json`. |
| `Test_Observations.py` | Accepted-vs-rejected diagnostics and PDF plots. |
| `MainPostprocessing.py` | (Commented) with/without-outlier comparison. |
| `ConcurentEstimationRealObservations.py` | Sets `residual_filtering` / `epoch_filter_dict`. |
| `Experiments/settings.py` | Default `residual_filtering=True`. |

---

## 12. Notes & Caveats

1. **The 1.5 arcsec residual filter is the primary rejection mechanism** in the OD pipeline.
2. **Filtering precedes weighting** — outliers are removed before weights are computed.
3. **Rejected epochs are always tracked** (`epochs_rejected`), even when filtering is disabled, enabling diagnostics.
4. The **legacy `remove_nsdc_outliers` threshold (25)** is hardcoded and the `outlier_limit` parameter is unused; a **2.5σ** criterion exists only as commented-out code.
5. **File-level filtering** is a coarse pre-processing step that can remove entire files, whereas residual filtering is fine-grained (per observation).
6. The **epoch filter** allows manual, inspection-driven removal of specific bad epochs.
7. The `FilterObservations` global flag gates the filtering block in the weights path — if disabled, no filtering occurs.

---

*Generated from the repository at commit state on `main` (2026-09-03).*
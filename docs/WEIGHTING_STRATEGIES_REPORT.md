# Weighting Strategies in the Neptune–Triton Orbit Determination Repository

**Scope:** All observation-weighting schemes implemented across the codebase.
**Purpose:** Reference document for the thesis describing how observation weights are computed, combined, and applied in the Triton orbit-determination (OD) pipeline.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Core Concepts](#2-core-concepts)
3. [The Modern Weight Engine](#3-the-modern-weight-engine)
4. [Weight Types (Selector Reference)](#4-weight-types-selector-reference)
5. [Timeframe Splitting](#5-timeframe-splitting)
6. [Sigma Clipping](#6-sigma-clipping)
7. [The Legacy Table-Driven Approach](#7-the-legacy-table-driven-approach)
8. [Weight-Table Generation (Postprocessing)](#8-weight-table-generation-postprocessing)
9. [Experiment Drivers](#9-experiment-drivers)
10. [Analysis & Export Configs](#10-analysis--export-configs)
11. [Manual DEC Bias Injection](#11-manual-dec-bias-injection)
12. [Iterative Weight Loop](#12-iterative-weight-loop)
13. [Summary of Distinct Strategies](#13-summary-of-distinct-strategies)
14. [File Map](#14-file-map)
15. [Notes & Caveats](#15-notes--caveats)

---

## 1. Overview

The repository implements **two generations** of weighting approaches for the Triton OD problem:

| Generation | Mechanism | Location | Status |
|---|---|---|---|
| **Legacy** | Precomputed weight table (`summary.txt`) read into `LoadObservations()` | `HelperFunctions/ObsFunc.py` | Superseded, still present |
| **Modern** | Residual-driven `compute_and_assign_weights()` | `HelperFunctions/ObsFunc.py` | **Primary / canonical** |

Both approaches use **inverse-variance weights**:

$$w = \frac{1}{\sigma^2}$$

where $\sigma$ is an estimate of the observation accuracy (standard deviation or RMSE of post-fit residuals), expressed in radians. Weights are therefore in units of $1/\text{rad}^2$.

The weighting philosophy is: **observations that fit the reference trajectory poorly (high residual scatter) are down-weighted**, while well-behaved observations are up-weighted. This is a standard approach in least-squares orbit determination to prevent a few noisy observations from dominating the solution.

---

## 2. Core Concepts

### 2.1 Observation "ID" (per-file grouping)

Observations are grouped into **files** identified by a `set_id` of the form `Observatory_StudyID` (e.g. `689_nm0077`). Each file corresponds to one observatory + one observing campaign. A **per-file (ID) weight** is a single constant applied to all observations in that file.

### 2.2 Timeframe (per-night grouping)

Within a file, observations are further split into **timeframes** (roughly "nights") based on time gaps. A **per-timeframe (TF) weight** varies from observation to observation, capturing short-term noise variations within a file.

### 2.3 RA/DEC independence

Angular-position observations have two components: **Right Ascension (RA)** and **Declination (DEC)**. Weights can be computed **independently** for RA and DEC (recommended) or as a single combined value. When tabulated, RA and DEC weights are **interleaved** into a single array (even index = RA, odd index = DEC).

### 2.4 Weight application in Tudat

- `set_tabulated_weights(array)` — assigns a per-observation weight vector.
- `set_constant_weight(value, parser)` — assigns a single constant weight to all observations in a set.

---

## 3. The Modern Weight Engine

**Function:** `ObsFunc.compute_and_assign_weights()`
**File:** `HelperFunctions/ObsFunc.py` (line ~653)

This is the canonical implementation. It takes residuals from a **reference simulation**, computes per-file and per-timeframe statistics, and assigns weights to an `ObservationCollection`.

### 3.1 Inputs

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `residuals` | `np.ndarray [n,3]` | — | `[time, RA_residual, DEC_residual]` |
| `observations` | `ObservationCollection` | — | Collection to weight |
| `gap_threshold_hours` | `float` | `4.0` | Max gap (h) within a timeframe |
| `min_obs_per_frame` | `int` | `1` | Min obs before a frame break |
| `weight_type` | `str` | `'hybrid'` | Selector (see §4) |
| `min_sigma_arcsec` | `float` | `0.01` | Lower clip on σ (10 mas) |

### 3.2 Computation Steps

For each observation set (file):

1. **Per-file (ID) RMSE** — global RMSE of RA and DEC residuals across the whole file:
   $$\sigma_{\text{RA,file}} = \sqrt{\frac{1}{N}\sum r_{\text{RA}}^2}, \qquad \sigma_{\text{DEC,file}} = \sqrt{\frac{1}{N}\sum r_{\text{DEC}}^2}$$
   Then $w_{\text{RA,file}} = 1/\sigma_{\text{RA,file}}^2$, etc.

2. **Per-timeframe (TF) RMSE** — split the file into timeframes, compute RMSE per frame, and assign a local weight to each observation:
   $$w_{\text{RA,tf}} = \frac{1}{\sigma_{\text{RA,tf}}^2 \cdot n_{\text{tf}}}$$
   (Note the division by $n_{\text{tf}}$, the number of observations in the timeframe — this "descales" the weight per night.)

3. **Descaled per-timeframe RMSE** (used for the "new ID" variants):
   $$\sigma_{\text{RA,tf,descaled}} = \sigma_{\text{RA,tf}} \cdot \sqrt{n_{\text{tf}}}$$

4. **ID v2 RMSE** — RMS of the descaled per-timeframe RMSEs:
   $$\sigma_{\text{RA,id\_v2}} = \sqrt{\frac{1}{n_{\text{tf}}}\sum \left(\sigma_{\text{RA,tf}}\sqrt{n_{\text{tf}}}\right)^2}$$

5. **Hybrid combinations** — combine global (file) and local (timeframe) weights (see §4).

6. **Interleave & assign** — build `weights_array[0::2] = RA`, `weights_array[1::2] = DEC`, then `obs_set.set_tabulated_weights(weights_array)`.

7. **Record** — append per-observation metadata to a `weights_df` DataFrame (saved as `observation_weights.csv`).

### 3.3 Output

Returns `(observations, weights_df)` where `weights_df` contains columns:
`set_index, ref_point_id, obs_index, global_obs_index, timeframe, time, ra_residual, dec_residual, ra_rmse_id, dec_rmse_id, weight_ra, weight_dec, weight_type`.

---

## 4. Weight Types (Selector Reference)

The `weight_type` argument selects which weight is assigned. All are computed from the same underlying per-file and per-timeframe statistics.

| `weight_type` | Label | Strategy | Formula |
|---|---|---|---|
| `id` | **ID** | Per-file RMSE, constant per file | $w = 1/\sigma_{\text{file}}^2$ |
| `id_new_1` | **ID v1** | Per-file RMSE, scaled variant 1 | $w = 1/\sigma_{\text{id\_v1}}^2$ |
| `id_new_2` | **ID v2** | Per-file RMSE, scaled variant 2 | $w = 1/\sigma_{\text{id\_v2}}^2$ |
| `timeframe` | **TF** | Per-timeframe RMSE, local | $w = 1/\sigma_{\text{tf}}^2$ |
| `hybrid` | **Hybrid G** | Geometric mean of file & TF | $w = \sqrt{w_{\text{file}} \cdot w_{\text{tf}}}$ |
| `hybrid_old` | **Hybrid A** | Arithmetic mean of file & TF | $w = (w_{\text{file}} + w_{\text{tf}})/2$ |
| `hybrid_new_id` | **Hybrid G+v2** | Geometric mean, ID v2 file component | $w = \sqrt{w_{\text{id\_v2}} \cdot w_{\text{tf}}}$ |
| `hybrid_old_new_id` | **Hybrid A+v2** | Arithmetic mean, ID v2 file component | $w = (w_{\text{id\_v2}} + w_{\text{tf}})/2$ |

### 4.1 ID v1 vs ID v2

Both are "scaled per-file" weights. The scaling normalizes the per-file RMSE by the global RMSE ratio. The two variants differ in how the per-file RMSE is aggregated:

- **v1** uses a direct per-file RMSE scaling.
- **v2** computes the RMS of the **descaled per-timeframe RMSEs** ($\sigma_{\text{tf}}\sqrt{n_{\text{tf}}}$), which accounts for the number of observations per night.

In practice, **`id_new_2` is the preferred/default variant** used across the experiment templates.

### 4.2 Hybrid G vs Hybrid A

- **Hybrid G (geometric mean)**: $\sqrt{w_{\text{file}} \cdot w_{\text{tf}}}$ — a multiplicative blend that stays between the two and is scale-invariant.
- **Hybrid A (arithmetic mean)**: $(w_{\text{file}} + w_{\text{tf}})/2$ — an additive blend.

The geometric mean is generally preferred because it avoids the arithmetic mean's sensitivity to the absolute magnitude of the weights.

---

## 5. Timeframe Splitting

**Function:** `ObsFunc.split_observations_into_timeframes()`

```python
def split_observations_into_timeframes(times, gap_threshold_hours=4.0, min_obs_per_frame=1):
```

- Iterates over sorted observation times.
- Starts a **new timeframe** when the gap between consecutive observations exceeds `gap_threshold_hours` **AND** the current frame already has at least `min_obs_per_frame` observations.
- Returns an integer array of timeframe indices.

**Default parameters:** `gap_threshold_hours = 4.0`, `min_obs_per_frame = 1`.

This effectively groups observations into **nights** (a ~4-hour gap typically separates observing nights).

---

## 6. Sigma Clipping

To prevent a single near-perfect observation (or a timeframe with an artificially tiny RMSE) from receiving an astronomically large weight, the RMSE is **floored** at a minimum value:

```python
min_sigma_rad = min_sigma_arcsec / (3600 * 180 / np.pi)  # arcsec -> rad
if ra_rmse < min_sigma_rad:
    ra_rmse = min_sigma_rad
```

- **Default:** `min_sigma_arcsec = 0.01` (10 mas).
- **No cap:** setting `min_sigma_arcsec = 0.0` disables clipping (used by the `tf_weights_no_limit` variant).

Clipping is applied to both the per-file and per-timeframe RMSEs. The number of clipped observations is reported per file.

---

## 7. The Legacy Table-Driven Approach

**Function:** `ObsFunc.LoadObservations()`
**File:** `HelperFunctions/ObsFunc.py` (line ~70)

This older path reads a **precomputed weights DataFrame** (typically from `summary.txt`) and assigns weights via a set of boolean flags. It is invoked when `settings['obs']['use_old_obs_func'] == True`.

### 7.1 Flags

| Flag | Meaning |
|---|---|
| `weights` | The precomputed DataFrame (indexed by `id`) |
| `ra_dec_independent_weights` | Use separate RA/DEC columns vs. a single mean column |
| `std_weights` | Select std-based columns instead of RMSE-based |
| `timeframe_weights` | Assign per-timeframe tabulated weights |
| `per_night_weights` | Use per-night descaled TF columns |
| `per_night_weights_id` | Use per-night descaled ID columns |
| `per_night_weights_hybrid` | Use per-night descaled hybrid TF-ID columns |

### 7.2 Assignment logic

- **Timeframe mode** (`timeframe_weights == True`): expands the weight rows by `n_obs` (×2 for RA/DEC), selects the appropriate column, and calls `set_tabulated_weights`. If the expanded length is short, missing values are filled with the mean weight.
- **Per-ID mode** (else): computes the average of RA/DEC weights and calls `set_constant_weight`.

### 7.3 Column selection

The selected column depends on the flags:

| Condition | Column (std) | Column (rmse) |
|---|---|---|
| `per_night_weights` | `mean_weight_std_scaled` | `mean_weight_rmse_scaled` |
| `per_night_weights_id` | `mean_weight_std_id_scaled` | `mean_weight_rmse_id_scaled` |
| `per_night_weights_hybrid` | `mean_weight_std_tf_id_scaled` | `mean_weight_rmse_tf_id_scaled` |
| default | `mean_weight_std` | `mean_weight_rmse` |

For `ra_dec_independent_weights == True`, the columns `weight_rmse_ra_tf_id_scaled` and `weight_rmse_dec_tf_id_scaled` are used and interleaved.

---

## 8. Weight-Table Generation (Postprocessing)

**Function:** `MainPostprocessing.create_weights_summary()`
**File:** `MainPostprocessing.py` (line ~1605)

This function generates the `summary.txt` weight table from a simulation's residuals. It produces a rich set of weight columns:

### 8.1 Per-ID statistics

- `std_ra`, `std_dec` — standard deviation per file (NaN → 1 arcsec for single-observation files).
- `rmse_ra`, `rmse_dec` — RMSE per file.
- `weight_std_ra_id`, `weight_std_dec_id` — $1/\sigma_{\text{std}}^2$.
- `weight_rmse_ra_id`, `weight_rmse_dec_id` — $1/\sigma_{\text{rmse}}^2$.

### 8.2 Per-timeframe statistics

- `residual_ra_std`, `residual_dec_std` — per-frame std (NaN-filled with n_obs-weighted means).
- `residual_ra_rms`, `residual_dec_rms` — per-frame RMSE.
- `weight_std_ra`, `weight_std_dec` — $1/\sigma_{\text{std}}^2$.
- `weight_rmse_ra`, `weight_rmse_dec` — $1/\sigma_{\text{rmse}}^2$.

### 8.3 Hybrid TF-ID

- `residual_ra_rms_tf_id` = (TF RMSE + ID RMSE)/2.
- `weight_rmse_tf_id_ra`, `weight_rmse_tf_id_dec` — $1/\sigma_{\text{tf\_id}}^2$.

### 8.4 Descaled (per-night) variants

All weights are also produced **descaled by $\sqrt{n_{\text{obs}}}$**:
- `weight_*_scaled` (TF), `weight_*_id_scaled` (ID), `weight_rmse_*_tf_id_scaled` (hybrid).

### 8.5 Mean weights

- `mean_weight_std`, `mean_weight_rmse` — mean of RA/DEC.
- Corresponding `_scaled`, `_id_scaled`, `_tf_id_scaled` means.

### 8.6 Difference diagnostics

- `ra_diff_id`, `dec_diff_id` = RMSE − std (per ID).
- `ra_diff`, `dec_diff` = RMSE − std (per TF).

---

## 9. Experiment Drivers

**File:** `EstimationAnalysisTemplates.py`

### 9.1 `CASE1_Manual_Bias()`

- Applies a manual DEC bias (−0.2 arcsec on `689_nm0077`).
- Computes weights from the initial (IAUPole) simulation residuals using `weight_type='id_new_2'`.
- Runs weighted variants (IAUPole vs SimPole rotation models).

### 9.2 `WeightSchemeAnalysis()`

- Runs **all** `weight_type` variants (ID v1/v2, ID, TF, TF-no-cap, hybrid G/A, hybrid G/A+v2) for `initial_state + pole_librations`.
- Uses a reference simulation (`SimPole_pole_lib_cov`) for the residuals.
- Applies manual DEC bias first.
- Saves `observation_weights.csv` per variant.

### 9.3 `WeightLoop()`

- Iterative re-estimation: 5 loops, each using `id_new_2` weights recomputed from the **previous iteration's** residuals.
- Updates the initial state and pole librations each iteration.
- See §12.

### 9.4 `SimObs_ParameterAnalysis()`

- Simulated-observation parameter analysis (contains commented weight-type variants).

---

## 10. Analysis & Export Configs

**Folder:** `ExportConfigs/`

| Config | Purpose |
|---|---|
| `WeightAnalysis.py` | Compares 9 weight schemes (initial_state only). |
| `WeightAnalysis_Pole.py` | Same schemes + pole estimation, plus a no-covariance reference. |
| `WeightAnalysis_Old.py` | Same figures on the old dataset. |
| `WeightComparison.py` | Cross-dataset 3-panel comparison (WA-IAU / WA-FIT / WA-FIT-PL). |

**Dashboard:** `DashInteractivePlotFull.py` computes WRMS and cost from `weight_info`:
- `ra_rmse_tf = 1/sqrt(weight_ra)`, `dec_rmse_tf = 1/sqrt(weight_dec)`.
- Weighted residuals in σ units: `residual_rad * sqrt(weight)`.

---

## 11. Manual DEC Bias Injection

**Function:** `ObsFunc.apply_dec_bias_to_observations()`

A **systematic-error handling** step applied *before* weighting. It adds a fixed DEC offset (in arcsec) to specific observation sets:

```python
bias_dict = {"689_nm0077": -0.2}  # arcsec
```

- Converts arcsec → rad and adds to the DEC component (`obs[1::2] += bias_rad`).
- Recomputes residuals afterward.
- Used in `CASE1_Manual_Bias`, `WeightSchemeAnalysis`, and `WeightLoop`.

This is not a weighting strategy per se, but it interacts with weighting: the bias is applied first, then weights are computed from the biased residuals.

---

## 12. Iterative Weight Loop

**Function:** `EstimationAnalysisTemplates.WeightLoop()`

A **self-consistent weighting** scheme:

1. Start from a reference simulation (`loop_0`).
2. For each iteration `i` (1–5):
   - Load the previous iteration's final state.
   - Extract residuals from the previous iteration.
   - Recompute `id_new_2` weights from those residuals.
   - Re-run the estimation with the updated weights.
3. Save `observation_weights.csv` per iteration.

This allows the weights to converge with the solution, rather than being fixed from a single reference.

---

## 13. Summary of Distinct Strategies

| # | Strategy | Description |
|---|---|---|
| 1 | **Unit weights** | No custom weighting (reference runs). |
| 2 | **Per-file (ID) RMSE** | Constant weight per observation file. |
| 3 | **Per-file (ID) std** | Constant weight per file from std (legacy). |
| 4 | **Per-timeframe (TF) RMSE** | Local weight per observation. |
| 5 | **Hybrid G** | Geometric mean of ID + TF. |
| 6 | **Hybrid A** | Arithmetic mean of ID + TF. |
| 7 | **ID v1 / v2** | Scaled per-file weights (v2 preferred). |
| 8 | **Hybrid G/A + v2** | Hybrids with scaled ID component. |
| 9 | **Sigma-clipped vs no-cap** | With/without 10 mas floor. |
| 10 | **Manual DEC bias** | Systematic-error injection before weighting. |
| 11 | **Iterative weight loop** | Weights recomputed each estimation iteration. |

---

## 14. File Map

| File | Role |
|---|---|
| `HelperFunctions/ObsFunc.py` | Modern engine (`compute_and_assign_weights`), legacy loader (`LoadObservations`), bias injection. |
| `MainPostprocessing.py` | `create_weights_summary()` → `summary.txt` weight table. |
| `EstimationAnalysisTemplates.py` | `CASE1_Manual_Bias`, `WeightSchemeAnalysis`, `WeightLoop`, `SimObs_ParameterAnalysis`. |
| `ObservationImplementation.py` | Main estimation engine; routes to old/new observation loading. |
| `ConcurentEstimationRealObservations.py` | Real-observation experiment driver. |
| `Experiments/settings.py` | Default settings (`use_weights=True`). |
| `Experiments/configs/weight_loop.py` | Weight-loop experiment variants. |
| `ExportConfigs/WeightAnalysis*.py` | Figure export configs. |
| `DashInteractivePlotFull.py` | Dashboard; WRMS/cost from weights. |
| `summary.txt` | Example weight table (per-ID/TF columns). |

---

## 15. Notes & Caveats

1. **Weights are inverse-variance** ($1/\sigma^2$), so larger weights = more trusted observations.
2. **RA/DEC interleaving** is critical: tabulated arrays must place RA at even indices and DEC at odd indices.
3. **`id_new_2` is the default/preferred** weight type across the experiment templates.
4. **Sigma clipping** (10 mas default) prevents over-weighting of near-perfect observations; disabling it (`tf_weights_no_limit`) can produce extreme weights.
5. The **legacy `LoadObservations` path** is superseded but retained; the modern `compute_and_assign_weights` is the canonical implementation.
6. **Manual DEC bias** is applied before weighting and is a systematic-error handling step, not a weighting scheme per se.
7. The **iterative weight loop** is the most sophisticated scheme, allowing weights to converge with the solution.

---

*Generated from the repository at commit state on `main` (2026-08-28).*
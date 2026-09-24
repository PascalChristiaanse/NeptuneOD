# Weighting Strategy Comparison — Methodology & Repository Readiness Report

> **Purpose:** Document the full methodology of the weighting strategy experiments from Atanas' thesis (Chapter 6) and assess the current NeptuneOD repository's ability to reproduce them.

---

## Table of Contents

1. [Overview of the Thesis Experiments](#1-overview-of-the-thesis-experiments)
2. [Chapter 3: Weighting Strategy Derivation (Theoretical Foundation)](#2-chapter-3-weighting-strategy-derivation)
3. [Chapter 6: Weighting Strategy Comparison Experiments](#3-chapter-6-weighting-strategy-comparison-experiments)
4. [The Two Analyses: WA-IAU and WA-FIT-PL](#4-the-two-analyses-wa-iau-and-wa-fit-pl)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Current Repository Implementation Status](#6-current-repository-implementation-status)
7. [Readiness Assessment: What Can Be Run Now](#7-readiness-assessment)
8. [Implementation Roadmap for Missing Features](#8-implementation-roadmap)
9. [Summary Table](#9-summary-table)

---

## 1. Overview of the Thesis Experiments

The thesis uses three sequential analyses:

| Analysis | Chapter | Purpose |
|---|---|---|
| **Simulated Observations Analysis** | Ch. 4 | Identify which dynamical parameters Triton's orbit is most sensitive to (using noise-free simulated 3D positions from NEP097) |
| **Pole Estimation Analysis** | Ch. 5 | Estimate Neptune pole parameters from real astrometric data using the best weighting scheme |
| **Weighting Strategy Analysis** | **Ch. 6** | Compare 6 weighting schemes across 2 initial-condition scenarios |

### Research Question Addressed (RQ3)

> *What weighting strategy for astrometric datasets produces formal errors that are statistically consistent with the actual solution accuracy?*

---

## 2. Chapter 3: Weighting Strategy Derivation

### 2.1 Naming Convention

The thesis uses these names. The mapping to the reference codebase names and current NeptuneOD configs is shown below.

| Thesis Label (Ch. 6) | Reference Codebase | NeptuneOD Config Key | Current Status |
|---|---|---|---|
| **Per file** | `id` | `conf/weighting/id.yaml` | ✅ **Implemented** |
| **Scaled per file** | `id_new_2` (ID v2) | `conf/weighting/id_v2.yaml` | ✅ Implemented |
| **Per timeframe** | `timeframe` (descaled) | `conf/weighting/timeframe.yaml` | ✅ **Implemented** |
| **Per timeframe free** | `timeframe` (no min sigma) | `conf/weighting/timeframe_free.yaml` | ✅ **Implemented** |
| **Scaled hybrid geom. (G+v2)** | `hybrid_new_id` | `conf/weighting/hybrid_geometric_v2.yaml` | ✅ **Implemented** (thesis-faithful) |
| **Scaled hybrid arith. (A+v2)** | `hybrid_old_new_id` | `conf/weighting/hybrid_arithmetic.yaml` | ✅ **Implemented** |

> **Note:** The original `hybrid_geometric` strategy (registered as `"hybrid_geometric"`) uses a different formula (plain ID + unscaled TF) and is **not** the thesis scheme. Use `hybrid_geometric_v2` for the thesis-faithful variant.

### 2.2 Mathematical Formulation (from §3.2.2)

All schemes operate on per-file residuals $r^\alpha_{i,f}$ (RA) and $r^\delta_{i,f}$ (DEC). Notation shown for RA only; DEC follows identically.

#### Base Scheme 1: Per-File Weighting (ID)

Assigns a **single constant weight** to all observations in a file $f$ with $N_f$ observations:

$$\epsilon^\alpha_f = \sqrt{\frac{1}{N_f}\sum_{i=1}^{N_f} (r^\alpha_{i,f})^2}$$

$$w^\alpha_f = \frac{1}{(\epsilon^\alpha_f)^2}$$

- **Assumption:** All observations in a file have comparable accuracy, zero-mean Gaussian noise.
- **Caveat:** Performs poorly when quality varies within a file (e.g., changing atmospheric conditions, instrument degradation).
- **Does not account for temporal clustering** — densely packed observations artificially reduce RMSE.

#### Base Scheme 2: Per-Timeframe Weighting (TF)

Divides each file into **timeframes** (nights) by time gaps $>4$ hours. For timeframe $t$ with $n_t$ observations:

$$\epsilon^\alpha_t = \sqrt{\frac{1}{n_t}\sum_{i=1}^{n_t} (r^\alpha_{t,i})^2}$$

To **account for correlated errors within a night**, the RMSE is scaled by $\sqrt{n_t}$:

$$\tilde{\epsilon}^\alpha_t = \epsilon^\alpha_t \cdot \sqrt{n_t}$$

$$w^\alpha_t = \frac{1}{(\tilde{\epsilon}^\alpha_t)^2}$$

- A **lower bound** is imposed: $\epsilon_t \geq \upsilon_{\min} = 10$ mas (prevents artificially high weights from near-perfect timeframes).
- The **per timeframe free** variant removes this upper limit on weight magnitude.

#### Base Scheme 3: Scaled Per-File Weighting (ID v2)

Combines the granularity of timeframe-level statistics with file-level weight assignment. For each timeframe $t$ in file $f$, compute the **descaled RMSE** $\tilde{\epsilon}^\alpha_t$ as in TF. Then:

$$\epsilon^\alpha_f = \sqrt{\frac{1}{T_f}\sum_{t=1}^{T_f} (\tilde{\epsilon}^\alpha_t)^2}$$

$$w^\alpha_f = \frac{1}{(\epsilon^\alpha_f)^2}$$

- **This is the scheme used for all Pole Estimation Analysis (Ch. 5) results.**
- Equivalent to "ID v2" in the reference codebase.
- **This is what `id_v2` in NeptuneOD implements.**

#### Hybrid Scheme 1: Scaled Hybrid Geometric (Hybrid G+v2)

Geometric mean of the scaled per-file weight $w_f$ (ID v2) and the per-timeframe weight $w_t$ (TF):

$$w^{(g)}_{t,f} = \sqrt{w_f \cdot w_t}$$

- Bounded between the two constituent weights.
- Equal multiplicative influence — a low weight in one component pulls the hybrid down.

#### Hybrid Scheme 2: Scaled Hybrid Arithmetic (Hybrid A+v2)

Arithmetic mean of the scaled per-file weight $w_f$ (ID v2) and the per-timeframe weight $w_t$ (TF):

$$w^{(a)}_{t,f} = \frac{w_f + w_t}{2}$$

- **Dominated by the larger weight.** Since per-timeframe weights tend to be orders of magnitude larger, this skews toward TF.
- Less balanced than the geometric variant — the thesis ultimately prefers the geometric mean.

### 2.3 Observation Uncertainty from Weights (§3.2.3)

For any weight $w$ with units $[\text{rad}^{-2}]$:

$$\upsilon = \frac{1}{\sqrt{w}}$$

This is directly interpretable as expected scatter in radians (or mas).

### 2.4 Goodness-of-Fit Metrics (§3.2.4)

Three scalar metrics:

1. **Cost function:** $J = \frac{1}{N}\sum_i w_i r_i^2$
2. **Weighted RMS:** $\text{WRMS} = \sqrt{\frac{\sum_i w_i r_i^2}{\sum_i w_i}}$
3. **Unweighted RMS:** $\text{RMS} = \sqrt{\frac{1}{N}\sum_i r_i^2}$

---

## 3. Chapter 6: Weighting Strategy Comparison Experiments

### 3.1 Experimental Design

Two **separate analyses** are performed due to the sensitivity of the weighting comparison to the reference trajectory used for residual computation:

| Analysis | Initial Conditions | Estimated Parameters |
|---|---|---|
| **WA-IAU** | IAU 2015 pole model + NEP097 Neptune position | Triton initial state only |
| **WA-FIT-PL** | NEP097-fitted pole model + fitted initial state (from Ch. 5 "Final" solution) | Triton initial state + pole libration rates |

Within each analysis, **6 weighting schemes** are compared (Table 6.2):

| Label | Description |
|---|---|
| `ref` | Reference run: FitPole rotation model, state + pole libration estimation, scaled per file (ID v2) weights |
| `per file` | Per-observation-file RMSE weights (ID) |
| `scaled per file` | Per-file RMSE weights scaled by $\sqrt{N_f/T_f}$ (ID v2) |
| `per timeframe` | Per-timeframe RMSE weights (TF) with $\upsilon_{\min}=10$ mas |
| `per timeframe free` | Per-timeframe RMSE weights, **no upper limit** on weight magnitude |
| `scaled hybrid geom.` | Geometric average of scaled per file and per timeframe (Hybrid G+v2) |
| `scaled hybrid arith.` | Arithmetic average of scaled per file and per timeframe (Hybrid A+v2) |

> **Note:** `ref` only exists in WA-FIT-PL (it is the pole estimation solution itself). In WA-IAU there are only 6 weight schemes.

### 3.2 Data Used

The astrometric dataset spans **1963–2025** and includes:

- **Earth-based astrometric observations** from the NSDB (Natural Satellite DataBase, IMCCE)
  - Absolute astrometry (CCD, photographic plates)
  - Relative astrometry (micrometric position angle & separation)
- **Voyager 2 pre-encounter images** (Jacobson, 1991)
- **Gaia astrometry** (Gaia DR3, provided by Gaia Data Processing and Analysis Consortium)
- **HST astrometry** (Showalter et al., 2019)

Outlier rejection: **1.5 arcsec residual threshold** applied before weighting.

### 3.3 Weight Computation Procedure

1. **Propagate** the reference trajectory from initial conditions using the dynamical model.
2. **Compute O−C residuals** against the astrometric observations.
3. **Apply outlier rejection** (1.5 arcsec residual threshold).
4. **Compute weights** from the post-rejection residuals using each weighting scheme.
5. **Estimate** parameters using weighted least-squares (Tudat).
6. **Evaluate** solutions against NEP097.

### 3.4 Dynamical Model

The dynamical model (same as Ch. 5 best solution) includes:

- **Bodies:** Sun, 8 planets, Triton, Triton Spice (SPICE reference)
- **Triton gravity:** Point mass
- **Neptune gravity:** Jacobson (2009) zonal harmonics ($J_2$, $J_4$)
- **Neptune rotation:** IAU 2015 (WA-IAU) or fitted (WA-FIT-PL)
- **Pole model:** IAU 2015 (WA-IAU) or NEP097-fitted (WA-FIT-PL) — includes pole position ($\alpha_0, \delta_0$) and libration terms ($\alpha_1, \delta_1$)
- **Earth:** Oblate spheroid, GCRS→ITRS rotation with IAU 2006 nutation
- **Integrator:** RKF78, 3600 s fixed step
- **Time span:** 1963-01-01 to 2025-01-01

### 3.5 Key Findings (Summary)

1. **True-to-formal-error ratio** is the most robust metric across different initial conditions.
2. **Per file (ID)** consistently produces **overconfident formal errors** (ratio well above 1).
3. **Scaled per file (ID v2)** substantially improves on ID — a simple $\sqrt{N_f/T_f}$ correction.
4. **Per timeframe free** has extreme ratio inflation — removing the weight cap causes formal error collapse.
5. **Hybrid schemes** perform competitively on both RMS difference and error ratio.
6. **RMS difference with NEP097** is sensitive to initial conditions — scheme rankings reverse between WA-IAU and WA-FIT-PL.

---

## 4. The Two Analyses: WA-IAU and WA-FIT-PL

### WA-IAU

| Property | Value |
|---|---|
| Initial Triton state | NEP097 SPICE kernel |
| Neptune pole model | IAU 2015 (Archinal et al., 2018) |
| Estimated parameters | Triton initial state only |
| Weight reference trajectory | IAU 2015 propagation |
| Config (NeptuneOD equivalent) | `experiment=classic_triton_state` + `weighting=id_v2` |

### WA-FIT-PL

| Property | Value |
|---|---|
| Initial Triton state | Fitted state from Ch. 4 simulated obs analysis |
| Neptune pole model | NEP097-fitted (pole + libration from Ch. 5) |
| Estimated parameters | Triton initial state + pole libration rates |
| Weight reference trajectory | FitPole propagation |
| Config (NeptuneOD equivalent) | `experiment=wa_fit_pl` (initially configured for state+lib., classical+Voyager data, residual threshold outlier rejection) |

> **Note:** The WA-FIT-PL config uses the IAU 2015 pole model by default. To fully reproduce the thesis WA-FIT-PL analysis, you will need to override the dynamics config with the NEP097-fitted pole parameters. See the dynamics config for the specific pole parameter values from Table 5.5 of the thesis.

---

## 5. Evaluation Metrics

### 5.1 NEP097 Comparison Metrics

1. **RMS difference with NEP097** (total position, km) — the "true error" analogue
2. **RMS difference decomposed into RSW components** (Radial, Along-track, Cross-track)
3. **True-to-formal error ratio:** $\text{RMS}_{\text{SPICE}} / \text{RMS}_{\sigma}$
4. **RSW timeseries** of position differences and formal errors

### 5.2 Observational Goodness-of-Fit

1. **RMS of post-fit residuals** (arcsec) — independent of weighting scheme
2. **Weighted RMS** — depends on weighting scheme
3. **Cost function** — depends on weighting scheme

---

## 6. Current Repository Implementation Status

### 6.1 Weighting Subsystem (`src/orbitdet/observations/weighting/`)

| Component | File | Status | Notes |
|---|---|---|---|
| `WeightStrategy` (ABC) | `base.py` | ✅ Complete | Abstract base class |
| `WeightEngine` | `engine.py` | ✅ Complete | Orchestrates weighting over collection |
| `WeightingConfig` | `configs.py` | ✅ Complete | Dataclass, but `strategy` field is a `str` |
| `Group`, `GroupList`, `build_group_list` | `grouping.py` | ✅ Complete | Flexible grouping: set, timeframe, section, nested |
| `Registry` | `registry.py` | ✅ Complete | Decorator-based, like outlier subsystem |
| `IDWeight` (per file) | `strategies.py` | ✅ **Implemented** | Registered as `"id"` |
| `IDv2Weight` (scaled per file) | `strategies.py` | ✅ **Implemented** | Registered as `"id_v2"` |
| `HybridGeometricWeight` (legacy) | `strategies.py` | ✅ **Implemented** | Registered as `"hybrid_geometric"` — plain ID + unscaled TF, **not** thesis variant |
| `TFWeight` (per timeframe) | `strategies.py` | ✅ **Implemented** | Registered as `"timeframe"` — with $\sqrt{n_t}$ descalement |
| `TFFreeWeight` (per timeframe free) | `strategies.py` | ✅ **Implemented** | Registered as `"timeframe_free"` — no sigma cap |
| `HybridGeometricV2Weight` (thesis G+v2) | `strategies.py` | ✅ **Implemented** | Registered as `"hybrid_geometric_v2"` — ID v2 + descaled TF |
| `HybridArithmeticWeight` (thesis A+v2) | `strategies.py` | ✅ **Implemented** | Registered as `"hybrid_arithmetic"` — ID v2 + descaled TF, arithmetic mean |
| `FixedWeight` | `strategies.py` | ✅ **Implemented** | Registered as `"fixed"` |

### 6.2 YAML Configs (`conf/weighting/`)

| Config File | Status | Maps to Thesis |
|---|---|---|
| `id_v2.yaml` | ✅ Complete | **Scaled per file** (ID v2) |
| `id.yaml` | ✅ **New** | **Per file** (ID) |
| `timeframe.yaml` | ✅ **New** | **Per timeframe** (TF, descaled) |
| `timeframe_free.yaml` | ✅ **New** | **Per timeframe free** (no sigma cap) |
| `hybrid_geometric_v2.yaml` | ✅ **New** | **Scaled hybrid geom.** (thesis-faithful G+v2) |
| `hybrid_arithmetic.yaml` | ✅ **New** | **Scaled hybrid arith.** (A+v2) |
| `hybrid_geometric.yaml` | ✅ Complete | Legacy variant (plain ID + unscaled TF, **not** thesis) |
| `fixed.yaml` | ✅ Complete | Fixed sigma (not in thesis Ch. 6) |
| `fixed_voyager.yaml` | ✅ Complete | Voyager fixed sigmas (not in thesis Ch. 6) |
| `disabled.yaml` | ✅ Complete | No weighting (not in thesis Ch. 6) |

### 6.3 Outlier Rejection Subsystem (`src/orbitdet/observations/outlier_rejection/`)

| Component | Status | Notes |
|---|---|---|
| `OutlierStrategy` (ABC) | ✅ Complete | |
| `OutlierEngine` | ✅ Complete | Supports per-set filtering |
| `ResidualThresholdOutlier` | ✅ Complete | 1.5 arcsec default, per-set thresholds possible |
| `EpochFilterOutlier` | ✅ Complete | |
| `per_set_thresholds.yaml` | ✅ Complete | Voyager: 0.5 arcsec, ground: 1.5 arcsec |
| `residual_threshold.yaml` | ✅ Complete | Matches thesis 1.5 arcsec filter |
| `epoch_filter.yaml` | ✅ Complete | |
| `disabled.yaml` | ✅ Complete | |

### 6.4 Experiment Configs (`conf/experiment/`)

| Config File | Status | Maps to Thesis Analysis |
|---|---|---|
| `classic_triton_state.yaml` | ✅ Complete | **WA-IAU** (classical data only, state estimation) |
| `wa_fit_pl.yaml` | ✅ **New** | **WA-FIT-PL** (classical+Voyager, state+lib., outlier rejection) |
| `atanas_triton_state.yaml` | ✅ Complete | Atanas' Triton state estimation (generated data) |
| `atanas2026_simulated.yaml` | ✅ Complete | Simulated observations analysis |
| `minimal_experiment.yaml` | ✅ Complete | Quick test config |

### 6.5 Data Configs (`conf/data/`)

| Config | Status | Notes |
|---|---|---|
| `data: classical` | ✅ Complete | XY-only collection, classical dynamics |
| `data: classical_voyager` | ✅ Complete | XY+Voyager collection, classical_voyager dynamics |
| `collection: classical_plus_voyager` | ✅ Complete | Both NSDB and Voyager |

### 6.6 Parameter Configs (`conf/parameters/`)

| Config | Status | Thesis Equivalent |
|---|---|---|
| `initial_state.yaml` | ✅ Complete | State-only estimation (WA-IAU) |
| `state_plus_pole_libration.yaml` | ✅ **New** | State + pole libration rates (WA-FIT-PL) |
| `state_plus_pole_position_libration.yaml` | ✅ **New** | State + pole position + libration |
| `state_plus_neptune_GM.yaml` | ✅ Complete | |
| `state_plus_neptune_GM_gravity.yaml` | ✅ Complete | |
| `state_plus_neptune_GM_rotation.yaml` | ✅ Complete | |
| `state_plus_gravity.yaml` | ✅ Complete | |
| `state_plus_rotation.yaml` | ✅ Complete | |

### 6.7 Sweep Configs

| Config | Status | Notes |
|---|---|---|
| `sweep/params_x_data.yaml` | ✅ Complete | Sweeps 6 parameter combos × 2 data variants — good baseline |

### 6.8 ⚠️ Critical Nuance: `hybrid_geometric` Formula Mismatch

The original `HybridGeometricWeight` in `strategies.py` computed:

```
w_ra_set = 1 / (set-level RA RMSE)²          ← plain per-file RMSE (ID, not ID v2!)
w_ra_tf  = 1 / (timeframe RA RMSE)²          ← unscaled TF RMSE (no √n_t factor!)
w_hybrid = √(w_ra_set · w_ra_tf)             ← geometric mean ✓
```

The thesis "scaled hybrid geom." (and the reference codebase `hybrid_new_id`) instead combines:

```
w_ra_set = 1 / (ID v2 RMSE)² = 1 / RMS(σ_tf · √n_t)²
w_ra_tf  = 1 / (σ_tf · √n_t)²                ← descaled TF RMSE
w_hybrid = √(w_ra_set · w_ra_tf)
```

**✅ FIX (implemented):** A new strategy `HybridGeometricV2Weight` (registered as `"hybrid_geometric_v2"`) was added that faithfully implements the thesis formula. Its config is at `conf/weighting/hybrid_geometric_v2.yaml`. The original `hybrid_geometric` is kept unchanged as a different variant (`Hybrid G+ID`, not used in the thesis).

---

## 7. Readiness Assessment

### ✅ What Can Be Run Immediately

The following experiments can be run **right now** without any code changes:

1. **WA-IAU base estimation (state only, classical data):**
   ```bash
   python scripts/solve_least_squares.py --config-name=experiment/classic_triton_state
   ```
   Uses `conf/experiment/classic_triton_state.yaml` which defaults to `hybrid_geometric` weighting.

2. **WA-IAU with any weighting scheme:**
   ```bash
   python scripts/solve_least_squares.py weighting=id_v2
   ```
   or via multirun:
   ```bash
   python scripts/solve_least_squares.py --multirun weighting=id,id_v2,timeframe,timeframe_free,hybrid_geometric_v2,hybrid_arithmetic
   ```

3. **Classical vs Classical+Voyager data comparison (as in thesis):**
   ```bash
   python scripts/solve_least_squares.py --multirun data=classical,classical_voyager
   ```

4. **Parameter sweep (as configured in `sweep/params_x_data.yaml`):**
   ```bash
   python scripts/solve_least_squares.py --config-name=sweep/params_x_data
   ```

5. **Outlier rejection with 1.5 arcsec threshold:**
   ```bash
   python scripts/solve_least_squares.py outlier_rejection=residual_threshold
   ```

6. **State + pole libration estimation:**
   ```bash
   python scripts/solve_least_squares.py parameters=state_plus_pole_libration
   ```

7. **WA-FIT-PL style estimation:**
   ```bash
   python scripts/solve_least_squares.py experiment=wa_fit_pl weighting=hybrid_geometric_v2
   ```

8. **RSW vs NEP097 comparison (on existing results):**
   ```bash
   python scripts/compare_weighting_strategies.py \
       --multirun-dir multirun/2026-09-10/ \
       --output-dir results/weighting_comparison
   ```

9. **Fixed sigma weights (Voyager pre/post encounter):**
   ```bash
   python scripts/solve_least_squares.py weighting=fixed_voyager
   ```

### ⚠️ What Can Be Run with Config-Only Changes

1. **Full data range (auto-detected from datasets):**
   Already supported — the script auto-detects observation date bounds from the dataset files.

2. **Combined outlier rejection + weighting:**
   Already in the pipeline — both engines are called sequentially in `solve_least_squares.py`.

3. **Hybrid geometric with custom grouping (different gap thresholds, sections):**
   Already configurable via YAML overrides.

### ✅ What Is Now Implemented (this session)

| Feature | New Files |
|---|---|
| **ID (per file) weighting strategy** | `conf/weighting/id.yaml` |
| **TF (per timeframe) weighting strategy** | `conf/weighting/timeframe.yaml` |
| **TF free (per timeframe free) weighting strategy** | `conf/weighting/timeframe_free.yaml` |
| **Hybrid A+v2 (arithmetic) weighting strategy** | `conf/weighting/hybrid_arithmetic.yaml` |
| **Hybrid G+v2 (thesis-faithful geometric) weighting strategy** | `conf/weighting/hybrid_geometric_v2.yaml` |
| **WA-FIT-PL experiment config** | `conf/experiment/wa_fit_pl.yaml` |
| **Pole libration parameter config** | `conf/parameters/state_plus_pole_libration.yaml` |
| **Pole position + libration parameter config** | `conf/parameters/state_plus_pole_position_libration.yaml` |
| **Pole rate estimation support** | added to `src/orbitdet/estimation/estimators.py` |
| **RSW difference computation against NEP097** | `scripts/compare_weighting_strategies.py` |
| **Tests for all new strategies** | added to `test/test_weight_strategies.py` (49→85 tests) |

### ❌ What Still Requires Work

| Feature | Effort | Priority |
|---|---|---|
| **NEP097-fitted dynamics config** (FitPole initial state + pole model) | Medium | High — needed for WA-FIT-PL initial conditions |
| **Iterative weight loop** (recomputing weights each estimation iteration) | High | Low — thesis does one-shot weighting |
| **Chapter 4 simulation analysis config** (noise-free 3D positions) | Low | Medium |
| **Figure styling parity** (match thesis exact figures 6.1–6.15) | Medium | Low |
| **Cost function / WRMS / RMS goodness-of-fit metrics** | Low | Medium |

---

## 8. Implementation Roadmap

### Phase 1: Missing Weighting Strategies (✅ All Implemented)

All six weighting strategies (ID, ID v2, TF, TF free, Hybrid G+v2, Hybrid A+v2) have been implemented, registered, tested, and given YAML configs. Registered strategy keys:

```
['fixed', 'hybrid_arithmetic', 'hybrid_geometric', 'hybrid_geometric_v2',
 'id', 'id_v2', 'timeframe', 'timeframe_free']
```

### Phase 2: WA-FIT-PL Experiment Config (✅ Implemented)

- `conf/parameters/state_plus_pole_libration.yaml` — pole libration estimation
- `conf/parameters/state_plus_pole_position_libration.yaml` — position + libration
- `conf/experiment/wa_fit_pl.yaml` — WA-FIT-PL style experiment
- `src/orbitdet/estimation/estimators.py` — supports `iau_rotation_model_pole_rate` and `iau_rotation_model_pole_librations` with configurable frequencies

### Phase 3: Evaluation Scripts (✅ Implemented)

- `scripts/compare_weighting_strategies.py` — calculates RSW difference vs NEP097, formal error propagation, true-to-formal-error ratio, generates comparison plots and summary tables

### Phase 4: YAML Configs (✅ All Created)

- `conf/weighting/id.yaml` — per file scheme
- `conf/weighting/timeframe.yaml` — per timeframe scheme (descaled)
- `conf/weighting/timeframe_free.yaml` — per timeframe free (no sigma cap)
- `conf/weighting/hybrid_arithmetic.yaml` — scaled hybrid arith.
- `conf/weighting/hybrid_geometric_v2.yaml` — scaled hybrid geom. (thesis-faithful)

### Future Work

| Item | Status |
|---|---|
| NEP097-fitted dynamics config (FitPole) | ❌ Not done |
| Iterative weight loop | ❌ Not done |
| Thesis figure styling | ❌ Not done |
| Chapter 4 simulated observations config | ❌ Not done |

---

## 9. Summary Table

| Thesis Scheme | NeptuneOD Status | Config Exists? | Strategy Implemented? | Effort to Add |
|---|---|---|---|---|
| **Per file (ID)** | ✅ **Implemented** | `conf/weighting/id.yaml` | `IDWeight` | Done |
| **Scaled per file (ID v2)** | ✅ Complete | `id_v2.yaml` | `IDv2Weight` | — |
| **Per timeframe (TF)** | ✅ **Implemented** | `conf/weighting/timeframe.yaml` | `TFWeight` | Done |
| **Per timeframe free** | ✅ **Implemented** | `conf/weighting/timeframe_free.yaml` | `TFFreeWeight` | Done |
| **Scaled hybrid geom. (G+v2)** | ✅ **Implemented** | `conf/weighting/hybrid_geometric_v2.yaml` | `HybridGeometricV2Weight` | Done |
| **Scaled hybrid arith. (A+v2)** | ✅ **Implemented** | `conf/weighting/hybrid_arithmetic.yaml` | `HybridArithmeticWeight` | Done |
| **Outlier rejection (1.5")** | ✅ Complete | `residual_threshold.yaml` | `ResidualThresholdOutlier` | — |
| **Voyager data** | ✅ Complete | `data/classical_voyager` | — | — |
| **State-only estimation** | ✅ Complete | `experiment/classic_triton_state` | — | — |
| **State+pole lib. estimation** | ✅ **Implemented** | `conf/parameters/state_plus_pole_libration.yaml` | + `estimators.py` | Done |
| **State+pole pos.+lib. estimation** | ✅ **Implemented** | `conf/parameters/state_plus_pole_position_libration.yaml` | + `estimators.py` | Done |
| **WA-FIT-PL experiment** | ✅ **Implemented** | `conf/experiment/wa_fit_pl.yaml` | — | Done |
| **Pole rate estimation** | ✅ **Implemented** | (via `estimators.py` `iau_rotation_model_pole_rate`) | + `estimators.py` | Done |
| **Fixed sigma weighting** | ✅ **Implemented** | `conf/weighting/fixed.yaml` | `FixedWeight` | Done |
| **RSW vs NEP097 evaluation** | ✅ **Implemented** | `scripts/compare_weighting_strategies.py` | — | Done |
| **Weight summary tables** | ✅ Complete | — | `WeightSummaryTable` | — |
| **LaTeX/Excel export** | ✅ Complete | — | In `solve_least_squares.py` | — |

### 9.1 Registered Strategies (verified)

```
>>> from orbitdet.observations.weighting.registry import list_registered_strategies
>>> list_registered_strategies()
['fixed', 'hybrid_arithmetic', 'hybrid_geometric', 'hybrid_geometric_v2',
 'id', 'id_v2', 'timeframe', 'timeframe_free']
```

---

## Appendix: Command Examples

### Run basic WA-IAU-like experiment:

```bash
# Activate environment
conda activate NeptuneOD

# State estimation, classical data, ID v2 weighting, outlier rejection
python scripts/solve_least_squares.py \
    experiment=classic_triton_state \
    weighting=id_v2 \
    outlier_rejection=residual_threshold
```

### Run with Voyager data:

```bash
python scripts/solve_least_squares.py \
    experiment=classic_triton_state \
    data=classical_voyager \
    weighting=id_v2
```

### Parameter sweep (existing config):

```bash
python scripts/solve_least_squares.py --config-name=sweep/params_x_data
```

### Full weighting strategy comparison (all 6 thesis schemes):

```bash
python scripts/solve_least_squares.py --multirun \
    experiment=classic_triton_state \
    weighting=id,id_v2,timeframe,timeframe_free,hybrid_geometric_v2,hybrid_arithmetic \
    outlier_rejection=residual_threshold
```

### Evaluate results against NEP097:

```bash
python scripts/compare_weighting_strategies.py \
    --multirun-dir multirun/<date>/ \
    --output-dir results/weighting_comparison
```

### Assess a single pre-existing results directory:

```bash
python scripts/compare_weighting_strategies.py \
    --results-dirs results/solve_least_squares/22b2c99_dirty/2026-08-24_15-39-38 \
    --output-dir results/weighting_comparison
```

---

*Report generated from analysis of `docs/ThesisAtanas.pdf` (Atanas D., 2026), repository source code at `/home/pascal/Documents/NeptuneOD`, and design documents in `docs/`.*
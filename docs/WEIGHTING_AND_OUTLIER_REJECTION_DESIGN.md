# Weighting & Outlier Rejection — Design Proposal

**Scope:** Architecture and implementation plan for adding observation weighting and outlier rejection to the NeptuneOD orbit-determination pipeline.
**Status:** Outlier rejection implemented (Phase 1 complete). Weighting, DEC bias, and iterative loop pending.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Design Principles](#2-design-principles)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Subsystem 1: Outlier Rejection](#4-subsystem-1-outlier-rejection)
5. [Subsystem 2: Weighting](#5-subsystem-2-weighting)
6. [Subsystem 3: Preprocessing (DEC Bias)](#6-subsystem-3-preprocessing-dec-bias)
7. [Iterative Weight Loop](#7-iterative-weight-loop)
8. [Integration into the Pipeline](#8-integration-into-the-pipeline)
9. [Hydra Configuration Design](#9-hydra-configuration-design)
10. [Implementation Phases](#10-implementation-phases)
11. [Comparison with the External Reference](#11-comparison-with-the-external-reference)
12. [Open Questions & Risks](#12-open-questions--risks)

---

## 1. Motivation

The NeptuneOD pipeline currently performs least-squares orbit determination without any observation weighting or outlier rejection. All observations contribute equally to the solution, regardless of their quality. This is suboptimal for two reasons:

1. **Noisy observations** can disproportionately bias the estimated orbit.
2. **Gross outliers** (e.g., transcription errors, bad plate reductions) can corrupt the solution entirely.

The external reference codebase (analysed in `docs/WEIGHTING_STRATEGIES_REPORT.md` and `docs/OUTLIER_REJECTION_REPORT.md`) implements a rich set of weighting strategies and outlier-rejection mechanisms. This document proposes bringing those capabilities into NeptuneOD while respecting the clean, modular architecture already established.

---

## 2. Design Principles

| Principle | Rationale |
|---|---|
| **Separation of concerns** | Weighting, outlier rejection, and DEC bias injection are conceptually distinct. Each gets its own package. |
| **Strategy pattern** | Each weighting scheme and rejection method is a pluggable strategy, registered via a decorator — exactly like the existing `register_dataset_factory` pattern. |
| **Composability** | Multiple outlier strategies can be chained. Weighting and outlier rejection are independent and can be enabled/disabled separately. |
| **Hydra-native configuration** | Every strategy is configurable via YAML files in `conf/`, composed via Hydra's `defaults` mechanism. |
| **Minimal pipeline changes** | The core estimation loop in `solve_least_squares.py` changes by only a few lines. New functionality is injected at well-defined points. |
| **Diagnostics-first** | Every operation returns metadata (rejected epochs, weight DataFrames) for downstream analysis and visualization. |

---

## 3. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      solve_least_squares.py                      │
│                                                                   │
│  1. Build environment & propagator                                │
│  2. Create observation collection  ──────────────────────────┐    │
│  3. Compute residuals (pre-fit)                                │    │
│  4. ┌──────────────────────────────────────────────────────┐   │    │
│     │  Outlier Rejection Engine  (optional)                 │   │    │
│     │  ┌────────────┐  ┌────────────┐  ┌──────────────┐   │   │    │
│     │  │Residual    │→│Epoch      │→│Sigma Clip   │   │   │    │
│     │  │Threshold   │  │Filter     │  │(legacy)     │   │   │    │
│     │  └────────────┘  └────────────┘  └──────────────┘   │   │    │
│     └──────────────────────────────────────────────────────┘   │    │
│  5. ┌──────────────────────────────────────────────────────┐   │    │
│     │  DEC Bias Injection  (optional)                       │   │    │
│     └──────────────────────────────────────────────────────┘   │    │
│  6. ┌──────────────────────────────────────────────────────┐   │    │
│     │  Weight Engine  (optional)                            │   │    │
│     │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │   │    │
│     │  │ID v2     │  │Hybrid G │  │TF / ID / HybridA │   │   │    │
│     │  │(default) │  │         │  │(extensible)      │   │   │    │
│     │  └──────────┘  └──────────┘  └──────────────────┘   │   │    │
│     └──────────────────────────────────────────────────────┘   │    │
│  7. Perform estimation (iterative loop)                         │    │
│     └── If recompute_each_iteration: goto step 6                │    │
│  8. Visualize & save results                                    │    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.1 Data Flow

```
Raw observations (SingleObservationSet)
    │
    ▼
[Optional] Outlier Rejection → filtered SingleObservationSet + rejected_epochs dict
    │
    ▼
[Optional] DEC Bias Injection → biased SingleObservationSet
    │
    ▼
[Optional] Weight Engine → weighted SingleObservationSet + weights_df DataFrame
    │
    ▼
Estimation (Tudat)
```

### 3.2 Package Structure

```
src/orbitdet/observations/
├── __init__.py
├── collection.py              # Existing: create_observation_collection
├── factory.py                 # Existing: central dispatcher
├── registry.py                # Existing: decorator-based registry
├── configs.py                 # Existing: ObservationDatasetConfig
├── weighting/                 # NEW
│   ├── __init__.py
│   ├── base.py                # Abstract WeightStrategy
│   ├── strategies.py          # ID, IDv2, TF, HybridG, HybridA, etc.
│   ├── engine.py              # WeightEngine (orchestrates computation + assignment)
│   ├── timeframe.py           # Timeframe splitting logic
│   └── configs.py             # Hydra-compatible config dataclasses
├── outlier_rejection/         # NEW
│   ├── __init__.py
│   ├── base.py                # Abstract OutlierStrategy
│   ├── strategies.py          # ResidualThreshold, EpochFilter, SigmaClip
│   ├── engine.py              # OutlierEngine (composes strategies)
│   └── configs.py             # Hydra-compatible config dataclasses
└── preprocessing/             # NEW
    ├── __init__.py
    ├── dec_bias.py            # DEC bias injection
    └── configs.py             # Hydra-compatible config dataclasses
```

---

## 4. Subsystem 1: Outlier Rejection

### 4.1 Core Abstraction

```python
class OutlierStrategy(ABC):
    """Base class for all outlier rejection strategies."""

    @abstractmethod
    def apply(
        self,
        observation_set: obs.SingleObservationSet,
        bodies: env.SystemOfBodies,
    ) -> tuple[obs.SingleObservationSet, dict]:
        """Apply the rejection strategy.

        Returns:
            Tuple of (filtered observation set, metadata dict).
            The metadata dict should contain at least:
                - 'n_accepted': int
                - 'n_rejected': int
                - 'rejected_epochs': list[float] (seconds since J2000)
        """
```

### 4.2 Strategies

| Strategy | Config Key | Description | Threshold |
|---|---|---|---|
| **Residual Threshold** | `residual_threshold` | Removes observations whose RA or DEC residual exceeds a fixed bound. Uses Tudat's `observation_filter` with `ObservationFilterType.residual_filtering`. | 1.5 arcsec (configurable) |
| **Epoch Filter** | `epoch_filter` | Removes specific epochs (by set_id). For manual removal of known bad observations. | Exact epochs (per set) |
| **Sigma Clip** | `sigma_clip` | Iterative sigma clipping: removes observations beyond N×σ, recomputes σ, repeats. | 3σ (configurable) |

### 4.3 Engine

```python
class OutlierEngine:
    """Orchestrates multiple outlier strategies over an ObservationCollection."""

    def __init__(self, strategies: list[OutlierStrategy]):
        self._strategies = strategies

    def apply(
        self,
        collection: obs.ObservationCollection,
        bodies: env.SystemOfBodies,
    ) -> tuple[obs.ObservationCollection, dict]:
        """Apply all strategies in sequence to each observation set.

        Returns:
            Tuple of (filtered collection, per-set rejection metadata).
        """
```

### 4.4 Config Example

```yaml
# conf/outlier_rejection/residual_threshold.yaml
outlier_rejection:
  enabled: true
  strategies:
    - type: residual_threshold
      threshold_arcsec: 1.5
```

```yaml
# conf/outlier_rejection/disabled.yaml
outlier_rejection:
  enabled: false
```

---

## 5. Subsystem 2: Weighting

### 5.1 Core Abstraction

```python
class WeightStrategy(ABC):
    """Base class for all weighting strategies."""

    @abstractmethod
    def compute_weights(
        self,
        observation_set: obs.SingleObservationSet,
        bodies: env.SystemOfBodies,
        gap_threshold_hours: float = 4.0,
        min_sigma_arcsec: float = 0.01,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Compute per-observation weights from residuals.

        Returns:
            Tuple of:
            - weights_array: np.ndarray of shape (2*n_obs,) with RA at even
              indices and DEC at odd indices (ready for set_tabulated_weights).
            - weights_df: pd.DataFrame with per-observation metadata.
        """
```

### 5.2 Strategies

| Strategy | Config Key | Description | Formula |
|---|---|---|---|
| **ID** | `id` | Per-file RMSE, constant per file | $w = 1/\sigma_{\text{file}}^2$ |
| **ID v1** | `id_v1` | Per-file RMSE, scaled variant 1 | $w = 1/\sigma_{\text{id\_v1}}^2$ |
| **ID v2** | `id_v2` | Per-file RMSE, scaled variant 2 (preferred default) | $w = 1/\sigma_{\text{id\_v2}}^2$ |
| **Timeframe** | `timeframe` | Per-timeframe RMSE, local | $w = 1/\sigma_{\text{tf}}^2$ |
| **Hybrid Geometric** | `hybrid_geometric` | Geometric mean of file & TF | $w = \sqrt{w_{\text{file}} \cdot w_{\text{tf}}}$ |
| **Hybrid Arithmetic** | `hybrid_arithmetic` | Arithmetic mean of file & TF | $w = (w_{\text{file}} + w_{\text{tf}})/2$ |
| **Hybrid G+v2** | `hybrid_geometric_v2` | Geometric mean, ID v2 file component | $w = \sqrt{w_{\text{id\_v2}} \cdot w_{\text{tf}}}$ |
| **Hybrid A+v2** | `hybrid_arithmetic_v2` | Arithmetic mean, ID v2 file component | $w = (w_{\text{id\_v2}} + w_{\text{tf}})/2$ |

### 5.3 Weight Formulas (Detailed)

#### Per-file (ID) RMSE

$$\sigma_{\text{RA,file}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} r_{\text{RA},i}^2}, \qquad
\sigma_{\text{DEC,file}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} r_{\text{DEC},i}^2}$$

$$w_{\text{RA,file}} = \frac{1}{\sigma_{\text{RA,file}}^2}, \qquad
w_{\text{DEC,file}} = \frac{1}{\sigma_{\text{DEC,file}}^2}$$

#### Per-timeframe (TF) RMSE

Observations are split into timeframes (nights) based on time gaps (default 4 hours). Within each timeframe $t$:

$$\sigma_{\text{RA,t}} = \sqrt{\frac{1}{n_t}\sum_{i \in t} r_{\text{RA},i}^2}, \qquad
w_{\text{RA,t}} = \frac{1}{\sigma_{\text{RA,t}}^2 \cdot n_t}$$

The division by $n_t$ (number of observations in the timeframe) "descales" the weight per night, preventing nights with many observations from dominating.

#### ID v2

First compute the descaled per-timeframe RMSE:

$$\sigma_{\text{RA,t,descaled}} = \sigma_{\text{RA,t}} \cdot \sqrt{n_t}$$

Then the ID v2 RMSE is the RMS of these descaled values:

$$\sigma_{\text{RA,id\_v2}} = \sqrt{\frac{1}{n_{\text{tf}}}\sum_{t} \left(\sigma_{\text{RA,t}} \sqrt{n_t}\right)^2}$$

This accounts for the number of observations per night, unlike the plain ID RMSE.

#### Hybrid Geometric

$$w_{\text{RA}} = \sqrt{w_{\text{RA,file}} \cdot w_{\text{RA,t}}}$$

A multiplicative blend that stays between the two and is scale-invariant.

### 5.4 Sigma Clipping

To prevent a single near-perfect observation from receiving an astronomically large weight, the RMSE is floored:

```python
min_sigma_rad = min_sigma_arcsec / (3600 * 180 / np.pi)  # arcsec → rad
sigma = max(sigma, min_sigma_rad)
```

- **Default:** `min_sigma_arcsec = 0.01` (10 mas).
- **Disable:** set `min_sigma_arcsec = 0.0`.

### 5.5 Timeframe Splitting

```python
def split_observations_into_timeframes(
    times: np.ndarray,
    gap_threshold_hours: float = 4.0,
    min_obs_per_frame: int = 1,
) -> np.ndarray:
    """Split sorted observation times into timeframe (night) indices.

    A new timeframe starts when the gap between consecutive observations
    exceeds gap_threshold_hours AND the current frame already has at least
    min_obs_per_frame observations.
    """
```

### 5.6 RA/DEC Interleaving

Tudat's `set_tabulated_weights` expects a flat array where even indices (0, 2, 4, …) correspond to RA and odd indices (1, 3, 5, …) to DEC. All strategies produce weights in this interleaved format.

### 5.7 Engine

```python
class WeightEngine:
    """Orchestrates weighting over an ObservationCollection."""

    def __init__(self, strategy: WeightStrategy):
        self._strategy = strategy

    def apply(
        self,
        collection: obs.ObservationCollection,
        bodies: env.SystemOfBodies,
        gap_threshold_hours: float = 4.0,
        min_sigma_arcsec: float = 0.01,
    ) -> tuple[obs.ObservationCollection, pd.DataFrame]:
        """Compute and assign weights to all observation sets.

        Returns:
            Tuple of (weighted collection, combined weights_df).
        """
```

### 5.8 Config Examples

```yaml
# conf/weighting/id_v2.yaml
weighting:
  enabled: true
  strategy: id_v2
  gap_threshold_hours: 4.0
  min_sigma_arcsec: 0.01
  ra_dec_independent: true
```

```yaml
# conf/weighting/hybrid_geometric.yaml
weighting:
  enabled: true
  strategy: hybrid_geometric
  gap_threshold_hours: 4.0
  min_sigma_arcsec: 0.01
  ra_dec_independent: true
```

```yaml
# conf/weighting/disabled.yaml
weighting:
  enabled: false
```

---

## 6. Subsystem 3: Preprocessing (DEC Bias)

### 6.1 Purpose

Inject systematic DEC offsets to specific observation sets before weighting. This models known systematic errors (e.g., a particular observatory's DEC measurements being consistently off by −0.2 arcsec).

### 6.2 Implementation

```python
def apply_dec_bias(
    collection: obs.ObservationCollection,
    biases: dict[str, float],  # set_id → bias_arcsec
) -> obs.ObservationCollection:
    """Apply DEC biases to specific observation sets.

    Biases are in arcseconds (positive = add to DEC). They are converted to
    radians and added to the DEC component of each observation.
    """
```

### 6.3 Config Example

```yaml
# conf/preprocessing/dec_bias.yaml
preprocessing:
  dec_bias:
    enabled: true
    biases:
      "689_nm0077": -0.2  # arcsec
```

---

## 7. Iterative Weight Loop

### 7.1 Concept

Instead of computing weights once from a reference trajectory, the iterative weight loop recomputes weights from each estimation iteration's residuals. This allows the weights to converge with the solution.

### 7.2 Algorithm

```
1. Compute initial residuals (pre-fit)
2. Compute weights from initial residuals
3. For iteration i = 1..N:
   a. Perform estimation iteration
   b. Extract post-fit residuals
   c. Recompute weights from post-fit residuals
   d. Update observation weights in the estimation input
4. Return final estimation output
```

### 7.3 Config

```yaml
estimation:
  max_iterations: 5
  weighting:
    enabled: true
    strategy: id_v2
    recompute_each_iteration: true   # enables iterative weight loop
    gap_threshold_hours: 4.0
    min_sigma_arcsec: 0.01
```

When `recompute_each_iteration: false` (default), weights are computed once from the pre-fit residuals and remain fixed throughout estimation.

---

## 8. Integration into the Pipeline

### 8.1 Changes to `solve_least_squares.py`

The existing pipeline flow is:

```python
# 1. Build environment, propagator, observations
observations, models = create_observation_collection(cfg, bodies)

# 2. Compute pre-fit residuals
obs.compute_residuals_and_dependent_variables(observations, simulators, bodies)

# 3. Plot pre-fit residuals
Residuals(cfg, observations).plot()

# 4. Build estimator and estimate
estimator = est_an.Estimator(...)
estimation_output = estimator.perform_estimation(estimation_input)
```

The proposed changes add three optional steps:

```python
# 1. Build environment, propagator, observations
observations, models = create_observation_collection(cfg, bodies)

# 2. Compute pre-fit residuals
obs.compute_residuals_and_dependent_variables(observations, simulators, bodies)

# === NEW: Outlier rejection ===
if cfg.outlier_rejection.enabled:
    outlier_engine = OutlierEngine.from_config(cfg.outlier_rejection)
    observations, rejected_metadata = outlier_engine.apply(observations, bodies)
    # Save rejected_metadata to JSON for diagnostics

# === NEW: DEC bias injection ===
if cfg.preprocessing.dec_bias.enabled:
    observations = apply_dec_bias(observations, cfg.preprocessing.dec_bias.biases)
    # Recompute residuals after bias
    obs.compute_residuals_and_dependent_variables(observations, simulators, bodies)

# === NEW: Weighting ===
if cfg.weighting.enabled:
    weight_engine = WeightEngine.from_config(cfg.weighting)
    observations, weights_df = weight_engine.apply(observations, bodies)
    # Save weights_df to CSV for diagnostics

# 3. Plot pre-fit residuals
Residuals(cfg, observations).plot()

# 4. Build estimator and estimate
estimator = est_an.Estimator(...)

# === MODIFIED: Iterative weight loop ===
if cfg.estimation.weighting.recompute_each_iteration:
    estimation_output = _run_iterative_weight_loop(
        cfg, estimator, estimation_input, observations, bodies
    )
else:
    estimation_output = estimator.perform_estimation(estimation_input)
```

### 8.2 New Public API

```python
# src/orbitdet/observations/__init__.py additions
from .weighting.engine import WeightEngine
from .weighting.strategies import list_registered_weight_strategies
from .outlier_rejection.engine import OutlierEngine
from .outlier_rejection.strategies import list_registered_outlier_strategies
from .preprocessing.dec_bias import apply_dec_bias
```

---

## 9. Hydra Configuration Design

### 9.1 Config File Structure

```
conf/
├── weighting/
│   ├── id_v2.yaml              # Default/preferred
│   ├── id.yaml
│   ├── id_v1.yaml
│   ├── timeframe.yaml
│   ├── hybrid_geometric.yaml
│   ├── hybrid_arithmetic.yaml
│   ├── hybrid_geometric_v2.yaml
│   ├── hybrid_arithmetic_v2.yaml
│   └── disabled.yaml
├── outlier_rejection/
│   ├── residual_threshold.yaml  # Default
│   ├── epoch_filter.yaml
│   ├── sigma_clip.yaml
│   └── disabled.yaml
└── preprocessing/
    └── dec_bias.yaml
```

### 9.2 Experiment Composition

Experiments compose these via Hydra's `defaults` list:

```yaml
# conf/experiment/classic_triton_state.yaml
defaults:
  - /data: classical
  - /parameters: initial_state
  - /weighting: id_v2
  - /outlier_rejection: residual_threshold
  - /logging: default
  - _self_
  - /figures/residuals
  ...
```

### 9.3 Sweep Integration

The existing sweep config (`conf/sweep/params_x_data.yaml`) can be extended:

```yaml
# conf/sweep/params_x_data_weighting.yaml
defaults:
  - /experiment: classic_triton_state
  - _self_

hydra:
  mode: MULTIRUN
  sweeper:
    params:
      parameters: initial_state,state_plus_neptune_GM
      data: classical,classical_voyager
      weighting: id_v2,hybrid_geometric,timeframe
      outlier_rejection: residual_threshold,disabled
```

### 9.4 Config Dataclasses

```python
# src/orbitdet/observations/weighting/configs.py

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WeightingConfig:
    """Configuration for the weighting subsystem."""
    enabled: bool = False
    strategy: str = "id_v2"
    gap_threshold_hours: float = 4.0
    min_sigma_arcsec: float = 0.01
    ra_dec_independent: bool = True


@dataclass(frozen=True)
class OutlierRejectionConfig:
    """Configuration for the outlier rejection subsystem."""
    enabled: bool = False
    strategies: tuple = ()  # list of strategy config dicts


@dataclass(frozen=True)
class DecBiasConfig:
    """Configuration for DEC bias injection."""
    enabled: bool = False
    biases: dict[str, float] = field(default_factory=dict)  # set_id → arcsec
```

---

## 10. Implementation Phases

### Phase 1 — Core (estimated: 3–5 days)

| Component | Files | Dependencies |
|---|---|---|
| `weighting/base.py` | Abstract `WeightStrategy` | None |
| `weighting/strategies.py` | `IDv2Weight`, `HybridGeometricWeight` | `base.py`, `timeframe.py` |
| `weighting/timeframe.py` | `split_observations_into_timeframes` | None |
| `weighting/engine.py` | `WeightEngine` | `strategies.py` |
| `weighting/configs.py` | `WeightingConfig` | None |
| `outlier_rejection/base.py` | Abstract `OutlierStrategy` | None |
| `outlier_rejection/strategies.py` | `ResidualThresholdOutlier` | `base.py` |
| `outlier_rejection/engine.py` | `OutlierEngine` | `strategies.py` |
| `outlier_rejection/configs.py` | `OutlierRejectionConfig` | None |
| Integration | Changes to `solve_least_squares.py` | All above |
| Config files | `conf/weighting/id_v2.yaml`, `conf/weighting/disabled.yaml`, `conf/outlier_rejection/residual_threshold.yaml`, `conf/outlier_rejection/disabled.yaml` | — |

### Phase 2 — Completeness (estimated: 2–3 days)

| Component | Description |
|---|---|
| `preprocessing/dec_bias.py` | DEC bias injection |
| Remaining weight strategies | `ID`, `IDv1`, `TF`, `HybridA`, `HybridG+v2`, `HybridA+v2` |
| Remaining outlier strategies | `EpochFilter`, `SigmaClip` |
| Iterative weight loop | `_run_iterative_weight_loop()` in `solve_least_squares.py` |
| Config files | All remaining YAML configs |

### Phase 3 — Polish (estimated: 2–3 days)

| Component | Description |
|---|---|
| Weight diagnostics visualization | Per-file RMSE bar charts, weight distribution histograms |
| Rejected-epoch visualization | Mark rejected points on residual plots |
| Weight DataFrame export | Save `weights_df` to CSV in output directory |
| Rejected epochs JSON export | Save `rejected_metadata` to JSON in output directory |
| Unit tests | Test each strategy with synthetic data |

---

## 11. Comparison with the External Reference

| Feature | External Repo | NeptuneOD (Proposed) |
|---|---|---|
| **Weight strategies** | 8 strategies in a single function | 8 strategies, each a separate class with registry |
| **Outlier rejection** | 3 mechanisms (residual, epoch, file-level) | 3 mechanisms + composable chaining |
| **DEC bias** | Hardcoded dict in function body | Config-driven via Hydra |
| **Iterative weight loop** | 5-iteration loop in a template function | Config-driven (`recompute_each_iteration`) |
| **Configurability** | Boolean flags in a Python dict | Hydra YAML composition |
| **Extensibility** | Modify the monolithic function | Register a new strategy class |
| **Diagnostics** | Ad-hoc CSV saving | Structured metadata + DataFrame return |
| **Sigma clipping** | 10 mas floor (hardcoded) | Configurable floor per strategy |
| **RA/DEC independence** | Always independent | Configurable |
| **Timeframe splitting** | 4h gap threshold (hardcoded) | Configurable per run |

### 11.1 Key Improvements Over the External Repo

1. **Registry pattern** — adding a new weight strategy requires one decorator and one class, no modification of existing code.
2. **Hydra composition** — weighting and outlier rejection can be mixed and matched freely across experiments and sweeps.
3. **Clean separation** — weighting, outlier rejection, and DEC bias are independent subsystems that can be enabled/disabled independently.
4. **Diagnostics metadata** — every operation returns structured data for downstream analysis, rather than relying on side-effect file writes.
5. **Configurable thresholds** — no hardcoded values; everything is configurable via YAML.

---

## 12. Open Questions & Risks

### 12.1 Questions

1. **Weight application timing:** Should weights be applied to the `ObservationCollection` before creating the `Estimator`, or should they be passed separately to `EstimationInput`? The external repo applies them before estimation, which is the simplest approach.

2. **Iterative weight loop implementation:** The external repo re-runs the full estimation from scratch each iteration. An alternative is to use Tudat's built-in iteration mechanism and only update weights between iterations. This needs investigation.

3. **File-level filtering:** The external repo has a pre-processing step that excludes entire observation files based on mean/std statistics. Should this be included in the outlier rejection subsystem, or kept as a separate data-preparation step?

4. **Weight recomputation after outlier rejection:** When outlier rejection removes observations, should weights be recomputed from the filtered residuals? The external repo does filtering first, then weights once — but the order matters.

### 12.2 Risks

| Risk | Mitigation |
|---|---|
| **Tudat API compatibility** — `set_tabulated_weights` may behave differently than expected | Prototype with a minimal test case first |
| **Performance** — iterating over every observation set for every strategy could be slow for large collections | Profile early; consider vectorized operations where possible |
| **Config complexity** — too many YAML files could confuse users | Provide sensible defaults (`id_v2` + `residual_threshold`); document clearly |
| **Interaction with existing `ObservationDatasetConfig.weight` field** — the existing scalar weight field is unused but could conflict | Remove or repurpose the field; document the migration |

---

*Generated from the design discussion on 2026-09-04.*
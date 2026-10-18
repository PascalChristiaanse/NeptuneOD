"""Layered weighting strategy.

Combines one weight per partitioning layer into a single per-observation
weight.  This is what makes multi-layer grouping useful for weighting: a
primary partition (e.g. pre/post-encounter **sections**) can supply a fixed
sigma, while a secondary partition (e.g. **timeframes**) supplies a
residual-driven weight, and the two are combined by a chosen formula.

Example — Voyager 2 fixed pre/post-encounter sigmas *with* timeframe
clustering:

.. code-block:: yaml

    grouping:
      levels:
        - type: section
          sections:
            - {name: "pre_encounter",  start: -1e30,       end: -326688543.8172541}
            - {name: "post_encounter", start: -326688543.8172541, end: 1e30}
        - type: timeframe
          gap_threshold_hours: 8.0

    weighting:
      strategy:
        type: layered
        combine: geometric
        layers:
          - level: section
            mode: fixed
            sigmas:
              pre_encounter:  {ra: 1.0, dec: 1.0}
              post_encounter: {ra: 4.0, dec: 4.0}
          - level: timeframe
            mode: rmse_descaled

Drop the ``timeframe`` layer to get fixed sigmas *without* clustering.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import tudatpy.estimation.observations as obs

from ..grouping import Group, GroupList
from .base import WeightStrategy
from .dataframe import (
    build_weights_df,
    interleave_weights,
    observable_type_str,
    residuals_array,
    rms,
)
from .registry import register_weight_strategy

logger = logging.getLogger(__name__)

_ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)

#: Supported per-layer modes.
_LAYER_MODES = ("fixed", "rmse", "rmse_descaled")

#: Supported cross-layer combination formulas.
_COMBINE_MODES = ("geometric", "arithmetic")


@register_weight_strategy("layered")
class LayeredWeight(WeightStrategy):
    """Combine per-layer weights into a single per-observation weight.

    Parameters
    ----------
    layers : list[dict]
        One entry per partitioning layer, each with:

        ``level``
            The grouping level to use (``"set"``, ``"section"``,
            ``"timeframe"``) **or** ``layer`` (int, 0-based).
        ``mode``
            One of ``"fixed"``, ``"rmse"``, ``"rmse_descaled"``.
        ``sigmas``
            Required for ``mode: fixed``.  Maps a group **name**
            (e.g. ``"pre_encounter"``) to either a scalar sigma in arcsec or a
            ``{ra, dec}`` mapping.
    combine : str
        How to combine the layers: ``"geometric"`` (default, the thesis
        hybrid formula) or ``"arithmetic"``.
    fallback_mode : str
        Mode used for any observation not covered by a layer
        (default ``"rmse"``).
    """

    def __init__(
        self,
        layers: list[dict[str, Any]] | None = None,
        combine: str = "geometric",
        fallback_mode: str = "rmse",
    ):
        if not layers:
            raise ValueError("LayeredWeight requires a non-empty 'layers' list.")
        if combine not in _COMBINE_MODES:
            raise ValueError(f"'combine' must be one of {_COMBINE_MODES}, got {combine!r}.")

        self._layers: list[dict[str, Any]] = []
        for i, spec in enumerate(layers):
            mode = str(spec.get("mode", "rmse"))
            if mode not in _LAYER_MODES:
                raise ValueError(
                    f"Layer {i}: unknown mode {mode!r}; expected one of {_LAYER_MODES}."
                )
            level = spec.get("level")
            layer_idx = spec.get("layer")
            if level is None and layer_idx is None:
                raise ValueError(f"Layer {i}: must specify either 'level' or 'layer'.")
            if mode == "fixed" and not spec.get("sigmas"):
                raise ValueError(f"Layer {i}: 'fixed' mode requires a 'sigmas' mapping.")
            self._layers.append(
                {
                    "level": level,
                    "layer": int(layer_idx) if layer_idx is not None else None,
                    "mode": mode,
                    "sigmas": dict(spec.get("sigmas") or {}),
                }
            )
        self._combine = combine
        self._fallback_mode = fallback_mode

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _groups_for_layer(groups: GroupList, spec: dict[str, Any]) -> list[Group]:
        if spec["layer"] is not None:
            return groups.by_layer(spec["layer"])
        return groups.by_level(str(spec["level"]))

    @staticmethod
    def _sigma_from_entry(entry: Any) -> tuple[float, float] | None:
        """Resolve a ``fixed`` sigma entry to ``(ra_arcsec, dec_arcsec)``."""
        if entry is None:
            return None
        if isinstance(entry, (int, float)):
            v = float(entry)
            return (v, v) if v > 0 else None
        single = entry.get("sigma") if isinstance(entry, dict) else None
        if single is not None and float(single) > 0:
            return float(single), float(single)
        ra = float(entry.get("ra", 0.0))
        dec = float(entry.get("dec", 0.0))
        if ra > 0 and dec > 0:
            return ra, dec
        return None

    def _layer_weight_array(
        self,
        spec: dict[str, Any],
        layer_groups: list[Group],
        residuals_rad: np.ndarray,
        set_id: str,
        min_sigma_rad: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-observation ``(w_ra, w_dec)`` arrays for one layer."""
        n_obs = residuals_rad.shape[0]
        w_ra = np.full(n_obs, np.nan)
        w_dec = np.full(n_obs, np.nan)

        # Fallback weight (used for observations no group covers).
        set_ra = max(rms(residuals_rad[:, 0]), min_sigma_rad)
        set_dec = max(rms(residuals_rad[:, 1]), min_sigma_rad)
        fb_ra, fb_dec = 1.0 / set_ra**2, 1.0 / set_dec**2

        for g in layer_groups:
            if spec["mode"] == "fixed":
                entry = spec["sigmas"].get(g.name)
                resolved = self._sigma_from_entry(entry)
                if resolved is None:
                    continue
                ra_sigma = max(resolved[0] * _ARCSEC_TO_RAD, min_sigma_rad)
                dec_sigma = max(resolved[1] * _ARCSEC_TO_RAD, min_sigma_rad)
                gr_ra, gr_dec = 1.0 / ra_sigma**2, 1.0 / dec_sigma**2
            else:
                ra_rms = rms(residuals_rad[g.indices, 0])
                dec_rms = rms(residuals_rad[g.indices, 1])
                scale = np.sqrt(g.size) if spec["mode"] == "rmse_descaled" else 1.0
                ra_sigma = max(ra_rms, min_sigma_rad) * scale
                dec_sigma = max(dec_rms, min_sigma_rad) * scale
                gr_ra, gr_dec = 1.0 / ra_sigma**2, 1.0 / dec_sigma**2
            for idx in g.indices:
                w_ra[idx] = gr_ra
                w_dec[idx] = gr_dec

        # Fill uncovered observations with the per-observation fallback.
        missing = ~np.isfinite(w_ra)
        if np.any(missing):
            if self._fallback_mode == "rmse_descaled":
                # Descaled over the whole set is ill-defined; reuse set RMSE.
                w_ra[missing], w_dec[missing] = fb_ra, fb_dec
            else:
                w_ra[missing], w_dec[missing] = fb_ra, fb_dec
        return w_ra, w_dec

    def _combine_arrays(self, arrays: list[np.ndarray]) -> np.ndarray:
        if len(arrays) == 1:
            return arrays[0]
        stacked = np.vstack(arrays)
        if self._combine == "arithmetic":
            return np.mean(stacked, axis=0)
        # Geometric mean = exp(mean(log(w))); weights are strictly positive.
        return np.exp(np.mean(np.log(stacked), axis=0))

    # ------------------------------------------------------------------
    # WeightStrategy API
    # ------------------------------------------------------------------

    def compute_weights(
        self,
        observation_set: obs.SingleObservationSet,
        groups: GroupList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        min_sigma_rad = min_sigma_arcsec * _ARCSEC_TO_RAD
        residuals_rad = residuals_array(observation_set)
        n_obs = residuals_rad.shape[0]
        times = np.array([t.to_float() for t in observation_set.observation_times])

        if n_obs == 0:
            return np.array([], dtype=float), pd.DataFrame()

        layer_ra: list[np.ndarray] = []
        layer_dec: list[np.ndarray] = []
        for spec in self._layers:
            layer_groups = self._groups_for_layer(groups, spec)
            if not layer_groups:
                logger.warning(
                    "LayeredWeight [%s]: layer %s/%s matched no groups; skipping.",
                    set_id,
                    spec.get("level"),
                    spec.get("layer"),
                )
                continue
            w_ra, w_dec = self._layer_weight_array(
                spec, layer_groups, residuals_rad, set_id, min_sigma_rad
            )
            layer_ra.append(w_ra)
            layer_dec.append(w_dec)

        if not layer_ra:
            raise ValueError(
                f"LayeredWeight [{set_id}]: no layer matched any groups. "
                f"Check the grouping config levels against the layer specs."
            )

        weights_ra_arr = self._combine_arrays(layer_ra)
        weights_dec_arr = self._combine_arrays(layer_dec)

        weights_array = interleave_weights(weights_ra_arr, weights_dec_arr)

        # Report against the finest layer used, so group_id is meaningful.
        report_level = "timeframe"
        if self._layers:
            last = self._layers[-1]
            report_level = str(last["level"]) if last["level"] else "timeframe"

        df = build_weights_df(
            n_obs,
            times,
            residuals_rad,
            groups,
            weights_ra_arr,
            weights_dec_arr,
            "layered",
            set_id,
            report_level,
            observable_type=observable_type_str(observation_set),
        )

        logger.debug(
            "LayeredWeight [%s]: %d layer(s), combine=%s, RA w range=[%.2e, %.2e]",
            set_id,
            len(layer_ra),
            self._combine,
            weights_ra_arr.min(),
            weights_ra_arr.max(),
        )
        return weights_array, df

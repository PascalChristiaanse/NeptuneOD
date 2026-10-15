"""Engine that orchestrates weighting over an ObservationCollection.

The :class:`WeightEngine` applies a :class:`WeightStrategy` to every
observation set in a collection, optionally partitioned by grouping levels,
and assigns the resulting weights via ``set_tabulated_weights``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import tudatpy.dynamics.environment as env
import tudatpy.estimation.observations as obs
from omegaconf import DictConfig, OmegaConf

from ..grouping import build_group_list
from ..utils import get_set_identifier
from .base import WeightStrategy
from .builder import build_strategy

logger = logging.getLogger(__name__)


class WeightEngine:
    """Orchestrates weighting over an ObservationCollection.

    Parameters
    ----------
    strategy : WeightStrategy
        The weighting strategy to apply.
    grouping : DictConfig | None
        Grouping configuration (defines levels like sections, timeframes).
        If None, each set is treated as a single group.
    min_sigma_arcsec : float
        Floor on sigma in arcseconds.
    """

    def __init__(
        self,
        strategy: WeightStrategy,
        grouping: DictConfig | None = None,
        min_sigma_arcsec: float = 0.01,
    ):
        self._strategy = strategy
        self._grouping = grouping
        self._min_sigma_arcsec = min_sigma_arcsec

    @property
    def strategy(self) -> WeightStrategy:
        """The weighting strategy (read-only)."""
        return self._strategy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        collection: obs.ObservationCollection,
        bodies: env.SystemOfBodies,
        output_dir: str | Path | None = None,
        dataset_metadata: dict[str, list[dict]] | None = None,
    ) -> tuple[obs.ObservationCollection, pd.DataFrame]:
        """Compute and assign weights to all observation sets.

        Parameters
        ----------
        collection : ObservationCollection
            The observation collection to weight.  Residuals must be populated.
        bodies : SystemOfBodies
            The system of bodies (passed through to strategies).
        output_dir : str | Path, optional
            If provided, the per-observation weights DataFrame is saved to
            ``<output_dir>/observation_weights.csv``.
        dataset_metadata : dict[str, list[dict]], optional
            Mapping from set_id to a list of metadata dicts (one per dataset
            config that produced observations for this observatory). Each dict
            has keys ``name`` (source author), ``dataset_id`` (e.g. ``nm0008``),
            ``observatory_code``, ``ra_type``, and ``dec_type``.
            Metadata entries are consumed FIFO per set_id.

        Returns
        -------
        tuple[ObservationCollection, pd.DataFrame]
            - The weighted ``ObservationCollection`` (modified in place).
            - A DataFrame with per-observation weight metadata.
        """
        all_sets = collection.get_single_observation_sets()
        logger.info(
            "WeightEngine: applying '%s' strategy to %d observation set(s)",
            self._strategy.__class__.__name__,
            len(all_sets),
        )

        all_dfs: list[pd.DataFrame] = []
        meta_queues: dict[str, list[dict]] = {}
        if dataset_metadata:
            meta_queues = {k: list(v) for k, v in dataset_metadata.items()}

        for obs_set in all_sets:
            set_id = get_set_identifier(obs_set)
            times = np.array([t.to_float() for t in obs_set.observation_times])
            n_obs = len(times)

            if n_obs == 0:
                logger.debug("WeightEngine: skipping empty set '%s'", set_id)
                continue

            # Build groups from the grouping config
            groups = build_group_list(times, set_id, self._grouping)
            logger.debug(
                "WeightEngine [%s]: %d groups across levels: %s",
                set_id,
                len(groups),
                {g.level for g in groups},
            )

            # Compute weights
            weights_array, weights_df = self._strategy.compute_weights(
                obs_set,
                groups,
                set_id,
                self._min_sigma_arcsec,
            )

            if len(weights_array) == 0:
                logger.warning("WeightEngine: no weights computed for set '%s'", set_id)
                continue

            # A router may preserve the full matrix only for certain routes
            # (e.g. Gaia).  Prefer the per-set answer when the strategy offers
            # one, falling back to the top-level property.
            preserve = getattr(self._strategy, "preserve_full_weight_matrix", False)
            route_source = getattr(self._strategy, "route_source", None)
            if callable(route_source):
                try:
                    sub = route_source(set_id)
                    preserve = getattr(sub, "preserve_full_weight_matrix", preserve)
                except Exception:  # pragma: no cover - diagnostics only
                    pass

            if getattr(obs_set, "has_full_weight_matrix_contribution", False) and preserve:
                # The set carries a full (off-diagonal) archive weight matrix (e.g. the Gaia
                # transit covariance).  Tudat adds the tabulated base weights on top of the full
                # matrix, so applying the strategy weights here would double-count them.  Leave
                # the full matrix untouched and only record the strategy weights for reporting.
                logger.info(
                    "WeightEngine [%s]: full archive weight matrix present; leaving weights "
                    "untouched (strategy '%s' recorded for reporting only).",
                    set_id,
                    self._strategy.__class__.__name__,
                )
            else:
                # Clear any pre-existing full weight-matrix contribution so it does not stack
                # additively on top of the base weights we are about to assign.
                if getattr(obs_set, "has_full_weight_matrix_contribution", False):
                    try:
                        obs_set.set_full_weight_matrix(np.zeros((0, 0)))
                    except Exception as exc:
                        logger.warning(
                            "WeightEngine [%s]: could not clear full weight matrix: %s",
                            set_id,
                            exc,
                        )

                # Assign weights via Tudat's set_tabulated_weights
                try:
                    obs_set.set_tabulated_weights(weights_array)
                except Exception as exc:
                    logger.error(
                        "WeightEngine: failed to set weights for set '%s': %s", set_id, exc
                    )
                    raise

            # Enrich with dataset metadata — pop FIFO per set_id
            meta_queue = meta_queues.get(set_id, [])
            if meta_queue:
                meta = meta_queue.pop(0)
            else:
                meta = None

            if meta:
                weights_df["source_name"] = meta.get("name", set_id)
                weights_df["dataset_id"] = meta.get("dataset_id", set_id)
                weights_df["observatory_code"] = meta.get("observatory_code", set_id)
                weights_df["ra_type"] = meta.get("ra_type", "")
                weights_df["dec_type"] = meta.get("dec_type", "")
            else:
                weights_df["source_name"] = set_id
                weights_df["dataset_id"] = set_id
                weights_df["observatory_code"] = set_id
                weights_df["ra_type"] = ""
                weights_df["dec_type"] = ""

            all_dfs.append(weights_df)

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            logger.info(
                "WeightEngine: assigned weights to %d observations",
                len(combined_df),
            )

            # Save to CSV if output_dir is provided
            if output_dir is not None:
                output_path = Path(output_dir) / "observation_weights.csv"
                combined_df.to_csv(output_path, index=False)
                logger.info("WeightEngine: weights saved to %s", output_path)
        else:
            combined_df = pd.DataFrame()

        return collection, combined_df

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls, cfg: DictConfig, default_grouping: DictConfig | None = None
    ) -> WeightEngine:
        """Build an engine from a Hydra configuration node.

        Two configuration forms are supported.

        **Legacy (named strategy):**::

            weighting:
              strategy: id_v2
              min_sigma_arcsec: 0.01
              grouping:
                levels: [...]

        **Descriptor (recommended)** — strategy options are given inline,
        which removes the need to edit the factory for every strategy::

            weighting:
              strategy:
                type: layered
                combine: geometric
                layers:
                  - {level: section, mode: fixed, sigmas: {...}}
                  - {level: timeframe, mode: rmse_descaled}

        Parameters
        ----------
        cfg : DictConfig
            The configuration node (typically ``cfg.weighting``).

        Returns
        -------
        WeightEngine
            An engine with the configured strategy and grouping.
        """
        strategy_node = OmegaConf.select(cfg, "strategy", default="id_v2")
        min_sigma = float(OmegaConf.select(cfg, "min_sigma_arcsec", default=0.01))
        grouping = OmegaConf.select(cfg, "grouping") or default_grouping

        # Descriptor form: {type: <name>, ...kwargs}
        if isinstance(strategy_node, Mapping) and "type" in strategy_node:
            strategy_instance = build_strategy(strategy_node, grouping=grouping)
        else:
            # Bare strategy name (e.g. `strategy: id_v2`).  Strategy-specific
            # options, if any, live under a sibling node named after the
            # strategy (e.g. ``weighting.gaia_archive``).
            strategy_name = str(strategy_node)
            node = OmegaConf.select(cfg, strategy_name)
            params = (
                {k: v for k, v in node.items()}
                if node is not None and hasattr(node, "items")
                else {}
            )
            strategy_instance = build_strategy({"type": strategy_name, **params}, grouping=grouping)

        return cls(strategy_instance, grouping=grouping, min_sigma_arcsec=min_sigma)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


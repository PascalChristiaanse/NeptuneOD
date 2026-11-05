"""Engine that orchestrates weighting over an ObservationCollection.

The :class:`WeightEngine` applies a :class:`WeightStrategy` to every
observation set in a collection, optionally partitioned by clustering levels,
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

from ..clustering import build_cluster_list
from ..utils import get_set_identifier
from .base import WeightStrategy
from .builder import build_strategy

logger = logging.getLogger(__name__)


def _view_value(view, column: str, default):
    """Return one standard-frame column for a set, or *default* without a view.

    The weights report is descriptive, so a missing view or a column the source
    did not populate degrades to the set identifier rather than raising.
    """
    if view is None:
        return default
    try:
        values = view.column(column)
    except KeyError:
        return default
    if len(values) == 0:
        return default
    return values[0]


def _coordinate_type_label(coordinate_kind: str) -> tuple[str, str]:
    """Map a ``coordinate_kind`` value to human-readable RA/DEC axis labels.

    These are presentation strings for the weight summary tables, so they live
    here rather than in the frame schema: ``coordinate_kind`` already carries the
    semantic content, and the LaTeX markup is a plotting concern.

    Parameters
    ----------
    coordinate_kind : str
        The standard frame's ``coordinate_kind`` value (``"ra_dec"``,
        ``"position_angle_separation"``, ``"cartesian_xy"``, ...).

    Returns
    -------
    tuple[str, str]
        ``(first_label, second_label)``, empty strings when the kind is unknown.
    """
    kind = str(coordinate_kind).lower()
    if "position_angle" in kind:
        return (r"$\rho$", r"$\theta$")
    if "cartesian" in kind:
        return ("$x$", "$y$")
    if "ra_dec" in kind:
        return (r"$\alpha$", r"$\delta$")
    return ("", "")


class WeightEngine:
    """Orchestrates weighting over an ObservationCollection.

    Parameters
    ----------
    strategy : WeightStrategy
        The weighting strategy to apply.
    clustering : DictConfig | None
        Grouping configuration (defines levels like sections, timeframes).
        If None, each set is treated as a single cluster.
    min_sigma_arcsec : float
        Floor on sigma in arcseconds.
    """

    def __init__(
        self,
        strategy: WeightStrategy,
        clustering: DictConfig | None = None,
        min_sigma_arcsec: float = 0.01,
    ):
        self._strategy = strategy
        self._clustering = clustering
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
        access=None,
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
        access : ObservationAccess, optional
            The access layer for this collection.  When provided, each set is
            wrapped in a :class:`~orbitdet.analysis.set_view.SetView` so
            strategies can read standard-frame columns (notably ``scan_angle``)
            and the weights report can carry the set's provenance
            (``source_set_id``, ``observatory_code``, ...).  When omitted, the
            strategies run without a view and must use their frame-free paths,
            and the report falls back to the set identifier.

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

        for set_index, obs_set in enumerate(all_sets):
            set_id = get_set_identifier(obs_set)
            times = np.array([t.to_float() for t in obs_set.observation_times])
            n_obs = len(times)

            if n_obs == 0:
                logger.debug("WeightEngine: skipping empty set '%s'", set_id)
                continue

            # The set's window onto the access layer, so strategies can read
            # standard-frame columns (scan_angle, observatory_code, ...).
            view = None
            if access is not None:
                # Imported here: ``orbitdet.analysis`` imports this package.
                from ...analysis.set_view import SetView

                view = SetView.build(access, set_index)

            # Build clusters from the clustering config
            clusters = build_cluster_list(times, set_id, self._clustering)
            logger.debug(
                "WeightEngine [%s]: %d clusters across levels: %s",
                set_id,
                len(clusters),
                {g.level for g in clusters},
            )

            # Compute weights
            weights_array, weights_df = self._strategy.compute_weights(
                obs_set,
                clusters,
                set_id,
                self._min_sigma_arcsec,
                view=view,
            )

            # Strategies whose weights are not diagonal in (RA, Dec) (e.g. fixed
            # along/across-scan weights) return per-observation 2x2 blocks.
            block_weights = None
            compute_blocks = getattr(self._strategy, "compute_block_weights", None)
            if callable(compute_blocks):
                block_weights = compute_blocks(
                    obs_set,
                    clusters,
                    set_id,
                    self._min_sigma_arcsec,
                    view=view,
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
                    if block_weights is not None:
                        # Non-diagonal (in RA/Dec) weights: one 2x2 block per observation.
                        obs_set.set_block_diagonal_weights(block_weights)
                    else:
                        obs_set.set_tabulated_weights(weights_array)
                except Exception as exc:
                    logger.error(
                        "WeightEngine: failed to set weights for set '%s': %s", set_id, exc
                    )
                    raise

            # Enrich the report with the set's provenance, read from the standard
            # frame through the view.  Reading by set index (rather than popping a
            # queue keyed by set_id) keeps the attribution correct when several
            # datasets observed from the same station: the merge reorders sets, so
            # a queue would hand a set another dataset's metadata.
            weights_df["source_name"] = _view_value(view, "reference", set_id)
            weights_df["dataset_id"] = _view_value(view, "source_set_id", set_id)
            weights_df["observatory_code"] = _view_value(view, "observatory_code", set_id)
            ra_type, dec_type = _coordinate_type_label(
                _view_value(view, "coordinate_kind", "")
            )
            weights_df["ra_type"] = ra_type
            weights_df["dec_type"] = dec_type

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
        cls, cfg: DictConfig, default_clustering: DictConfig | None = None
    ) -> WeightEngine:
        """Build an engine from a Hydra configuration node.

        Two configuration forms are supported.

        **Legacy (named strategy):**::

            weighting:
              strategy: id_v2
              min_sigma_arcsec: 0.01
              clustering:
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
            An engine with the configured strategy and clustering.
        """
        strategy_node = OmegaConf.select(cfg, "strategy", default="id_v2")
        min_sigma = float(OmegaConf.select(cfg, "min_sigma_arcsec", default=0.01))
        clustering = OmegaConf.select(cfg, "clustering") or default_clustering

        # Descriptor form: {type: <name>, ...kwargs}
        if isinstance(strategy_node, Mapping) and "type" in strategy_node:
            strategy_instance = build_strategy(strategy_node, clustering=clustering)
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
            strategy_instance = build_strategy({"type": strategy_name, **params}, 
                clustering=clustering)

        return cls(strategy_instance, clustering=clustering, min_sigma_arcsec=min_sigma)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
